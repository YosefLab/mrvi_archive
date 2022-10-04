from copy import deepcopy
import logging
from typing import List, Optional, Union

from anndata import AnnData
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
)
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.model.base import BaseModelClass, VAEMixin
from scvi.model._utils import _init_library_size

from ._module import MrVAE

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_KWARGS = dict(
    early_stopping=True,
    early_stopping_patience=15,
    check_val_every_n_epoch=1,
    batch_size=256,
    train_size=0.9,
    plan_kwargs=dict(
        lr=1e-2,
        n_epochs_kl_warmup=20,
        do_comp=False,
        lambd=0.1,
    ),
)


class MrVI(UnsupervisedTrainingMixin, VAEMixin, BaseModelClass):
    def __init__(
        self,
        adata,
        **model_kwargs,
    ):
        super().__init__(adata)
        n_cats_per_nuisance_keys = (
            self.adata_manager.get_state_registry(
                "categorical_nuisance_keys"
            ).n_cats_per_key
            if "categorical_nuisance_keys" in self.adata_manager.data_registry
            else []
        )

        n_cats_per_bio_keys = (
            self.adata_manager.get_state_registry(
                "categorical_biological_keys"
            ).n_cats_per_key
            if "categorical_biological_keys" in self.adata_manager.data_registry
            else []
        )
        n_batch = self.summary_stats.n_batch
        library_log_means, library_log_vars = _init_library_size(
            self.adata_manager, n_batch
        )
        # n_obs_per_batch = adata.obs.groupby(self.adata_manager.get_state_registry("batch")["original_key"]).size()
        n_obs_per_batch = (
            adata.obs.groupby(
                self.adata_manager.get_state_registry("batch")["original_key"]
            )
            .size()
            .loc[self.adata_manager.get_state_registry("batch")["categorical_mapping"]]
            .values
        )
        n_obs_per_batch = torch.from_numpy(n_obs_per_batch).float()
        self.data_splitter = None
        self.module = MrVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_cats_per_nuisance_keys=n_cats_per_nuisance_keys,
            n_cats_per_bio_keys=n_cats_per_bio_keys,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            n_obs_per_batch=n_obs_per_batch,
            **model_kwargs,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        categorical_nuisance_keys: Optional[List[str]] = None,
        categorical_biological_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            CategoricalJointObsField(
                "categorical_nuisance_keys", categorical_nuisance_keys
            ),
            CategoricalJointObsField(
                "categorical_biological_keys", categorical_biological_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        train_kwargs = dict(
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            early_stopping=early_stopping,
            **trainer_kwargs,
        )
        train_kwargs = dict(deepcopy(DEFAULT_TRAIN_KWARGS), **train_kwargs)
        plan_kwargs = plan_kwargs or {}
        train_kwargs["plan_kwargs"] = dict(
            deepcopy(DEFAULT_TRAIN_KWARGS["plan_kwargs"]), **plan_kwargs
        )
        super().train(**train_kwargs)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices=None,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        give_z: bool = False,
    ) -> np.ndarray:
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        u = []
        z = []
        for tensors in tqdm(scdl):
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(n_samples=mc_samples, **inference_inputs)
            u.append(outputs["u"].mean(0).cpu())
            z.append(outputs["z"].mean(0).cpu())

        u = torch.cat(u, 0).numpy()
        z = torch.cat(z, 0).numpy()
        return z if give_z else u

    def get_cf_degs(
        self,
        adata: Optional[AnnData] = None,
        indices=None,
        batch_size: Optional[int] = None,
    ):

        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        cf_degs = []
        for tensors in tqdm(scdl):
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            cf_degs.append(outputs["cf_degs"].cpu().numpy())

        cf_degs = np.concatenate(cf_degs, 0)
        return cf_degs

    @torch.no_grad()
    def get_local_donor_representation(
        self,
        adata=None,
        batch_size=256,
        mc_samples: int = 10,
        x_space=False,
        x_log=True,
        x_dim=50,
        eps=1e-6,
    ):
        """Computes the local donor representation of the cells in the adata object.
        For each cell, it returns a matrix of size (n_donors, n_features)
        """
        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=None, batch_size=batch_size)

        if x_space & (x_dim is not None):
            hs = self.get_normalized_expression(
                adata, batch_size=batch_size, eps=eps, x_log=x_log
            )
            means = np.mean(hs, axis=0)
            stds = np.std(hs, axis=0)
            hs = (hs - means) / stds
            pca = PCA(n_components=x_dim).fit(hs)
            w = torch.tensor(pca.components_, dtype=torch.float32, device=self.device).T
            means = torch.tensor(means, dtype=torch.float32, device=self.device)
            stds = torch.tensor(stds, dtype=torch.float32, device=self.device)

        reps = []
        for tensors in tqdm(scdl):
            xs = []
            for batch in range(self.summary_stats.n_batch):
                if x_space:
                    tensors[
                        "categorical_nuisance_keys"
                    ] *= 0.0  # set to 0 all nuisance factors

                    cf_batch = batch * torch.ones_like(tensors["batch"])
                    inference_inputs = self.module._get_inference_input(tensors)
                    inference_outputs = self.module.inference(
                        n_samples=mc_samples, cf_batch=cf_batch, **inference_inputs
                    )
                    generative_inputs = self.module._get_generative_input(
                        tensors=tensors, inference_outputs=inference_outputs
                    )
                    generative_outputs = self.module.generative(**generative_inputs)
                    new = generative_outputs["h"]
                    if x_log:
                        new = (eps + generative_outputs["h"]).log()
                    if x_dim is not None:
                        new = (new - means) / stds
                        new = new @ w
                else:
                    cf_batch = batch * torch.ones_like(tensors["batch"])
                    inference_inputs = self.module._get_inference_input(tensors)
                    inference_outputs = self.module.inference(
                        n_samples=mc_samples, cf_batch=cf_batch, **inference_inputs
                    )
                    new = inference_outputs["z"]

                xs.append(new[:, :, None])

            xs = torch.cat(xs, 2).mean(0)
            reps.append(xs.cpu().numpy())
        # n_cells, n_donors, n_donors
        reps = np.concatenate(reps, 0)
        return reps
