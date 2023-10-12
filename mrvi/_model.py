from __future__ import annotations

import logging
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalJointObsField, CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from ._constants import MRVI_REGISTRY_KEYS
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
    ),
)


class MrVI(UnsupervisedTrainingMixin, VAEMixin, BaseModelClass):
    """
    Multi-resolution Variational Inference (MrVI) :cite:`Boyeau2022mrvi`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via
        :meth:`~scvi.model.MrVI.setup_anndata`.
    n_latent
        Dimensionality of the latent space.
    n_latent_donor
        Dimensionality of the latent space for sample embeddings.
    linear_decoder_zx
        Whether to use a linear decoder for the decoder from z to x.
    linear_decoder_uz
        Whether to use a linear decoder for the decoder from u to z.
    linear_decoder_uz_scaler
        Whether to incorporate a learned scaler term in the decoder from u to z.
    linear_decoder_uz_scaler_n_hidden
        If `linear_decoder_uz_scaler` is True, the number of hidden
        units in the neural network used to produce the scaler term
        in decoder from u to z.
    px_kwargs
        Keyword args for :class:`~mrvi.components.DecoderZX`.
    pz_kwargs
        Keyword args for :class:`~mrvi.components.DecoderUZ`.
    """

    def __init__(
        self,
        adata: AnnData,
        **model_kwargs,
    ):
        super().__init__(adata)
        n_cats_per_nuisance_keys = (
            self.adata_manager.get_state_registry(
                MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS
            ).n_cats_per_key
            if MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS
            in self.adata_manager.data_registry
            else []
        )

        n_sample = self.summary_stats.n_sample
        n_obs_per_sample = (
            adata.obs.groupby(
                self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY)[
                    "original_key"
                ]
            )
            .size()
            .loc[
                self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY)[
                    "categorical_mapping"
                ]
            ]
            .values
        )
        n_obs_per_sample = torch.from_numpy(n_obs_per_sample).float()
        self.data_splitter = None
        self.module = MrVAE(
            n_input=self.summary_stats.n_vars,
            n_sample=n_sample,
            n_obs_per_sample=n_obs_per_sample,
            n_cats_per_nuisance_keys=n_cats_per_nuisance_keys,
            **model_kwargs,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        sample_key: Optional[str] = None,
        categorical_nuisance_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(MRVI_REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            CategoricalJointObsField(
                MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS, categorical_nuisance_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        train_kwargs = dict(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
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
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        mc_samples: int = 5000,
        batch_size: int | None = None,
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
            outputs = self.module.inference(mc_samples=mc_samples, **inference_inputs)
            u.append(outputs["u"].mean(0).cpu())
            z.append(outputs["z"].mean(0).cpu())

        u = torch.cat(u, 0).numpy()
        z = torch.cat(z, 0).numpy()
        return z if give_z else u

    @staticmethod
    def compute_distance_matrix_from_representations(
        representations: np.ndarray, metric: str = "euclidean"
    ) -> np.ndarray:
        """
        Compute distance matrices from counterfactual sample representations.

        Parameters
        ----------
        representations
            Counterfactual sample representations of shape
            (n_cells, n_sample, n_features).
        metric
            Metric to use for computing distance matrix.
        """
        n_cells, n_donors, _ = representations.shape
        pairwise_dists = np.zeros((n_cells, n_donors, n_donors))
        for i, cell_rep in enumerate(representations):
            d_ = pairwise_distances(cell_rep, metric=metric)
            pairwise_dists[i, :, :] = d_
        return pairwise_dists

    @torch.no_grad()
    def get_local_sample_representation(
        self,
        adata: AnnData | None = None,
        batch_size: int = 256,
        mc_samples: int = 10,
        return_distances: bool = False,
    ):
        """
        Computes the local sample representation of the cells in the adata object.

        For each cell, it returns a matrix of size (n_sample, n_features).

        Parameters
        ----------
        adata
            AnnData object to use for computing the local sample representation.
        batch_size
            Batch size to use for computing the local sample representation.
        mc_samples
            Number of Monte Carlo samples to use for computing the local sample
            representation.
        return_distances
            If ``return_distances`` is ``True``, returns a distance matrix of
            size (n_sample, n_sample) for each cell.
        """
        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=None, batch_size=batch_size)

        reps = []
        for tensors in tqdm(scdl):
            xs = []
            for sample in range(self.summary_stats.n_sample):
                cf_sample = sample * torch.ones_like(
                    tensors[MRVI_REGISTRY_KEYS.SAMPLE_KEY]
                )
                inference_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(
                    mc_samples=mc_samples, cf_sample=cf_sample, **inference_inputs
                )
                new = inference_outputs["z"]

                xs.append(new[:, :, None])

            xs = torch.cat(xs, 2).mean(0)
            reps.append(xs.cpu().numpy())
        # n_cells, n_sample, n_latent
        reps = np.concatenate(reps, 0)

        if return_distances:
            return self.compute_distance_matrix_from_representations(reps)

        return reps
