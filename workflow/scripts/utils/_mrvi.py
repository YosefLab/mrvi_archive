from mrvi import MrVI
from ._base_model import BaseModelClass
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm


class MrVIWrapper(BaseModelClass):
    has_cell_representation = True
    has_donor_representation = True
    has_local_donor_representation = True
    has_save = True

    default_model_kwargs = dict(
        observe_library_sizes=True,
        n_latent_donor=5,
    )
    default_train_kwargs = dict(
        max_epochs=100,
        check_val_every_n_epoch=1,
        batch_size=256,
        plan_kwargs=dict(lr=1e-2, n_epochs_kl_warmup=20),
    )
    model_name = "mrvi"

    def __init__(self, model_kwargs=None, train_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs = self.default_model_kwargs.copy()
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)
        self.train_kwargs = self.default_train_kwargs.copy()
        if train_kwargs is not None:
            self.train_kwargs.update(train_kwargs)

    def fit(self):
        self.preprocess_data()
        adata_ = self.adata.copy()
        self.adata_ = adata_

        MrVI.setup_anndata(
            adata_,
            batch_key=self.donor_key,
            categorical_nuisance_keys=self.categorical_nuisance_keys,
            categorical_biological_keys=None,
        )
        self.model = MrVI(adata=adata_, **self.model_kwargs)
        self.model.train(**self.train_kwargs)
        return True

    def save(self, save_path, overwrite=True):
        self.model.save(save_path, overwrite=overwrite)
        return True

    @classmethod
    def load(self, adata, save_path):
        mapper = adata.uns["mapper"]
        cls = MrVIWrapper(adata=adata, **mapper)
        cls.model = MrVI.load(save_path, adata)
        adata_ = cls.adata.copy()
        cls.adata_ = adata_
        return cls

    def get_donor_representation(self):
        d_embeddings = self.model.module.donor_embeddings.weight.cpu().detach().numpy()
        index_ = (
            self.adata_.obs.drop_duplicates("_scvi_batch")
            .set_index("_scvi_batch")
            .sort_index()
            .loc[:, self.donor_key]
        )
        return pd.DataFrame(d_embeddings, index=index_)

    def get_cell_representation(self, adata=None, batch_size=512, give_z=False):
        return self.model.get_latent_representation(
            adata, batch_size=batch_size, give_z=give_z
        )

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata=None,
        x_log=True,
        batch_size=256,
        eps=1e-6,
        cf_site=0.0,
    ):
        adata = self.adata_ if adata is None else adata
        self.model._check_if_trained(warn=False)
        adata = self.model._validate_anndata(adata)
        scdl = self.model._make_data_loader(
            adata=adata, indices=None, batch_size=batch_size
        )

        reps = []
        for tensors in tqdm(scdl):
            xs = []
            if cf_site is not None:
                tensors[
                    "categorical_nuisance_keys"
                ] *= cf_site  # set to 0 all nuisance factors
            inference_inputs = self.model.module._get_inference_input(tensors)
            outputs_n = self.model.module.inference(use_mean=True, **inference_inputs)
            outs_g = self.model.module.generative(
                **self.model.module._get_generative_input(
                    tensors, inference_outputs=outputs_n
                )
            )
            xs = outs_g["h"]
            if x_log:
                xs = (eps + xs).log()
            reps.append(xs.cpu().numpy())
        # n_cells, n_donors, n_donors
        reps = np.concatenate(reps, 0)
        return reps

    @torch.no_grad()
    def get_average_expression(
        self,
        adata=None,
        indices=None,
        batch_size=256,
        eps=1e-6,
        mc_samples=10,
    ):
        adata = self.adata_ if adata is None else adata
        # self.model._check_if_trained(warn=False)
        # adata = self.model._validate_anndata(adata)
        scdl = self.model._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        reps = np.zeros((self.model.summary_stats.n_batch, adata.n_vars))
        for tensors in tqdm(scdl):
            xs = []
            for batch in range(self.model.summary_stats.n_batch):
                tensors[
                    "categorical_nuisance_keys"
                ] *= 0.0  # set to 0 all nuisance factors

                cf_batch = batch * torch.ones_like(tensors["batch"])
                inference_inputs = self.model.module._get_inference_input(tensors)
                inference_outputs = self.model.module.inference(
                    n_samples=mc_samples, cf_batch=cf_batch, **inference_inputs
                )
                generative_inputs = self.model.module._get_generative_input(
                    tensors=tensors, inference_outputs=inference_outputs
                )
                generative_outputs = self.model.module.generative(**generative_inputs)
                new = generative_outputs["h"]
                new = (eps + generative_outputs["h"]).log()
                xs.append(new[:, :, None])

            xs = torch.cat(xs, 2).mean(0)  # size (n_cells, n_donors, n_genes)
            reps += xs.mean(0).cpu().numpy()
        return reps

    @torch.no_grad()
    def get_local_donor_representation(self, adata=None, **kwargs):
        return self.model.get_local_donor_representation(adata=adata, **kwargs)

    def get_donor_representation_metadata(self):
        return (
            self.model.adata.obs.drop_duplicates("_scvi_batch")
            .set_index("_scvi_batch")
            .sort_index()
        )
