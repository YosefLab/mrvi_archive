import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scanpy as sc
from scvi.model import SCVI
import torch
import pynndescent
from tqdm import tqdm
from scvi import REGISTRY_KEYS


from ._base_model import BaseModelClass


class CTypeProportions(BaseModelClass):
    has_cell_representation = False
    has_donor_representation = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_donor_representation(self, adata=None):
        adata = self.adata if adata is None else adata
        ct_props = (
            self.adata.obs.groupby(self.donor_key)[self.cell_type_key]
            .value_counts()
            .unstack()
            .fillna(0.0)
            .apply(lambda x: x / x.sum(), axis=1)
        )
        return ct_props


class PseudoBulkPCA(BaseModelClass):
    has_cell_representation = False
    has_donor_representation = True

    def __init__(self, n_components=50, **kwargs):
        super().__init__(**kwargs)
        self.n_components = np.minimum(n_components, self.n_donors)

    def get_donor_representation(self, adata=None):
        adata = self.adata if adata is None else adata
        idx_donor = adata.obs[self.donor_key]

        X = np.zeros((self.n_donors, self.n_genes))
        unique_donors = idx_donor.unique()
        for idx, unique_donor in enumerate(unique_donors):
            cell_is_selected = idx_donor == unique_donor
            X[idx, :] = adata.X[cell_is_selected].sum(0)
        X_cpm = 1e6 * X / X.sum(-1, keepdims=True)
        X_logcpm = np.log1p(X_cpm)
        z = PCA(n_components=self.n_components).fit_transform(X_logcpm)
        return pd.DataFrame(z, index=unique_donors)


class StatifiedPseudoBulkPCA(BaseModelClass):
    has_cell_representation = False
    has_donor_representation = False
    has_local_donor_representation = True

    def __init__(self, n_components=50, **kwargs):
        super().__init__(**kwargs)
        self.n_components = np.minimum(n_components, self.n_donors)
        self.cell_type_key = None

    def fit(self):
        pass

    def get_local_donor_representation(self, adata=None):
        self.cell_type_key = self.adata.uns["mapper"]["cell_type_key"]
        adata = self.adata
        idx_donor = adata.obs[self.donor_key]
        cell_types = adata.obs[self.cell_type_key]

        X = []
        unique_donors = idx_donor.unique()
        unique_types = cell_types.unique()
        reps_all = dict()
        for cell_type in unique_types:
            X = []
            donors = []
            # Computing the pseudo-bulk for each cell type
            for unique_donor in unique_donors:
                cell_is_selected = (idx_donor == unique_donor) & (
                    cell_types == cell_type
                )
                new_counts = adata.X[cell_is_selected].sum(0)
                if new_counts.sum() > 0:
                    X.append(new_counts)
                    donors.append(unique_donor)
            X = np.array(X).squeeze(1)
            donors = np.array(donors).astype(str)
            X_cpm = 1e6 * X / X.sum(-1, keepdims=True)
            X_logcpm = np.log1p(X_cpm)
            n_comps = np.minimum(self.n_components, X.shape[0])
            z = PCA(n_components=n_comps).fit_transform(X_logcpm)
            reps = pd.DataFrame(z, index=donors)
            reps.columns = ["PC_{}".format(i) for i in range(n_comps)]
            # reps.index = pd.CategoricalIndex(donors, categories=donors)
            reps_all[cell_type] = reps
        return reps_all


class CompositionBaseline(BaseModelClass):
    has_cell_representation = True
    has_donor_representation = False
    has_local_donor_representation = True

    default_model_kwargs = dict(
        dim_reduction_approach="PCA",
        n_dim=50,
        clustering_on="leiden",
    )
    default_train_kwargs = {}

    def __init__(self, model_kwargs=None, train_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs = (
            model_kwargs if model_kwargs is not None else self.default_model_kwargs
        )
        self.train_kwargs = (
            train_kwargs if train_kwargs is not None else self.default_train_kwargs
        )

        self.dim_reduction_approach = model_kwargs.pop("dim_reduction_approach")
        self.n_dim = model_kwargs.pop("n_dim")
        self.clustering_on = model_kwargs.pop(
            "clustering_on"
        )  # one of leiden, celltype

    def preprocess_data(self):
        super().preprocess_data()
        self.adata_ = self.adata.copy()
        if self.dim_reduction_approach == "PCA":
            self.adata_ = self.adata.copy()
            sc.pp.normalize_total(self.adata_, target_sum=1e4)
            sc.pp.log1p(self.adata_)

    def fit(self):
        self.preprocess_data()
        if self.dim_reduction_approach == "PCA":
            sc.pp.pca(self.adata_, n_comps=self.n_dim)  # saves "X_pca" in obsm
            self.adata_.obsm["X_red"] = self.adata_.obsm["X_pca"]
        elif self.dim_reduction_approach == "SCVI":
            SCVI.setup_anndata(
                self.adata_, categorical_covariate_keys=self.categorical_nuisance_keys
            )
            scvi_model = SCVI(self.adata_, **self.model_kwargs)
            scvi_model.train(**self.train_kwargs)
            self.adata_.obsm["X_red"] = scvi_model.get_latent_representation()

    def get_cell_representation(self, adata=None):
        assert adata is None
        return self.adata_.obsm["X_red"]

    def get_local_donor_representation(self, adata=None):
        if self.clustering_on == "leiden":
            sc.pp.neighbors(self.adata_, n_neighbors=30, use_rep="X_red")
            sc.tl.leiden(self.adata_, resolution=1.0, key_added="leiden_1.0")
            clustering_key = "leiden_1.0"
        elif self.clustering_on == "celltype":
            clustering_key = self.cell_type_key

        freqs_all = dict()
        for unique_cluster in self.adata_.obs[clustering_key].unique():
            cell_is_selected = self.adata_.obs[clustering_key] == unique_cluster
            subann = self.adata_[cell_is_selected].copy()

            # Step 1: subcluster
            sc.pp.neighbors(subann, n_neighbors=30, use_rep="X_red")
            sc.tl.leiden(subann, resolution=1.0, key_added=clustering_key)

            szs = (
                subann.obs.groupby([clustering_key, self.donor_key])
                .size()
                .to_frame("n_cells")
                .reset_index()
            )
            szs_total = (
                szs.groupby(self.donor_key)
                .sum()
                .rename(columns={"n_cells": "n_cells_total"})
            )
            comps = szs.merge(szs_total, on=self.donor_key).assign(
                freqs=lambda x: x.n_cells / x.n_cells_total
            )
            freqs = (
                comps.loc[:, [self.donor_key, clustering_key, "freqs"]]
                .set_index([self.donor_key, clustering_key])
                .squeeze()
                .unstack()
            )
            freqs_ = freqs
            freqs_all[unique_cluster] = freqs_
            # n_donors, n_clusters
        return freqs_all

    def get_donor_representation_metadata(self, adata=None):
        pass


class PCAKNN(BaseModelClass):
    has_cell_representation = True
    has_donor_representation = False
    has_local_donor_representation = True

    def __init__(self, n_components=25, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.pca = None
        self.adata_ = None
        self.donor_order = None

    def preprocess_data(self):
        super().preprocess_data()
        self.adata_ = self.adata.copy()
        sc.pp.normalize_total(self.adata_, target_sum=1e4)
        sc.pp.log1p(self.adata_)

    def fit(self):
        self.preprocess_data()
        sc.pp.pca(self.adata_, n_comps=self.n_components)

    def get_cell_representation(self, adata=None):
        assert adata is None
        return self.adata_.obsm["X_pca"]

    def get_local_donor_representation(self, adata=None):
        # for each cell, compute nearest neighbor in given donor
        pca_rep = self.adata_.obsm["X_pca"]

        local_reps = []
        self.donor_order = self.adata_.obs[self.donor_key].unique()
        for donor in self.donor_order:
            donor_is_selected = self.adata_.obs[self.donor_key] == donor
            pca_donor = pca_rep[donor_is_selected]
            nn_donor = pynndescent.NNDescent(pca_donor)
            nn_indices = nn_donor.query(pca_rep, k=1)[0].squeeze(-1)
            nn_rep = pca_donor[nn_indices][:, None, :]
            local_reps.append(nn_rep)
        local_reps = np.concatenate(local_reps, axis=1)
        return local_reps

    def get_donor_representation_metadata(self):
        donor_to_id_map = pd.DataFrame(
            self.donor_order, columns=[self.donor_key]
        ).assign(donor_order=lambda x: np.arange(len(x)))
        res = self.adata_.obs.drop_duplicates(self.donor_key)
        res = res.merge(donor_to_id_map, on=self.donor_key, how="left").sort_values(
            "donor_order"
        )
        return res


class SCVIModel(BaseModelClass):
    has_cell_representation = True
    has_donor_representation = False
    has_local_donor_representation = True
    has_save = True

    default_model_kwargs = dict(
        dropout_rate=0.0,
        dispersion="gene",
        gene_likelihood="nb",
    )
    default_train_kwargs = dict(
        max_epochs=100,
        check_val_every_n_epoch=1,
        batch_size=256,
        plan_kwargs=dict(lr=1e-2, n_epochs_kl_warmup=20),
    )

    def __init__(self, model_kwargs=None, train_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs = (
            self.default_model_kwargs if model_kwargs is None else model_kwargs
        )
        self.train_kwargs = (
            self.default_train_kwargs if train_kwargs is None else train_kwargs
        )
        self.adata_ = None

    @property
    def has_donor_representation(self):
        has_donor_embedding = self.model_kwargs.get("do_batch_embedding", False)
        return has_donor_embedding

    def fit(self):
        self.preprocess_data()
        adata_ = self.adata.copy()

        feed_nuisance = self.model_kwargs.pop("feed_nuisance", True)
        categorical_nuisance_keys = (
            self.categorical_nuisance_keys if feed_nuisance else None
        )
        SCVI.setup_anndata(
            adata_,
            batch_key=self.donor_key,
            categorical_covariate_keys=categorical_nuisance_keys,
        )
        self.adata_ = adata_
        self.model = SCVI(adata=adata_, **self.model_kwargs)
        self.model.train(**self.train_kwargs)
        return True

    def save(self, save_path, overwrite=True):
        self.model.save(save_path, overwrite=overwrite)
        return True

    # def get_donor_representation(self, adata=None):
    #     assert self.has_donor_representation

    def get_cell_representation(self, adata=None, batch_size=256):
        return self.model.get_latent_representation(adata, batch_size=batch_size)

    @torch.no_grad()
    def get_local_donor_representation(
        self, adata=None, batch_size=256, x_dim=50, eps=1e-8, mc_samples=10
    ):
        # z = self.model.get_latent_representation(adata, batch_size=batch_size)
        # index = pynndescent.NNDescent(z, n_neighbors=n_neighbors)
        # index.prepare()

        # neighbors, _ = index.query(z)
        # return neighbors[:, 1:]

        adata = self.adata_ if adata is None else adata
        self.model._check_if_trained(warn=False)
        adata = self.model._validate_anndata(adata)
        scdl = self.model._make_data_loader(
            adata=adata, indices=None, batch_size=batch_size
        )

        # hs = self.get_normalized_expression(adata, batch_size=batch_size, eps=eps)
        hs = []
        for tensors in tqdm(scdl):
            inference_inputs = self.model.module._get_inference_input(tensors)
            inference_outputs = self.model.module.inference(**inference_inputs)

            generative_inputs = self.model.module._get_generative_input(
                tensors=tensors, inference_outputs=inference_outputs
            )
            generative_outputs = self.model.module.generative(**generative_inputs)
            new = (eps + generative_outputs["px"].scale).log()
            hs.append(new.cpu())
        hs = torch.cat(hs, dim=0).numpy()
        means = np.mean(hs, axis=0)
        stds = np.std(hs, axis=0)
        hs = (hs - means) / stds
        pca = PCA(n_components=x_dim).fit(hs)
        w = torch.tensor(
            pca.components_, dtype=torch.float32, device=self.model.device
        ).T
        means = torch.tensor(means, dtype=torch.float32, device=self.model.device)
        stds = torch.tensor(stds, dtype=torch.float32, device=self.model.device)

        reps = []
        for tensors in tqdm(scdl):
            xs = []
            for batch in range(self.model.summary_stats.n_batch):
                cf_batch = batch * torch.ones_like(tensors["batch"])
                tensors[REGISTRY_KEYS.BATCH_KEY] = cf_batch
                inference_inputs = self.model.module._get_inference_input(tensors)
                inference_outputs = self.model.module.inference(
                    n_samples=mc_samples, **inference_inputs
                )

                generative_inputs = self.model.module._get_generative_input(
                    tensors=tensors, inference_outputs=inference_outputs
                )
                generative_outputs = self.model.module.generative(**generative_inputs)
                new = (eps + generative_outputs["px"].scale).log()
                if x_dim is not None:
                    new = (new - means) / stds
                    new = new @ w
                xs.append(new[:, :, None])

            xs = torch.cat(xs, 2).mean(0)
            reps.append(xs.cpu().numpy())
        # n_cells, n_donors, n_donors
        reps = np.concatenate(reps, 0)
        return reps

    def get_donor_representation_metadata(self):
        return (
            self.model.adata.obs.drop_duplicates("_scvi_batch")
            .set_index("_scvi_batch")
            .sort_index()
        )
