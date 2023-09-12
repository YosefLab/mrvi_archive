import scanpy as sc
from scvi.model import SCVI

from ._base_model import BaseModelClass


class MILO(BaseModelClass):
    has_donor_representation = False
    has_cell_representation = False
    has_local_donor_representation = False
    has_custom_representation = True
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

    def fit(self, **kwargs):
        import milopy.core as milo

        adata_ = self.adata.copy()
        embedding = self.model_kwargs.pop("embedding", "mnn")

        if embedding == "mnn":
            # Run MNN
            alldata = []
            batch_key = self.categorical_nuisance_keys[0]
            adata_.obs[batch_key] = adata_.obs[batch_key].astype("category")
            for batch_cat in adata_.obs[batch_key].cat.categories.tolist():
                alldata.append(adata_[adata_.obs[batch_key] == batch_cat,])

            cdata = sc.external.pp.mnn_correct(
                *alldata, svd_dim=50, batch_key=batch_key, n_jobs=8
            )[0]
            if isinstance(cdata, tuple):
                cdata = cdata[0]

            # Run PCA
            cell_rep = sc.tl.pca(cdata.X, svd_solver="arpack", return_info=False)
        elif embedding == "scvi":
            # Run scVI
            self.preprocess_data()
            adata_ = self.adata.copy()

            batch_key = self.categorical_nuisance_keys[0]
            SCVI.setup_anndata(
                adata_,
                batch_key=batch_key,
                categorical_covariate_keys=self.categorical_nuisance_keys[1:]
                if len(self.categorical_nuisance_keys) > 1
                else None,
            )
            self.adata_ = adata_
            scvi_model = SCVI(adata_, **self.model_kwargs)
            scvi_model.train(**self.train_kwargs)
            cell_rep = scvi_model.get_latent_representation()
        else:
            raise ValueError(f"Unknown embedding: {self.embedding}")

        adata_.obsm["_cell_rep"] = cell_rep
        sc.pp.neighbors(adata_, n_neighbors=10, use_rep="_cell_rep")

        ## Assign cells to neighbourhoods
        milo.make_nhoods(adata_, prop=0.05)

        ## Count cells from each sample in each nhood
        milo.count_nhoods(adata_, sample_col=self.donor_key)
        self.adata_ = adata_

    def compute(self):
        return self.adata_
