import logging
import os

import anndata as ad
import numpy as np
import scanpy as sc
from utils import MILO, PCAKNN, CompositionBaseline, MrVIWrapper, SCVIModel

profile = snakemake.config[snakemake.wildcards.dataset]
N_EPOCHS = profile["nEpochs"]
if "batchSize" in profile:
    BATCH_SIZE = profile["batchSize"]
else:
    BATCH_SIZE = 256

MRVI_BASE_MODEL_KWARGS = dict(
    observe_library_sizes=True,
    # n_latent_donor=5,
    px_kwargs=dict(n_hidden=32),
    pz_kwargs=dict(
        n_layers=1,
        n_hidden=32,
    ),
)
MRVI_BASE_TRAIN_KWARGS = dict(
    max_epochs=N_EPOCHS,
    early_stopping=True,
    early_stopping_patience=15,
    check_val_every_n_epoch=1,
    batch_size=BATCH_SIZE,
    train_size=0.9,
)

MODELS = dict(
    PCAKNN=(PCAKNN, dict()),
    MILO=(MILO, dict(model_kwargs=dict(embedding="mnn"))),
    MILOSCVI=(
        MILO,
        dict(
            model_kwargs=dict(
                embedding="scvi",
                dropout_rate=0.0,
                dispersion="gene",
                gene_likelihood="nb",
            ),
            train_kwargs=dict(
                batch_size=BATCH_SIZE,
                plan_kwargs=dict(lr=1e-2, n_epochs_kl_warmup=20),
                max_epochs=N_EPOCHS,
                early_stopping=True,
                early_stopping_patience=15,
            ),
        ),
    ),
    CompositionPCA=(
        CompositionBaseline,
        dict(
            model_kwargs=dict(
                dim_reduction_approach="PCA",
                n_dim=50,
                clustering_on="celltype",
            ),
            train_kwargs=None,
        ),
    ),
    CompositionSCVI=(
        CompositionBaseline,
        dict(
            model_kwargs=dict(
                dim_reduction_approach="SCVI",
                n_dim=10,
                clustering_on="celltype",
            ),
            train_kwargs=dict(
                batch_size=BATCH_SIZE,
                plan_kwargs=dict(lr=1e-2),
                max_epochs=N_EPOCHS,
                early_stopping=True,
                early_stopping_patience=15,
            ),
        ),
    ),
    SCVI=(
        SCVIModel,
        dict(
            model_kwargs=dict(
                dropout_rate=0.0,
                feed_nuisance=False,
                dispersion="gene",
                gene_likelihood="nb",
            ),
            train_kwargs=dict(
                batch_size=BATCH_SIZE,
                plan_kwargs=dict(lr=1e-2, n_epochs_kl_warmup=20),
                max_epochs=N_EPOCHS,
                early_stopping=True,
                early_stopping_patience=15,
            ),
        ),
    ),
    MrVISmall=(
        MrVIWrapper,
        dict(
            model_kwargs=dict(
                linear_decoder_zx=False,
                **MRVI_BASE_MODEL_KWARGS,
            ),
            train_kwargs=dict(
                plan_kwargs=dict(
                    lr=1e-2, n_epochs_kl_warmup=20, do_comp=True, lambd=1.0
                ),
                **MRVI_BASE_TRAIN_KWARGS,
            ),
        ),
    ),
    MrVILinear=(
        MrVIWrapper,
        dict(
            model_kwargs=dict(
                linear_decoder_zx=True,
                n_latent=10,
                **MRVI_BASE_MODEL_KWARGS,
            ),
            train_kwargs=dict(
                plan_kwargs=dict(
                    lr=1e-2, n_epochs_kl_warmup=20, do_comp=False, lambd=1.0
                ),
                **MRVI_BASE_TRAIN_KWARGS,
            ),
        ),
    ),
    MrVILinear50=(
        MrVIWrapper,
        dict(
            model_kwargs=dict(
                linear_decoder_zx=True,
                n_latent=50,
                n_latent_donor=2,
                **MRVI_BASE_MODEL_KWARGS,
            ),
            train_kwargs=dict(
                plan_kwargs=dict(
                    lr=1e-2, n_epochs_kl_warmup=20, do_comp=False, lambd=1.0
                ),
                **MRVI_BASE_TRAIN_KWARGS,
            ),
        ),
    ),
    MrVILinear50COMP=(
        MrVIWrapper,
        dict(
            model_kwargs=dict(
                linear_decoder_zx=True,
                n_latent=50,
                n_latent_donor=2,
                **MRVI_BASE_MODEL_KWARGS,
            ),
            train_kwargs=dict(
                plan_kwargs=dict(
                    lr=1e-2,
                    n_epochs_kl_warmup=20,
                    do_comp=True,
                    lambd=0.1,
                ),
                **MRVI_BASE_TRAIN_KWARGS,
            ),
        ),
    ),
    MrVILinear10COMP=(
        MrVIWrapper,
        dict(
            model_kwargs=dict(
                linear_decoder_zx=True,
                n_latent=10,
                n_latent_donor=2,
                **MRVI_BASE_MODEL_KWARGS,
            ),
            train_kwargs=dict(
                plan_kwargs=dict(
                    lr=1e-2,
                    n_epochs_kl_warmup=20,
                    do_comp=True,
                    lambd=0.1,
                ),
                **MRVI_BASE_TRAIN_KWARGS,
            ),
        ),
    ),
    MrVILinearLinear10COMP=(
        MrVIWrapper,
        dict(
            model_kwargs=dict(
                linear_decoder_zx=True,
                linear_decoder_uz=True,
                n_latent=10,
                n_latent_donor=2,
                **MRVI_BASE_MODEL_KWARGS,
            ),
            train_kwargs=dict(
                plan_kwargs=dict(
                    lr=1e-2,
                    n_epochs_kl_warmup=20,
                    do_comp=True,
                    lambd=0.1,
                ),
                **MRVI_BASE_TRAIN_KWARGS,
            ),
        ),
    ),
    MrVILinearLinear10=(
        MrVIWrapper,
        dict(
            model_kwargs=dict(
                linear_decoder_zx=True,
                linear_decoder_uz=True,
                n_latent=10,
                n_latent_donor=2,
                **MRVI_BASE_MODEL_KWARGS,
            ),
            train_kwargs=dict(
                plan_kwargs=dict(
                    lr=1e-2,
                    n_epochs_kl_warmup=20,
                    do_comp=False,
                    lambd=0.1,
                ),
                **MRVI_BASE_TRAIN_KWARGS,
            ),
        ),
    ),
    MrVILinearLinear10SCALER=(
        MrVIWrapper,
        dict(
            model_kwargs=dict(
                linear_decoder_zx=True,
                linear_decoder_uz=True,
                linear_decoder_uz_scaler=True,
                n_latent=10,
                n_latent_donor=2,
                **MRVI_BASE_MODEL_KWARGS,
            ),
            train_kwargs=dict(
                plan_kwargs=dict(
                    lr=1e-2,
                    n_epochs_kl_warmup=20,
                    do_comp=False,
                    lambd=0.1,
                ),
                **MRVI_BASE_TRAIN_KWARGS,
            ),
        ),
    ),
)


def compute_model_predictions(model_name, path_to_h5ad, path_to_output, random_seed):
    np.random.seed(random_seed)
    adata = sc.read_h5ad(path_to_h5ad)
    logging.info("adata shape: {}".format(adata.shape))
    mapper = adata.uns["mapper"]
    algo_cls, algo_kwargs = MODELS[model_name]
    model = algo_cls(adata=adata, **algo_kwargs, **mapper)
    model.fit()
    if model.has_donor_representation:
        rep = model.get_donor_representation().assign(model=model_name)
        rep.columns = rep.columns.astype(str)
        adata.uns["{}_donor".format(model_name)] = rep
    if model.has_cell_representation:
        repb = model.get_cell_representation()
        adata.obsm["{}_cell".format(model_name)] = repb
    if model.has_local_donor_representation:
        _adata = None
        scores = model.get_local_sample_representation(adata=_adata)
        adata.uns["{}_local_donor_rep".format(model_name)] = scores
        if hasattr(model, "get_donor_representation_metadata"):
            metadata = model.get_donor_representation_metadata()
            adata.uns["{}_local_donor_rep_metadata".format(model_name)] = metadata
    if model.has_custom_representation:
        adata = model.compute()

    adata.uns["model_name"] = model_name
    adata.X = None
    adata_ = ad.AnnData(
        obs=adata.obs,
        var=adata.var,
        obsm=adata.obsm,
        uns=adata.uns,
    )
    adata_.write_h5ad(path_to_output)
    if model.has_save:
        dir_path = os.path.dirname(path_to_output)
        model_dir_path = os.path.join(dir_path, model_name)
        model.save(model_dir_path)


if __name__ == "__main__":
    compute_model_predictions(
        model_name=snakemake.wildcards.model,
        path_to_h5ad=snakemake.input[0],
        path_to_output=snakemake.output[0],
        random_seed=int(snakemake.wildcards.seed),
    )
