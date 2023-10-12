import logging
from functools import partial

import anndata as ad
import numpy as np
import pynndescent
import scanpy as sc
import torch
from tqdm import tqdm
from utils import compute_ks, compute_manova


def compute_autocorrelation_metric(
    local_representation,
    donor_labels,
    metric_fn,
    has_significance,
    batch_size,
    minibatched=True,
    desc="desc",
):
    # Compute autocorrelation
    if minibatched:
        scores = []
        pvals = []
        for local_rep in tqdm(local_representation.split(batch_size), desc=desc):
            local_rep_ = local_rep.to("cuda")
            if has_significance:
                score, pval = metric_fn(local_rep_, donor_labels)
                scores.append(score)
                pvals.append(pval)
            else:
                scores_ = metric_fn(local_rep_, donor_labels=donor_labels)
                scores.append(scores_)
        scores = np.concatenate(scores, 0)
        pvals = np.concatenate(pvals, 0) if has_significance else None
    else:
        if has_significance:
            scores, pvals = metric_fn(
                local_representation.numpy(), donor_labels=donor_labels
            )
        else:
            scores = metric_fn(local_representation.numpy(), donor_labels=donor_labels)
            pvals = None
    return scores, pvals


def compute_mrvi(adata, model_name, obs_key, batch_size=256, redo=True):
    uns_key = "{}_local_donor_rep".format(model_name)
    local_donor_rep = adata.uns[uns_key]
    if not isinstance(local_donor_rep, np.ndarray):
        logging.warn(f"{model_name} local donor rep not compatible with metric.")
        return
    local_representation = torch.from_numpy(local_donor_rep)
    # Assumes local_representation is n_cells, n_donors, n_donors
    metadata_key = "{}_local_donor_rep_metadata".format(model_name)
    metadata = adata.uns[metadata_key]
    donor_labels = metadata[obs_key].values

    # Compute and save various metric scores
    _compute_ks = partial(compute_ks, do_smoothing=False)
    configs = [
        dict(
            metric_name="ks",
            metric_fn=_compute_ks,
            has_significance=True,
            minibatched=True,
        ),
        dict(
            metric_name="manova",
            metric_fn=compute_manova,
            has_significance=True,
            minibatched=False,
        ),
    ]

    for config in configs:
        metric_name = config.pop("metric_name")
        scores, pvals = compute_autocorrelation_metric(
            local_representation,
            donor_labels,
            batch_size=batch_size,
            **config,
        )
        output_key = f"{model_name}_{obs_key}_{metric_name}_score"
        sig_key = f"{model_name}_{obs_key}_{metric_name}_significance"

        is_obs = scores.shape[0] == adata.shape[0]
        adata.uns[output_key] = scores
        if pvals is not None:
            adata.uns[sig_key] = pvals
        if is_obs:
            adata.obs[output_key] = scores
            if pvals is not None:
                adata.obs[sig_key] = pvals


def compute_milo(adata, model_name, obs_key):
    import milopy.core as milo

    sample_col = adata.uns["nhood_adata"].uns["sample_col"]
    if sample_col == obs_key:
        logging.warning("Milo cannot run a GLM against the sample col, skipping.")
        return

    try:
        adata.obs[obs_key] = adata.obs[obs_key].astype(str).astype("category")
        design = f"~ {obs_key}"

        # Issue with None valued uns values being dropped.
        if "nhood_neighbors_key" not in adata.uns:
            adata.uns["nhood_neighbors_key"] = None
        milo.DA_nhoods(adata, design=design)
        milo_results = adata.uns["nhood_adata"].obs

    except Exception as e:
        logging.warning(
            f"Skipping test since key {obs_key} may be invalid or model did not complete"
            f" with the error message: {e.__class__.__name__} - {str(e)}."
        )
        return

    is_index_cell_key = f"{model_name}_{obs_key}_is_index_cell"
    is_index_cell = adata.obs.index.isin(milo_results["index_cell"])
    adata.obs[is_index_cell_key] = is_index_cell

    cell_rep = adata.obsm["_cell_rep"]
    cell_anchors = cell_rep[is_index_cell]
    nn_donor = pynndescent.NNDescent(cell_anchors)
    nn_indices = nn_donor.query(cell_rep, k=1)[0].squeeze(-1)

    output_key = f"{model_name}_{obs_key}_LFC_score"
    lfc_score = milo_results["logFC"].values[nn_indices]
    adata.obs[output_key] = lfc_score

    sig_key = f"{model_name}_{obs_key}_LFC_significance"
    lfc_sig = milo_results["SpatialFDR"].values[nn_indices]
    adata.obs[sig_key] = lfc_sig
    return


def compute_metrics(adata, model_name, obs_key, batch_size=256, redo=True):
    if model_name.startswith("MILO"):
        compute_milo(adata, model_name, obs_key)
    else:
        compute_mrvi(adata, model_name, obs_key, batch_size, redo)


def process_predictions(
    model_name,
    path_to_h5ad,
    path_to_output,
    donor_obs_keys,
):
    adata = sc.read_h5ad(path_to_h5ad)
    for donor_obs_key in donor_obs_keys:
        compute_metrics(adata, model_name, donor_obs_key)
    adata_ = ad.AnnData(
        obs=adata.obs,
        obsm=adata.obsm,
        var=adata.var,
        uns=adata.uns,
    )
    adata_.write(path_to_output)


if __name__ == "__main__":
    process_predictions(
        model_name=snakemake.wildcards.model,
        path_to_h5ad=snakemake.input[0],
        path_to_output=snakemake.output[0],
        donor_obs_keys=snakemake.config[snakemake.wildcards.dataset]["keyMapping"][
            "relevantKeys"
        ],
    )
