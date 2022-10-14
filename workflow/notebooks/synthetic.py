# %%
from collections import defaultdict
import glob
import os
from itertools import product
import re

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import ete3
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from sklearn.metrics import pairwise_distances

# %%
workflow_dir = "../"
base_dir = os.path.join(workflow_dir, "results/synthetic")
full_data_path = os.path.join(workflow_dir, "data/synthetic/adata.processed.h5ad")
input_files = glob.glob(os.path.join(base_dir, "final_adata*"))

figure_dir = os.path.join(workflow_dir, "figures/synthetic/")

# %%
# Ground truth similarity matrix
meta_corr = [0, 0.5, 0.9]
donor_combos = list(product(*[[0, 1] for _ in range(len(meta_corr))]))

# E[||x - y||_2^2]
meta_dist = 2 - 2 * np.array(meta_corr)
dist_mtx = np.zeros((len(donor_combos), len(donor_combos)))
for i in range(len(donor_combos)):
    for j in range(i):
        donor_i, donor_j = donor_combos[i], donor_combos[j]
        donor_diff = abs(np.array(donor_i) - np.array(donor_j))
        dist_mtx[i, j] = dist_mtx[j, i] = np.sum(donor_diff * meta_dist)
dist_mtx = np.sqrt(dist_mtx)  # E[||x - y||_2]

donor_replicates = 4

gt_donor_combos = sum([donor_replicates * [str(dc)] for dc in donor_combos], [])
gt_dist_mtx = np.zeros((len(gt_donor_combos), len(gt_donor_combos)))
for i in range(len(gt_donor_combos)):
    for j in range(i + 1):
        gt_dist_mtx[i, j] = gt_dist_mtx[j, i] = dist_mtx[
            i // donor_replicates, j // donor_replicates
        ]
gt_control_dist_mtx = 1 - np.eye(len(gt_donor_combos))

# %%
model_adatas = defaultdict(list)

mapper = None
for file in tqdm(input_files):
    adata_ = sc.read_h5ad(file)
    file_re = re.match(r".*final_adata_([\w\d]+)_(\d+).h5ad", file)
    if file_re is None:
        continue
    model_name, seed = file_re.groups()
    print(model_name, seed)

    uns_keys = list(adata_.uns.keys())
    for uns_key in uns_keys:
        uns_vals = adata_.uns[uns_key]
        if uns_key.endswith("local_donor_rep"):
            meta_keys = [
                key
                for key in adata_.uns.keys()
                if key.endswith("_local_donor_rep_metadata")
            ]
            print(uns_key)
            if len(meta_keys) != 0:
                # SORT UNS by donor_key values
                meta_key = meta_keys[0]
                print(uns_key, meta_key)
                mapper = adata_.uns["mapper"]
                donor_key = mapper["donor_key"]
                metad = adata_.uns[meta_key].reset_index().sort_values(donor_key)
                # print(metad.index.values)
                uns_vals = uns_vals[:, metad.index.values]
            else:
                uns_vals = adata_.uns[uns_key]
                for key in uns_vals:
                    uns_vals[key] = uns_vals[key].sort_index()
                # print(uns_vals[key].index)

        adata_.uns[uns_key] = uns_vals
    model_adatas[model_name].append(dict(adata=adata_, seed=seed))

# %%
ct_key = mapper["cell_type_key"]
sample_key = mapper["donor_key"]

# %%
# Unsupervised analysis
MODELS = [
    dict(model_name="CompositionPCA", cell_specific=False),
    dict(model_name="CompositionSCVI", cell_specific=False),
    dict(model_name="MrVISmall", cell_specific=True),
    dict(model_name="MrVILinear", cell_specific=True),
    dict(model_name="MrVILinear50", cell_specific=True),
    dict(model_name="MrVILinear10COMP", cell_specific=True),
    dict(model_name="MrVILinear50COMP", cell_specific=True),
]

# %%
def compute_aggregate_dmat(reps):
    # return pairwise_distances(reps.mean(0))
    n_cells, n_donors, _ = reps.shape
    pairwise_ds = np.zeros((n_donors, n_donors))
    for x in tqdm(reps):
        d_ = pairwise_distances(x, metric=METRIC)
        pairwise_ds += d_ / n_cells
    return pairwise_ds


# %%
METRIC = "euclidean"

dist_mtxs = defaultdict(list)
for model_params in MODELS:
    model_name = model_params["model_name"]
    rep_key = f"{model_name}_local_donor_rep"
    metadata_key = f"{model_name}_local_donor_rep_metadata"
    is_cell_specific = model_params["cell_specific"]

    for model_res in model_adatas[model_name]:
        adata, seed = model_res["adata"], model_res["seed"]
        if rep_key not in adata.uns:
            continue
        rep = adata.uns[rep_key]

        print(model_name)
        for cluster in adata.obs[ct_key].unique():
            if not is_cell_specific:
                rep_ct = rep[cluster]
                ss_matrix = pairwise_distances(rep_ct.values, metric=METRIC)
                cats = metad.set_index(sample_key).loc[rep_ct.index].index.values
            else:
                good_cells = adata.obs[ct_key] == cluster
                rep_ct = rep[good_cells]
                subobs = adata.obs[good_cells]
                observed_donors = subobs[sample_key].value_counts()
                observed_donors = observed_donors[observed_donors >= 1].index
                good_d_idx = metad.loc[
                    lambda x: x[sample_key].isin(observed_donors)
                ].index.values

                ss_matrix = compute_aggregate_dmat(rep_ct[:, good_d_idx, :])
                cats = metad.loc[lambda x: x[sample_key].isin(observed_donors)][
                    sample_key
                ].values

            dist_mtxs[model_name].append(
                dict(dist_matrix=ss_matrix, cats=cats, seed=seed, ct=cluster)
            )

# %%
dist_mtxs["GroundTruth"] = [
    dict(dist_matrix=gt_dist_mtx, cats=gt_donor_combos, seed=None, ct="CT1:1"),
    dict(dist_matrix=gt_control_dist_mtx, cats=gt_donor_combos, seed=None, ct="CT2:1"),
]

# %%
# https://stackoverflow.com/questions/9364609/converting-ndarray-generated-by-hcluster-into-a-newick-string-for-use-with-ete2/17657426#17657426
def linkage_to_ete(linkage_obj):
    R = to_tree(linkage_obj)
    root = ete3.Tree()
    root.dist = 0
    root.name = "root"
    item2node = {R.get_id(): root}
    to_visit = [R]

    while to_visit:
        node = to_visit.pop()
        cl_dist = node.dist / 2.0

        for ch_node in [node.get_left(), node.get_right()]:
            if ch_node:
                ch_node_id = ch_node.get_id()
                ch_node_name = (
                    f"t{int(ch_node_id) + 1}" if ch_node.is_leaf() else str(ch_node_id)
                )
                ch = ete3.Tree()
                ch.dist = cl_dist
                ch.name = ch_node_name

                item2node[node.get_id()].add_child(ch)
                item2node[ch_node_id] = ch
                to_visit.append(ch_node)
    return root


# %%
ete_trees = defaultdict(list)
for model in dist_mtxs:
    for model_dist_mtx in dist_mtxs[model]:
        dist_mtx = model_dist_mtx["dist_matrix"]
        seed = model_dist_mtx["seed"]
        ct = model_dist_mtx["ct"]
        cats = model_dist_mtx["cats"]
        cats = [cat[10:] if cat[:10] == "donor_meta" else cat for cat in cats]

        # Heatmaps
        fig, ax = plt.subplots()
        im = ax.imshow(dist_mtx)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(cats)), labels=cats, fontsize=5)
        ax.set_yticks(np.arange(len(cats)), labels=cats, fontsize=5)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        cell_type = "Experimental Cell Type" if ct == "CT1:1" else "Control Cell Type"
        ax.set_title(f"{model} {cell_type}")

        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, f"{model}_{ct}_{seed}_dist_matrix.svg"))
        plt.show()
        plt.close()

        # Dendrograms
        z = linkage(dist_mtx, method="ward")

        fig, ax = plt.subplots()
        dn = dendrogram(z, ax=ax, orientation="top")
        ax.set_title(f"{model} {cell_type}")
        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, f"{model}_{ct}_{seed}_dendrogram.svg"))
        plt.close()

        z_ete = linkage_to_ete(z)
        ete_trees[model].append(dict(t=z_ete, ct=ct))

# %%
# Boxplot for RF distance
for gt_trees in ete_trees["GroundTruth"]:
    if gt_trees["ct"] == "CT1:1":
        gt_tree = gt_trees["t"]
    elif gt_trees["ct"] == "CT2:1":
        gt_control_tree = gt_trees["t"]
    else:
        continue

rf_df_rows = []
for model, ts in ete_trees.items():
    if model == "GroundTruth":
        continue

    for t_dict in ts:
        t = t_dict["t"]
        ct = t_dict["ct"]
        if ct == "CT1:1":
            rf_dist = gt_tree.robinson_foulds(t)
        elif ct == "CT2:1":
            rf_dist = gt_control_tree.robinson_foulds(t)
        else:
            continue
        norm_rf = rf_dist[0] / rf_dist[1]

        rf_df_rows.append((model, ct, norm_rf))

rf_df = pd.DataFrame(rf_df_rows, columns=["model", "cell_type", "rf_dist"])

# %%
# Experimental Cell Type Plot
fig, ax = plt.subplots(figsize=(2, 5))
sns.barplot(
    data=rf_df[rf_df["cell_type"] == "CT1:1"],
    x="model",
    y="rf_dist",
    ax=ax,
)
sns.swarmplot(
    data=rf_df[rf_df["cell_type"] == "CT1:1"],
    x="model",
    y="rf_dist",
    color="black",
    ax=ax,
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
ax.set_xlabel("Robinson-Foulds Distance")
ax.set_ylabel("Model")
ax.set_title("Robinson-Foulds Comparison to Ground Truth for Experimental Cell Type")
fig.tight_layout()
fig.savefig(
    os.path.join(figure_dir, f"robinson_foulds_CT1:1_boxplot.svg"), bbox_inches="tight"
)
plt.show()

# %%
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import (
    precision_score,
    average_precision_score,
)


# %%
def correct_for_mult(x):
    # print(x.name)
    if x.name.startswith("MILO"):
        print("MILO already FDR controlled; skipping")
        return x
    return multipletests(x, method="fdr_bh")[1]


padj_dfs = []
ct_obs = None
for model in model_adatas:
    for model_adata in model_adatas[model]:
        adata = model_adata["adata"]
        if ct_obs is None:
            ct_obs = adata.obs["celltype"]

        # Selecting right columns in adata.obs
        sig_keys = [col for col in adata.obs.columns if col.endswith("significance")]
        # Adjust for multiple testing if needed
        padj_dfs.append(adata.obs.loc[:, sig_keys].apply(correct_for_mult, axis=0))

# %%
# Select right cell subpopulations
padjs = pd.concat(padj_dfs, axis=1)
target = ct_obs == "CT1:1"

plot_df = (
    padjs.apply(
        lambda x: pd.Series(
            {
                **{
                    target_fdr: 1.0
                    - precision_score(target, x <= target_fdr, zero_division=0)
                    for target_fdr in [0.05, 0.1, 0.2]
                },
            }
        ),
        axis=0,
    )
    .stack()
    .to_frame("FDP")
    .reset_index()
    .rename(columns=dict(level_0="targetFDR", level_1="approach"))
    .assign(
        model=lambda x: x.approach.str.split("_").str[0],
        test=lambda x: x.approach.str.split("_").str[-2],
        model_test=lambda x: x.model + " " + x.test,
        metadata=lambda x: x.approach.str.split("_")
        .str[1:-2]
        .apply(lambda y: "_".join(y)),
    )
)
# %%
# Keep MrVILinear Manova, PCAKNN Manova, SCVI MANOVA, MRVILINEAR50 KS
# MrVILinear KS, PCAKNN KS, SCVI KS, MrVILiear50 KS, MILOSCVI LFC
for metadata in plot_df.metadata.unique():
    fig, ax = plt.subplots()
    sns.barplot(
        data=plot_df[
            (plot_df["metadata"] == metadata)
            & (
                plot_df["model"].isin(
                    (
                        "MrVILinear",
                        "MrVILinear50",
                        "PCAKNN",
                        "SCVI",
                        "MILOSCVI",
                    )
                )
            )
            & (
                plot_df["test"].isin(
                    (
                        # "manova",
                        "ks",
                        "LFC",
                    )
                )
            )
        ],
        x="targetFDR",
        y="FDP",
        hue="model_test",
        ax=ax,
        hue_order=[
            "MrVILinear ks",
            "MrVILinear50 ks",
            "SCVI ks",
            "MILOSCVI LFC",
            "PCAKNN ks",
        ],
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_xlabel("Target FDR")
    ax.set_ylabel("FDP")
    ax.set_title(f"False Discovery Rate Comparison ({metadata[6:]})")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_dir, f"FDR_comparison_{metadata}_ks_only.svg"),
        bbox_inches="tight",
    )
    plt.show()
# %%
# AP Comparison
ap_plot_df = (
    padjs.apply(
        lambda x: pd.Series(
            {
                **{
                    target_fdr: average_precision_score(target, x <= target_fdr)
                    for target_fdr in [0.05, 0.1, 0.2]
                },
            }
        ),
        axis=0,
    )
    .stack()
    .to_frame("AP")
    .reset_index()
    .rename(columns=dict(level_0="targetFDR", level_1="approach"))
    .assign(
        model=lambda x: x.approach.str.split("_").str[0],
        test=lambda x: x.approach.str.split("_").str[-2],
        model_test=lambda x: x.model + " " + x.test,
        metadata=lambda x: x.approach.str.split("_")
        .str[1:-2]
        .apply(lambda y: "_".join(y)),
    )
)

# %%
for metadata in ap_plot_df.metadata.unique():
    fig, ax = plt.subplots()
    sns.barplot(
        data=ap_plot_df[
            (ap_plot_df["metadata"] == metadata)
            & (
                plot_df["model"].isin(
                    (
                        "MrVILinear",
                        "MrVILinear50",
                        "PCAKNN",
                        "SCVI",
                        "MILOSCVI",
                    )
                )
            )
            & (
                plot_df["test"].isin(
                    (
                        # "manova",
                        "ks",
                        "LFC",
                    )
                )
            )
        ],
        x="targetFDR",
        y="AP",
        hue="model_test",
        ax=ax,
        hue_order=[
            "MrVILinear ks",
            "MrVILinear50 ks",
            "SCVI ks",
            "MILOSCVI LFC",
            "PCAKNN ks",
        ],
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_xlabel("Target FDR")
    ax.set_ylabel("AP")
    ax.set_ylim((0.5, 1.05))
    ax.set_title(f"Average Precision Rate Comparison ({metadata})")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_dir, f"AP_comparison_{metadata}_ks_only.svg"),
        bbox_inches="tight",
    )
    plt.show()

# %%
