# %%
import glob
import os
import re
from collections import defaultdict

import ete3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
from scipy import stats
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.metrics import pairwise_distances, precision_score, recall_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# %%
workflow_dir = "../"
base_dir = os.path.join(workflow_dir, "results/semisynthetic")
full_data_path = os.path.join(workflow_dir, "data/semisynthetic/adata.processed.h5ad")
input_files = glob.glob(os.path.join(base_dir, "final_adata*"))

figure_dir = os.path.join(workflow_dir, "figures/semisynthetic/")


model_adatas = defaultdict(list)
mapper = None
for file in tqdm(input_files):
    print(file)
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
adata_ = model_adatas["MrVILinearLinear10"][0]["adata"]
affected_ct_a = adata_.obs.affected_ct1.unique()[0]
ordered_tree_a = adata_.uns[
    "MrVILinearLinear10_local_donor_rep_metadata"
].tree_id1.sort_values()
t_gt_a = ete3.Tree(
    "((({}, {}), ({}, {})), (({},{}), ({}, {})));".format(
        *["t{}".format(val + 1) for val in ordered_tree_a.index]
    )
)
d_order_a = ordered_tree_a.index

affected_ct_b = adata_.obs.affected_ct2.unique()[0]
ordered_tree_b = adata_.uns[
    "MrVILinearLinear10_local_donor_rep_metadata"
].tree_id2.sort_values()
t_gt_b = ete3.Tree(
    "((({}, {}), ({}, {})), (({},{}), ({}, {})));".format(
        *["t{}".format(val + 1) for val in ordered_tree_b.index]
    )
)
d_order_b = ordered_tree_b.index
print(affected_ct_a, affected_ct_b)


# %%

similarity_mat = np.zeros((len(ordered_tree_b), len(ordered_tree_b)))
for i in range(len(ordered_tree_b)):
    for j in range(len(ordered_tree_b)):
        x = np.array([int(v) for v in ordered_tree_b.iloc[i]], dtype=bool)
        y = np.array([int(v) for v in ordered_tree_b.iloc[j]], dtype=bool)
        shared = x == y
        shared_ = np.where(~shared)[0]
        if len(shared_) == 0:
            similarity_mat[i, j] = 5
        else:
            similarity_mat[i, j] = np.where(~shared)[0][0]
dissimilarity_mat = 5 - similarity_mat
# %%
ct_key = "leiden"
sample_key = mapper["donor_key"]

# %%
# Unsupervised analysis
MODELS = [
    dict(model_name="CompositionPCA", cell_specific=False),
    dict(model_name="CompositionSCVI", cell_specific=False),
    dict(model_name="MrVILinearLinear10", cell_specific=True),
]


# %%
def compute_aggregate_dmat(reps):
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
fig, ax = plt.subplots()
im = ax.imshow(dissimilarity_mat)
fig.savefig(os.path.join(figure_dir, "CT B_GT_dist_matrix.svg"))
plt.show()
plt.close()
# %%
rf_df = []
ete_trees = defaultdict(list)
for model in dist_mtxs:
    for model_dist_mtx in dist_mtxs[model]:
        dist_mtx = model_dist_mtx["dist_matrix"]
        seed = model_dist_mtx["seed"]
        ct = model_dist_mtx["ct"]
        cats = model_dist_mtx["cats"]
        # cats = [cat[10:] if cat[:10] == "donor_meta" else cat for cat in cats]
        if ct == affected_ct_a:
            # Heatmaps
            fig, ax = plt.subplots()
            vmin = np.quantile(dist_mtx, 0.05)
            vmax = np.quantile(dist_mtx, 0.7)
            im = ax.imshow(
                dist_mtx[d_order_a][:, d_order_a],
                vmin=vmin,
                vmax=vmax,
            )

            # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(len(cats)), fontsize=5)
            ax.set_yticks(np.arange(len(cats)), fontsize=5)

            # Rotate the tick labels and set their alignment.
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            cell_type = "CT A"
            ax.set_title(f"{model} {cell_type}")

            fig.tight_layout()
            fig.savefig(
                os.path.join(figure_dir, f"{cell_type}_{model}_{seed}_dist_matrix.svg")
            )
            plt.show()
            plt.close()

            z = linkage(dist_mtx, method="ward")
            z_ete = linkage_to_ete(z)
            ete_trees[model].append(dict(t=z_ete, ct=ct))

            rf_dist = t_gt_a.robinson_foulds(z_ete)
            norm_rf = rf_dist[0] / rf_dist[1]
            rf_df.append(
                dict(
                    ct=ct,
                    model=model,
                    rf=norm_rf,
                )
            )

        if ct == affected_ct_b:
            fig, ax = plt.subplots()
            vmin = np.quantile(dist_mtx, 0.05)
            vmax = np.quantile(dist_mtx, 0.7)
            im = ax.imshow(
                dist_mtx[d_order_b][:, d_order_b],
                vmin=vmin,
                vmax=vmax,
            )

            # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(len(cats)), fontsize=5)
            ax.set_yticks(np.arange(len(cats)), fontsize=5)

            # Rotate the tick labels and set their alignment.
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            cell_type = "CT B"
            ax.set_title(f"{model} {cell_type}")

            fig.tight_layout()
            fig.savefig(
                os.path.join(figure_dir, f"{cell_type}_{model}_{seed}_dist_matrix.svg")
            )
            plt.show()
            plt.close()

            # Dendrograms
            z = linkage(dist_mtx, method="ward")
            z_ete = linkage_to_ete(z)
            ete_trees[model].append(dict(t=z_ete, ct=ct))

            rf_dist = t_gt_b.robinson_foulds(z_ete)
            norm_rf = rf_dist[0] / rf_dist[1]
            rf_df.append(
                dict(
                    ct=ct,
                    model=model,
                    rf=norm_rf,
                )
            )
rf_df = pd.DataFrame(rf_df)

# %%

for ct in [0, 1]:
    subs = rf_df.query(f"ct == '{ct}'")
    pop1 = subs.query("model == 'MrVILinearLinear10'").rf
    pop2 = subs.query("model == 'CompositionSCVI'").rf
    pop3 = subs.query("model == 'CompositionPCA'").rf

    pval21 = stats.ttest_rel(pop1, pop2, alternative="less").pvalue
    print("mrVI vs CompositionSCVI", pval21)

    pval31 = stats.ttest_rel(pop1, pop3, alternative="less").pvalue
    print("mrVI vs CPCA", pval31)
    print()

# %%
rf_subplot = rf_df.loc[
    lambda x: x["model"].isin(
        [
            "CompositionSCVI",
            "CompositionPCA",
            "MrVILinearLinear10",
        ]
    )
]

fig = (
    p9.ggplot(rf_subplot, p9.aes(x="model", y="rf"))
    + p9.geom_bar(p9.aes(fill="model"), stat="summary")
    + p9.coord_flip()
    + p9.facet_wrap("~ct")
    + p9.theme_classic()
    + p9.theme(
        axis_text=p9.element_text(size=12),
        axis_title=p9.element_text(size=15),
        aspect_ratio=1,
        strip_background=p9.element_blank(),
        legend_position="none",
    )
    + p9.labs(x="", y="Robinson-Foulds Distance")
    + p9.scale_y_continuous(expand=[0, 0])
)
fig.save(os.path.join(figure_dir, "semisynth_rf_dist.svg"), verbose=False)
fig


# %%
def correct_for_mult(x):
    # print(x.name)
    if x.name.startswith("MILO"):
        print("MILO already FDR controlled; skipping")
        return x
    return multipletests(x, method="fdr_bh")[1]


padj_dfs = []
ct_obs = adata.obs[adata.uns["mapper"]["cell_type_key"]]
for model in model_adatas:
    for model_adata in model_adatas[model]:
        adata = model_adata["adata"]
        # Selecting right columns in adata.obs
        sig_keys = [col for col in adata.obs.columns if col.endswith("significance")]
        # Adjust for multiple testing if needed
        padj_dfs.append(adata.obs.loc[:, sig_keys].apply(correct_for_mult, axis=0))

# Select right cell subpopulations
padjs = pd.concat(padj_dfs, axis=1)
target_a = ct_obs == "1"
good_cols_a = [col for col in padjs.columns if "tree_id2" in col]


ALPHA = 0.05


def get_pr(padj, target):
    fdr = 1.0 - precision_score(target, padj <= ALPHA, zero_division=1)
    tpr = recall_score(target, padj <= ALPHA, zero_division=1)
    res = pd.Series(dict(FDP=fdr, TPR=tpr))
    return res


plot_df = (
    padjs.loc[:, good_cols_a]
    .apply(get_pr, target=target_a, axis=0)
    .stack()
    .to_frame("score")
    .reset_index()
    .rename(columns=dict(level_0="metric", level_1="approach"))
    .assign(
        model=lambda x: x.approach.str.split("_").str[0],
        test=lambda x: x.approach.str.split("_").str[-2],
        model_test=lambda x: x.model + " " + x.test,
        metadata=lambda x: x.approach.str.split("_")
        .str[1:-2]
        .apply(lambda y: "_".join(y)),
    )
)
plot_df

# %%
MODEL_SELECTION = [
    "MrVILinearLinear10",
    "MILOSCVI",
]

plot_df_ = plot_df.loc[
    lambda x: (x["model"].isin(MODEL_SELECTION))
    & (
        x["test"].isin(
            (
                "ks",
                "LFC",
            )
        )
    )
    & (x["metadata"] != "batch")
]

fig = (
    p9.ggplot(
        plot_df_.query('metric == "TPR"'),
        p9.aes(x="factor(metadata)", y="score", fill="model_test"),
    )
    + p9.geom_bar(stat="identity", position="dodge")
    # + p9.facet_wrap("~metadata", scales="free")
    # + p9.coord_flip()
    + p9.theme_classic()
    + p9.theme(
        axis_text=p9.element_text(size=12),
        axis_title=p9.element_text(size=15),
        aspect_ratio=1.5,
    )
    + p9.labs(x="", y="TPR", fill="")
    + p9.scale_y_continuous(expand=(0, 0))
)
fig.save(
    os.path.join(figure_dir, "semisynth_TPR_comparison_synth.svg"),
)
fig

# %%
fig = (
    p9.ggplot(
        plot_df_.query('metric == "FDP"'),
        p9.aes(x="factor(metadata)", y="score", fill="model_test"),
    )
    + p9.geom_bar(stat="identity", position="dodge")
    # + p9.facet_wrap("~metadata", scales="free")
    # + p9.coord_flip()
    + p9.theme_classic()
    + p9.labs(x="", y="FDP", fill="")
    + p9.theme(
        axis_text=p9.element_text(size=12),
        axis_title=p9.element_text(size=15),
        aspect_ratio=1.5,
    )
    + p9.scale_y_continuous(expand=(0, 0))
)
fig.save(
    os.path.join(figure_dir, "semisynth_FDR_comparison_synth.svg"),
)
fig
