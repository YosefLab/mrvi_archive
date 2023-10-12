# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
import scipy.stats as stats
from scib.metrics import silhouette, silhouette_batch
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

METRIC = "euclidean"


def compute_aggregate_dmat(reps, metric="cosine"):
    n_cells, n_donors, _ = reps.shape
    pairwise_ds = np.zeros((n_donors, n_donors))
    for x in tqdm(reps):
        d_ = pairwise_distances(x, metric=metric)
        pairwise_ds += d_ / n_cells
    return pairwise_ds


def return_distances_cluster_rep(rep, metric="cosine"):
    ordered_metad = metad.set_index(sample_key).loc[rep.index]
    ss_matrix = pairwise_distances(rep.values, metric=metric)
    cats = ordered_metad[bio_group_key].values[:, None]
    suspension_cats = ordered_metad[techno_key].values[:, None]
    return ss_matrix, cats, suspension_cats, ordered_metad


def return_distances_cell_specific(rep, good_cells, metric="cosine"):
    good_cells = adata.obs[ct_key] == cluster
    rep_ct = rep[good_cells]
    subobs = adata.obs[good_cells]

    observed_donors = subobs[sample_key].value_counts()
    observed_donors = observed_donors[observed_donors >= 1].index
    meta_ = metad.reset_index()
    good_d_idx = meta_.loc[lambda x: x[sample_key].isin(observed_donors)].index.values
    ss_matrix = compute_aggregate_dmat(rep_ct[:, good_d_idx, :], metric=metric)
    cats = meta_.loc[good_d_idx][bio_group_key].values[:, None]
    suspension_cats = meta_.loc[good_d_idx][techno_key].values[:, None]
    return ss_matrix, cats, suspension_cats, meta_.loc[good_d_idx]


# %%
dataset_name = "snrna"

base_dir = "workflow/results_V1/{}".format(dataset_name)
full_data_path = "workflow/data/{}/adata.processed.h5ad".format(dataset_name)
model_path = os.path.join(base_dir, "MrVI")

input_files = glob.glob(os.path.join(base_dir, "final_adata*"))
input_files

# %%
adata = sc.read_h5ad(input_files[0])
print(adata.shape)
for file in tqdm(input_files[1:]):
    adata_ = sc.read_h5ad(file)
    print(file, adata_.shape)
    new_cols = np.setdiff1d(adata_.obs.columns, adata.obs.columns)
    adata.obs.loc[:, new_cols] = adata_.obs.loc[:, new_cols]

    new_uns = np.setdiff1d(list(adata_.uns.keys()), list(adata.uns.keys()))
    for new_uns_ in new_uns:
        uns_vals = adata_.uns[new_uns_]
        if new_uns_.endswith("local_donor_rep"):
            meta_keys = [
                key
                for key in adata_.uns.keys()
                if key.endswith("_local_donor_rep_metadata")
            ]
            print(new_uns_)
            if len(meta_keys) != 0:
                # SORT UNS by donor_key values
                meta_key = meta_keys[0]
                print(new_uns_, meta_key)
                donor_key = adata_.uns["mapper"]["donor_key"]
                metad = adata_.uns[meta_key].reset_index().sort_values(donor_key)
                print(metad.index.values)
                uns_vals = uns_vals[:, metad.index.values]
            else:
                uns_vals = adata_.uns[new_uns_]
                for key in uns_vals:
                    uns_vals[key] = uns_vals[key].sort_index()
                print(uns_vals[key].index)

        adata.uns[new_uns_] = uns_vals

    new_obsm = np.setdiff1d(list(adata_.obsm.keys()), list(adata.obsm.keys()))
    for new_obsm_ in new_obsm:
        adata.obsm[new_obsm_] = adata_.obsm[new_obsm_]


# %%
mapper = adata.uns["mapper"]
ct_key = mapper["cell_type_key"]
sample_key = mapper["donor_key"]
bio_group_key = "donor_uuid"
techno_key = "suspension_type"

adata.obs.loc[:, "Sample_id"] = adata.obs["sample"].str[:3]
metad.loc[:, "Sample_id"] = metad["sample"].str[:3]
metad.loc[:, "Sample_id"] = metad.loc[:, "Sample_id"].astype("category")
# %%
adata.obs["sample"].value_counts()


# %%
# Unservised analysis
MODELS = [
    dict(model_name="CompositionPCA", cell_specific=False),
    dict(model_name="CompositionSCVI", cell_specific=False),
    dict(model_name="MrVILinearLinear10", cell_specific=True),
]
# %%
_dd_plot_df = pd.DataFrame()

for model_params in MODELS:
    rep_key = "{}_local_donor_rep".format(model_params["model_name"])
    metadata_key = "{}_local_donor_rep_metadata".format(model_params["model_name"])
    is_cell_specific = model_params["cell_specific"]
    rep = adata.uns[rep_key]

    for cluster in adata.obs[ct_key].unique():
        if not is_cell_specific:
            ss_matrix, cats, suspension_cats, meta_ = return_distances_cluster_rep(
                rep[cluster], metric=METRIC
            )
        else:
            good_cells = adata.obs[ct_key] == cluster
            ss_matrix, cats, suspension_cats, meta_ = return_distances_cell_specific(
                rep, good_cells, metric=METRIC
            )

        suspension_cats_oh = OneHotEncoder(sparse=False).fit_transform(suspension_cats)
        where_similar_sus = suspension_cats_oh @ suspension_cats_oh.T
        where_similar_sus = where_similar_sus.astype(bool)

        cats_oh = OneHotEncoder(sparse=False).fit_transform(cats)
        where_similar = cats_oh @ cats_oh.T
        n_donors = where_similar.shape[0]
        # where_similar = (where_similar - np.eye(where_similar.shape[0])).astype(bool)
        where_similar = where_similar.astype(bool)
        print(where_similar.shape)
        offdiag = ~np.eye(where_similar.shape[0], dtype=bool)

        if "library_uuid" not in meta_.columns:
            meta_ = meta_.reset_index()
        library_names = np.concatenate(
            np.array(
                [n_donors * [lib_name] for lib_name in meta_["library_uuid"].values]
            )[None]
        )

        new_vals = pd.DataFrame(
            dict(
                dist=ss_matrix.reshape(-1),
                is_similar=where_similar.reshape(-1),
                has_similar_suspension=where_similar_sus.reshape(-1),
                library_name1=library_names.reshape(-1),
                library_name2=library_names.T.reshape(-1),
            )
        ).assign(model=model_params["model_name"], cluster=cluster)
        _dd_plot_df = pd.concat([_dd_plot_df, new_vals], axis=0)

# %%
# Construct metadata
final_meta = metad.copy()
final_meta.loc[:, "donor_name"] = metad["sample"].str[:3]

lib_to_sample_name = pd.Series(
    {
        "24723d89-8db6-4e5b-a227-5805b49bb8e6": "C41",
        "4059d4aa-b0d5-4b88-92f3-f5623e744c2f": "C58_TST",
        "7bdadd5c-74cc-4aef-9baf-cd2a75382a0c": "C58_RESEQ",
        "7ec5239b-b687-46ea-9c6b-9e2ea970ba21": "C72_RESEQ",
        "7ec5239b-b687-46ea-9c6b-9e2ea970ba21_split": "C72_RESEQ_split",
        "c557eece-31dc-4825-83c6-7af195076696": "C41_TST",
        "d62041ea-a566-4b7b-8280-2a8e5f776270": "C70_RESEQ",
        "da945071-1938-4ed5-b0fb-bcf5eef6f92f": "C70_TST",
        "f4a052f1-ffd8-4372-ae45-777811d945ee": "C72_TST",
    }
).to_frame("sample_name")
final_meta = final_meta.merge(
    lib_to_sample_name, left_on="library_uuid", right_index=True
).assign(
    sample_name1=lambda x: x["sample_name"],
    sample_name2=lambda x: x["sample_name"],
    donor_name1=lambda x: x["donor_name"],
    donor_name2=lambda x: x["donor_name"],
)


dd_plot_df_full = (
    _dd_plot_df.merge(
        final_meta.loc[:, ["library_uuid", "sample_name1", "donor_name1"]],
        left_on="library_name1",
        right_on="library_uuid",
        how="left",
    )
    .merge(
        final_meta.loc[:, ["library_uuid", "sample_name2", "donor_name2"]],
        left_on="library_name2",
        right_on="library_uuid",
        how="left",
    )
    .assign(
        sample_name1=lambda x: pd.Categorical(
            x["sample_name1"], categories=np.sort(x.sample_name1.unique())
        ),
        sample_name2=lambda x: pd.Categorical(
            x["sample_name2"], categories=np.sort(x.sample_name2.unique())
        ),
        donor_name1=lambda x: pd.Categorical(
            x["donor_name1"], categories=np.sort(x.donor_name1.unique())
        ),
        donor_name2=lambda x: pd.Categorical(
            x["donor_name2"], categories=np.sort(x.donor_name2.unique())
        ),
    )
)
dd_plot_df_full.loc[:, "sample_name2_r"] = pd.Categorical(
    dd_plot_df_full["sample_name2"].values,
    categories=dd_plot_df_full["sample_name2"].cat.categories[::-1],
)
dd_plot_df = dd_plot_df_full.loc[lambda x: x.library_name1 != x.library_name2]


# %%
for model in dd_plot_df.model.unique():
    plot_ = dd_plot_df_full.query("cluster == 'periportal region hepatocyte'").loc[
        lambda x: x.model == model
    ]
    vmin, vmax = np.quantile(plot_.dist, [0.2, 0.8])
    plot_.loc[:, "dist_clip"] = np.clip(plot_.dist, vmin, vmax)
    fig = (
        p9.ggplot(p9.aes(x="sample_name1", y="sample_name2_r"))
        + p9.geom_raster(plot_, p9.aes(fill="dist_clip"))
        + p9.geom_tile(
            plot_.query("is_similar"),
            color="#ff3f05",
            fill="none",
            size=2,
        )
        + p9.theme_void()
        + p9.theme(
            axis_ticks=p9.element_blank(),
        )
        + p9.scale_y_discrete()
        + p9.labs(
            title=model,
            x="",
            y="",
        )
        + p9.coord_flip()
        + p9.scale_fill_cmap("viridis")
    )
    fig.save("figures/snrna_heatmaps_{}.svg".format(model))
    fig.draw()


# %%
gp1 = dd_plot_df.groupby(["model", "cluster", "donor_name1"])
v1 = gp1.apply(lambda x: x.query("is_similar").dist.median())
v2 = gp1.apply(
    lambda x: x.loc[lambda x: x.has_similar_suspension & ~(x.is_similar)].dist.median()
)

min_dist = (
    dd_plot_df.query("sample_name1 == 'C72_RESEQ'")
    .query("sample_name2 == 'C72_RESEQ_split'")
    .groupby(["model", "cluster"])
    .dist.mean()
    .to_frame("min_dist")
    .reset_index()
    # averages the two pairwise distances
)
v2_s = v2.to_frame("denom_dist").reset_index().query("donor_name1 == 'C72'")
lower_bound_ratio = min_dist.merge(v2_s, on=["model", "cluster"]).assign(
    lower_bound_ratio=lambda x: x.min_dist / x.denom_dist
)


ratio = (v1 / v2).to_frame("ratio").reset_index().dropna()

SUBSELECTED_MODELS = [
    "MrVILinearLinear10",
    "CompositionPCA",
    "CompositionSCVI",
]
ratio_plot = ratio.loc[lambda x: x.model.isin(SUBSELECTED_MODELS)]
lower_bound_ratio_plot = lower_bound_ratio.loc[
    lambda x: x.model.isin(SUBSELECTED_MODELS)
]

fig = (
    p9.ggplot(ratio_plot, p9.aes(x="model", y="ratio", fill="model"))
    + p9.geom_boxplot()
    + p9.theme_classic()
    + p9.coord_flip()
    + p9.theme(legend_position="none")
    # + p9.labs(y="Ratio of mean distance between similar over dissimilar samples", x="")
)
fig

# %%


(
    p9.ggplot(ratio_plot, p9.aes(x="ratio", color="model"))
    + p9.stat_ecdf()
    + p9.xlim(0, 2)
    + p9.labs(y="ECDF", x="Ratio of mean distances")
)

# %%
vplot = pd.DataFrame(
    dict(
        ratio=[
            1,
        ],
    )
)

fig = (
    p9.ggplot(ratio_plot, p9.aes(y="ratio", x="model", fill="model"))
    + p9.facet_wrap("donor_name1")
    + p9.geom_boxplot()
    + p9.geom_hline(p9.aes(yintercept="ratio"), data=vplot, linetype="dashed", size=1)
    + p9.coord_flip()
    + p9.theme_classic()
    + p9.theme(
        legend_position="none",
        strip_background=p9.element_blank(),
        aspect_ratio=0.6,
        axis_ticks_major_y=p9.element_blank(),
        axis_title=p9.element_text(size=15),
        strip_text=p9.element_text(size=15),
        axis_text=p9.element_text(size=15),
    )
    + p9.labs(y="Ratio", x="")
)
fig.save("figures/snrna_ratios.svg")
fig


# %%
for donor_name in ratio_plot.donor_name1.unique():
    subs = ratio_plot.query("donor_name1 == @donor_name")
    pop1 = subs.query("model == 'MrVILinearLinear10'").ratio
    pop2 = subs.query("model == 'CompositionSCVI'").ratio
    pop3 = subs.query("model == 'CompositionPCA'").ratio

    print(donor_name)
    pval21 = stats.ttest_rel(pop1, pop2, alternative="less").pvalue
    print("mrVI vs CompositionSCVI", pval21)

    pval31 = stats.ttest_rel(pop1, pop3, alternative="less").pvalue
    print("mrVI vs CPCA", pval31)
    print()


# %%
MODELS


def return_distances_cell_specific(rep, good_cells, metric="cosine"):
    good_cells = adata.obs[ct_key] == cluster
    rep_ct = rep[good_cells]
    subobs = adata.obs[good_cells]

    observed_donors = subobs[sample_key].value_counts()
    observed_donors = observed_donors[observed_donors >= 1].index
    meta_ = metad.reset_index()
    good_d_idx = meta_.loc[lambda x: x[sample_key].isin(observed_donors)].index.values
    ss_matrix = compute_aggregate_dmat(rep_ct[:, good_d_idx, :], metric=metric)
    cats = meta_.loc[good_d_idx][bio_group_key].values[:, None]
    suspension_cats = meta_.loc[good_d_idx][techno_key].values[:, None]
    return ss_matrix, cats, suspension_cats, meta_.loc[good_d_idx]


# %%
_dd_plot_df = pd.DataFrame()

MODELS = [
    dict(model_name="CompositionPCA", cell_specific=False),
    dict(model_name="CompositionSCVI", cell_specific=False),
    dict(model_name="MrVILinearLinear10", cell_specific=True),
]


rdm_idx = np.random.choice(adata.n_obs, 10000, replace=False)

for model_params in MODELS:
    rep_key = "{}_local_donor_rep".format(model_params["model_name"])
    metadata_key = "{}_local_donor_rep_metadata".format(model_params["model_name"])
    is_cell_specific = model_params["cell_specific"]
    rep = adata.uns[rep_key]

    rep_ = rep[rdm_idx]
    ss_matrix = compute_aggregate_dmat(rep_, metric=METRIC)
    ds = []
    donors = metad.Sample_id.cat.codes.values[..., None]
    select_num = OneHotEncoder(sparse=False).fit_transform(donors)
    where_same_d = select_num @ select_num.T

    donors = metad.Sample_id.cat.codes.values[..., None]
    select_num = OneHotEncoder(sparse=False).fit_transform(donors)
    where_same_d = select_num @ select_num.T
    for x in tqdm(rep_):
        d_ = pairwise_distances(x, metric=METRIC)

# %%
cell_reps = [
    "MrVILinearLinear10_cell",
    "SCVI_cell",
]

mixing_df = []

for rep in cell_reps:
    algo_name = rep.split("_")[0]
    batch_aws = silhouette_batch(
        adata, "suspension_type", "author_cell_type", rep, verbose=False
    )
    sample_aws = silhouette_batch(
        adata, "library_uuid", "author_cell_type", rep, verbose=False
    )
    ct_asw = silhouette(adata, "author_cell_type", rep)

    mixing_df.append(
        dict(
            batch_aws=batch_aws,
            sample_aws=sample_aws,
            algo_name=algo_name,
            ct_asw=ct_asw,
        )
    )
mixing_df = pd.DataFrame(mixing_df)
# %%
mixing_df
# %%
for u_key in [
    "MrVILinearLinear10_cell",
]:
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=u_key)
    sc.tl.umap(adata)

    savename = "_".join([dataset_name, "u_sample_suspension_type"])
    savename += ".png"
    with plt.rc_context({"figure.dpi": 500}):
        sc.pl.umap(
            adata,
            color=["Sample_id", "suspension_type", "author_cell_type"],
            title=u_key,
            save=savename,
        )
