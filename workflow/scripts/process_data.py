import itertools
import string

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def create_joint_obs_key(adata, keys):
    joint_key_name = "_".join(keys)
    adata.obs[joint_key_name] = (
        adata.obs[keys].astype(str).apply(lambda x: "-".join(x), axis=1)
    )
    return joint_key_name


def make_categorical(adata, obs_key):
    adata.obs[obs_key] = adata.obs[obs_key].astype("category")


def create_obs_mapper(adata, dataset_name):
    dataset_config = snakemake.config[snakemake.wildcards.dataset]["keyMapping"]

    donor_key = dataset_config["donorKey"]
    if isinstance(donor_key, list):
        donor_key = create_joint_obs_key(adata, donor_key)
    make_categorical(adata, donor_key)

    cell_type_key = dataset_config["cellTypeKey"]
    make_categorical(adata, cell_type_key)

    nuisance_keys = dataset_config["nuisanceKeys"]
    for nuisance_key in nuisance_keys:
        make_categorical(adata, nuisance_key)

    adata.uns["mapper"] = dict(
        donor_key=donor_key,
        categorical_nuisance_keys=nuisance_keys,
        cell_type_key=cell_type_key,
    )


def assign_symsim_donors(adata):
    np.random.seed(1)
    dataset_config = snakemake.config["synthetic"]["keyMapping"]
    donor_key = dataset_config["donorKey"]
    batch_key = dataset_config["nuisanceKeys"][0]

    n_donors = 32
    n_meta = len([k for k in adata.obs.keys() if "meta_" in k])
    meta_keys = [f"meta_{i + 1}" for i in range(n_meta)]
    make_categorical(adata, batch_key)
    batches = adata.obs[batch_key].cat.categories.tolist()
    n_batch = len(batches)

    meta_combos = list(itertools.product([0, 1], repeat=n_meta))
    donors_per_meta_batch_combo = n_donors // len(meta_combos) // n_batch

    # Assign donors uniformly at random for cells with matching metadata.
    donor_assignment = np.empty(adata.n_obs, dtype=object)
    for batch in batches:
        batch_donors = []
        for meta_combo in meta_combos:
            match_cats = [f"CT{meta_combo[i]+1}:1" for i in range(n_meta)]
            eligible_cell_idxs = (
                (
                    np.all(
                        adata.obs[meta_keys].values == match_cats,
                        axis=1,
                    )
                    & (adata.obs[batch_key] == batch)
                )
                .to_numpy()
                .nonzero()[0]
            )
            meta_donors = [
                f"donor_meta{meta_combo}_batch{batch}_{ch}"
                for ch in string.ascii_lowercase[:donors_per_meta_batch_combo]
            ]
            donor_assignment[eligible_cell_idxs] = np.random.choice(
                meta_donors, replace=True, size=len(eligible_cell_idxs)
            )
            batch_donors += meta_donors

    adata.obs[donor_key] = donor_assignment

    donor_meta = adata.obs[donor_key].str.extractall(
        r"donor_meta\(([0-1]), ([0-1]), ([0-1])\)_batch[0-9]_[a-z]"
    )
    for match_idx, meta_key in enumerate(meta_keys):
        adata.obs[f"donor_{meta_key}"] = donor_meta[match_idx].astype(int).tolist()


def construct_tree_semisynth(adata, depth_tree=3, dataset_name="semisynthetic"):
    """Modifies gene expression in two cell subpopulations according to a controlled
    donor tree structure

    """
    # construct donors
    n_donors = int(2**depth_tree)
    np.random.seed(0)
    random_donors = np.random.randint(0, n_donors, adata.n_obs)
    n_modules = sum([int(2**k) for k in range(1, depth_tree + 1)])

    # ct_key = snakemake.config[dataset_name]["keyMapping"]["cellTypeKey"]
    # cts = adata.obs.groupby(ct_key).size().sort_values(ascending=False)[:2]
    # ct1, ct2 = cts.index.values
    ct_key = "leiden"
    ct1, ct2 = "0", "1"

    # construct donor trees
    leaves_id = np.array(
        [format(i, "0{}b".format(depth_tree)) for i in range(n_donors)]
    )  # ids of leaves

    all_node_ids = []
    for dep in range(1, depth_tree + 1):
        node_ids = [format(i, "0{}b".format(dep)) for i in range(2**dep)]
        all_node_ids += node_ids  # ids of all nodes in the tree

    def perturb_gene_exp(ct, X_perturbed, all_node_ids, leaves_id):
        leaves_id1 = leaves_id.copy()
        np.random.shuffle(leaves_id1)
        genes = np.arange(adata.n_vars)
        np.random.shuffle(genes)
        gene_modules = np.array_split(genes, n_modules)
        gene_modules = {
            node_id: gene_modules[i] for i, node_id in enumerate(all_node_ids)
        }
        gene_modules = {
            node_id: np.isin(np.arange(adata.n_vars), gene_modules[node_id])
            for node_id in all_node_ids
        }
        # convert to one hots to make life easier for saving

        # modifying gene expression
        # e.g., 001 has perturbed modules 0, 00, and 001
        subpop = adata.obs.loc[:, ct_key].values == ct
        print("perturbing {}".format(ct))
        for donor_id in range(n_donors):
            selected_pop = subpop & (random_donors == donor_id)
            leaf_id = leaves_id1[donor_id]
            perturbed_mod_ids = [leaf_id[:i] for i in range(1, depth_tree + 1)]
            perturbed_modules = np.zeros(adata.n_vars, dtype=bool)
            for id in perturbed_mod_ids:
                perturbed_modules = perturbed_modules | gene_modules[id]

            Xmat = X_perturbed[selected_pop].copy()
            print(
                "Perturbing {} genes in {} cells".format(
                    perturbed_modules.sum(), selected_pop.sum()
                )
            )
            print(
                "Non-zero values in the relevant subpopulation and modules:   ",
                (Xmat[:, perturbed_modules] != 0).sum(),
            )
            Xmat[:, perturbed_modules] = Xmat[:, perturbed_modules] * 2

            X_perturbed[selected_pop] = Xmat
        return X_perturbed, gene_modules, leaves_id1

    X_pert = adata.X.copy()
    X_pert, gene_mod1, leaves1 = perturb_gene_exp(ct1, X_pert, all_node_ids, leaves_id)
    X_pert, gene_mod2, leaves2 = perturb_gene_exp(ct2, X_pert, all_node_ids, leaves_id)

    gene_modules1 = pd.DataFrame(gene_mod1)
    gene_modules2 = pd.DataFrame(gene_mod2)
    donor_metadata = pd.DataFrame(
        dict(
            donor_id=np.arange(n_donors),
            tree_id1=leaves1,
            affected_ct1=ct1,
            tree_id2=leaves2,
            affected_ct2=ct2,
        )
    )

    meta_id1 = pd.DataFrame([list(x) for x in donor_metadata.tree_id1.values]).astype(
        int
    )
    n_features1 = meta_id1.shape[1]
    new_cols1 = ["tree_id1_{}".format(i) for i in range(n_features1)]
    donor_metadata.loc[:, new_cols1] = meta_id1.values
    meta_id2 = pd.DataFrame([list(x) for x in donor_metadata.tree_id2.values]).astype(
        int
    )
    n_features2 = meta_id2.shape[1]
    new_cols2 = ["tree_id2_{}".format(i) for i in range(n_features2)]
    donor_metadata.loc[:, new_cols2] = meta_id2.values

    donor_metadata.loc[:, new_cols1 + new_cols2] = donor_metadata.loc[
        :, new_cols1 + new_cols2
    ].astype("category")

    adata.obs.loc[:, "batch"] = random_donors
    adata.obs.loc[:, "Site"] = 1
    original_index = adata.obs.index.copy()
    adata.obs = adata.obs.merge(
        donor_metadata, left_on="batch", right_on="donor_id", how="left"
    )
    adata.obs.index = original_index
    adata.uns["gene_modules1"] = gene_modules1
    adata.uns["gene_modules2"] = gene_modules2
    adata.uns["donor_metadata"] = donor_metadata
    adata = sc.AnnData(X_pert, obs=adata.obs, var=adata.var, uns=adata.uns)
    return adata


def process_snrna(adata):
    # We first remove samples processed with specific protocols that
    # are not used in other samples
    select_cell = ~(adata.obs["sample"].isin(["C41_CST", "C41_NST"]))
    adata = adata[select_cell].copy()

    # We also artificially subsample
    # one of the samples into two samples
    # as a negative control
    subplit_sample = "C72_RESEQ"
    mask_selected_lib = (adata.obs["sample"] == subplit_sample).values
    np.random.seed(0)
    mask_split = (
        np.random.randint(0, 2, size=mask_selected_lib.shape[0]).astype(bool)
        * mask_selected_lib
    )
    libraries = adata.obs["library_uuid"].astype(str)
    libraries.loc[mask_split] = libraries.loc[mask_split] + "_split"
    assert libraries.unique().shape[0] == 9
    adata.obs.loc[:, "library_uuid"] = libraries.astype("category")
    return adata


def process_dataset(dataset_name, input_h5ad, output_h5ad):
    adata = sc.read_h5ad(input_h5ad)

    if isinstance(adata.X, sp.csc_matrix):
        adata.X = adata.X.tocsr()

    dataset_info = snakemake.config[dataset_name]
    preprocessing_kwargs = dataset_info.get("preprocessing")
    if preprocessing_kwargs is not None:
        if "subsample" in preprocessing_kwargs:
            sc.pp.subsample(adata, **preprocessing_kwargs.get("subsample"))
        if "filter_genes" in preprocessing_kwargs:
            sc.pp.filter_genes(adata, **preprocessing_kwargs.get("filter_genes"))
        if "highly_variable_genes" in preprocessing_kwargs:
            sc.pp.highly_variable_genes(
                adata, **preprocessing_kwargs.get("highly_variable_genes")
            )
            adata = adata[:, adata.var.highly_variable].copy()
    if dataset_name == "synthetic":
        assign_symsim_donors(adata)
    elif dataset_name == "semisynthetic":
        adata = construct_tree_semisynth(
            adata,
            depth_tree=4,
        )
    elif dataset_name == "snrna":
        adata = process_snrna(adata)

    create_obs_mapper(adata, dataset_name)

    adata.write(output_h5ad)


if __name__ == "__main__":
    process_dataset(
        dataset_name=snakemake.wildcards.dataset,
        input_h5ad=snakemake.input[0],
        output_h5ad=snakemake.output[0],
    )
