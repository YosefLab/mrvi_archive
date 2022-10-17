# Multi-resolution Variational Inference

Multi-resolution Variational Inference (MrVI) is a package for analysis of sample-level heterogeneity in multi-site, multi-sample single-cell omics data. Built with [scvi-tools](https://scvi-tools.org).

---

To install, run:

```
pip install mrvi
```

`mrvi.MrVI` follows the same API used in scvi-tools.

```python
import mrvi
import anndata

adata = anndata.read_h5ad("path/to/adata.h5ad")
# Sample (e.g. donors, perturbations, etc.) should go in sample_key
# Sites, plates, and other factors should go in categorical_nuisance_keys
mrvi.MrVI.setup_anndata(adata, sample_key="donor", categorical_nuisance_keys=["site"])
mrvi_model = mrvi.MrVI(adata)
mrvi_model.train()
# Get z representation
adata.obsm["X_mrvi_z"] = mrvi_model.get_latent_representation(give_z=True)
# Get u representation
adata.obsm["X_mrvi_u"] = mrvi_model.get_latent_representation(give_z=False)
# Cells by n_sample by n_latent
cell_sample_representations = mrvi_model.get_local_sample_representation()
# Cells by n_sample by n_sample
cell_sample_sample_distances = mrvi_model.get_local_sample_representation(return_distances=True)
```

## Citation

```
@article {Boyeau2022.10.04.510898,
	author = {Boyeau, Pierre and Hong, Justin and Gayoso, Adam and Jordan, Michael and Azizi, Elham and Yosef, Nir},
	title = {Deep generative modeling for quantifying sample-level heterogeneity in single-cell omics},
	elocation-id = {2022.10.04.510898},
	year = {2022},
	doi = {10.1101/2022.10.04.510898},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Contemporary single-cell omics technologies have enabled complex experimental designs incorporating hundreds of samples accompanied by detailed information on sample-level conditions. Current approaches for analyzing condition-level heterogeneity in these experiments often rely on a simplification of the data such as an aggregation at the cell-type or cell-state-neighborhood level. Here we present MrVI, a deep generative model that provides sample-sample comparisons at a single-cell resolution, permitting the discovery of subtle sample-specific effects across cell populations. Additionally, the output of MrVI can be used to quantify the association between sample-level metadata and cell state variation. We benchmarked MrVI against conventional meta-analysis procedures on two synthetic datasets and one real dataset with a well-controlled experimental structure. This work introduces a novel approach to understanding sample-level heterogeneity while leveraging the full resolution of single-cell sequencing data.Competing Interest StatementN.Y. is an advisor and/or has equity in Cellarity, Celsius Therapeutics, and Rheos Medicine.},
	URL = {https://www.biorxiv.org/content/early/2022/10/06/2022.10.04.510898},
	eprint = {https://www.biorxiv.org/content/early/2022/10/06/2022.10.04.510898.full.pdf},
	journal = {bioRxiv}
}
```
