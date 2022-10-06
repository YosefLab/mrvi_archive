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
mrvi.MrVI.setup_anndata(adata, batch_key="donor", categorical_nuisance_keys=["site"])
mrvi_model = mrvi.MrVI(adata)
mrvi_model.train()
# Get z representation
adata.obsm["X_mrvi_z"] = mrvi_model.get_latent_representation(give_z=True)
# Get u representation
adata.obsm["X_mrvi_u"] = mrvi_model.get_latent_representation(give_z=False)
# Cells by n_samples by n_latent
cell_sample_representations = mrvi_model.get_local_sample_representation()
```
