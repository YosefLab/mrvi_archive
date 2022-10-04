import scanpy as sc


class BaseModelClass:
    has_donor_representation = False
    has_cell_representation = False
    has_local_donor_representation = False
    has_custom_representation = False
    has_save = False

    def __init__(
        self,
        adata,
        cell_type_key,
        donor_key,
        categorical_nuisance_keys=None,
        n_hvg=None,
    ):
        self.adata = adata
        self.cell_type_key = cell_type_key
        self.donor_key = donor_key

        self.n_genes = self.adata.X.shape[1]
        self.n_donors = self.adata.obs[self.donor_key].unique().shape[0]
        self.categorical_nuisance_keys = categorical_nuisance_keys
        self.n_hvg = n_hvg

    def get_donor_representation(self, adata=None):
        return None

    def _filter_hvg(self):
        if (self.n_hvg is not None) and (self.n_hvg <= self.n_genes - 1):
            adata_ = self.adata.copy()
            sc.pp.highly_variable_genes(
                adata=adata_, n_top_genes=self.n_hvg, flavor="seurat_v3"
            )
            self.adata = adata_[:, self.highly_variable]

    def preprocess_data(self):
        self._filter_hvg()

    def fit(self):
        return None

    def get_cell_representation(self, adata=None):
        return None

    def save(self, save_path):
        return
