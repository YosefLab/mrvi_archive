import numpy as np

from scvi.data import synthetic_iid

from mrvi import MrVI


def test_mrvi():
    adata = synthetic_iid()
    adata.obs["donor"] = np.random.choice(15, size=adata.shape[0])
    MrVI.setup_anndata(adata, batch_key="donor", categorical_nuisance_keys=["batch"])
    for linear_decoder_uz in [True, False]:
        for linear_decoder_zx in [True, False]:
            model = MrVI(
                adata,
                observe_library_sizes=True,
                n_latent_donor=5,
                linear_decoder_zx=linear_decoder_zx,
                linear_decoder_uz=linear_decoder_uz,
            )
            model.train(1, check_val_every_n_epoch=1, train_size=0.5)
            model.history

    model = MrVI(
        adata,
        observe_library_sizes=True,
        n_latent_donor=5,
        linear_decoder_zx=True,
        linear_decoder_uz=True,
        linear_decoder_uz_scaler=True,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_latent_representation()
    assert model.get_local_sample_representation().shape == (adata.shape[0], 15, 10)
    assert model.get_local_sample_representation(return_distances=True).shape == (
        adata.shape[0],
        15,
        15,
    )
    # tests __repr__
    print(model)
