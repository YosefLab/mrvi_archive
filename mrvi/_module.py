from __future__ import annotations

import torch
import torch.distributions as db
import torch.nn as nn
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import one_hot
from torch.distributions import kl_divergence as kl

from ._components import DecoderUZ, DecoderZX, LinearDecoderUZ
from ._constants import MRVI_REGISTRY_KEYS
from ._utils import ConditionalBatchNorm1d, NormalNN

DEFAULT_PX_HIDDEN = 32
DEFAULT_PZ_LAYERS = 1
DEFAULT_PZ_HIDDEN = 32


class MrVAE(BaseModuleClass):
    def __init__(
        self,
        n_input: int,
        n_sample: int,
        n_obs_per_sample: int,
        n_cats_per_nuisance_keys: list[int],
        n_latent: int = 10,
        n_latent_sample: int = 2,
        linear_decoder_zx: bool = True,
        linear_decoder_uz: bool = True,
        linear_decoder_uz_scaler: bool = False,
        linear_decoder_uz_scaler_n_hidden: int = 32,
        px_kwargs: dict | None = None,
        pz_kwargs: dict | None = None,
    ):
        super().__init__()
        px_kwargs = dict(n_hidden=DEFAULT_PX_HIDDEN)
        if px_kwargs is not None:
            px_kwargs.update(px_kwargs)
        pz_kwargs = dict(n_layers=DEFAULT_PZ_LAYERS, n_hidden=DEFAULT_PZ_HIDDEN)
        if pz_kwargs is not None:
            pz_kwargs.update(pz_kwargs)

        self.n_cats_per_nuisance_keys = n_cats_per_nuisance_keys
        self.n_sample = n_sample
        assert n_latent_sample != 0
        self.sample_embeddings = nn.Embedding(n_sample, n_latent_sample)

        n_nuisance = sum(self.n_cats_per_nuisance_keys)
        # Generative model
        self.px = DecoderZX(
            n_latent + n_nuisance,
            n_input,
            n_nuisance=n_nuisance,
            linear_decoder=linear_decoder_zx,
            **px_kwargs,
        )
        self.qu = NormalNN(128 + n_latent_sample, n_latent, n_categories=1)
        self.ql = NormalNN(n_input, 1, n_categories=1)

        self.linear_decoder_uz = linear_decoder_uz
        if linear_decoder_uz:
            self.pz = LinearDecoderUZ(
                n_latent,
                self.n_sample,
                n_latent,
                scaler=linear_decoder_uz_scaler,
                scaler_n_hidden=linear_decoder_uz_scaler_n_hidden,
            )
        else:
            self.pz = DecoderUZ(
                n_latent,
                n_latent_sample,
                n_latent,
                **pz_kwargs,
            )
        self.n_obs_per_sample = nn.Parameter(n_obs_per_sample, requires_grad=False)

        self.x_featurizer = nn.Sequential(nn.Linear(n_input, 128), nn.ReLU())
        self.bnn = ConditionalBatchNorm1d(128, n_sample)
        self.x_featurizer2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.bnn2 = ConditionalBatchNorm1d(128, n_sample)

    def _get_inference_input(
        self, tensors: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        sample_index = tensors[MRVI_REGISTRY_KEYS.SAMPLE_KEY]
        categorical_nuisance_keys = tensors[
            MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS
        ]
        return dict(
            x=x,
            sample_index=sample_index,
            categorical_nuisance_keys=categorical_nuisance_keys,
        )

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        sample_index: torch.Tensor,
        categorical_nuisance_keys: torch.Tensor,
        mc_samples: int = 1,
        cf_sample: torch.Tensor | None = None,
        use_mean: bool = False,
    ) -> dict[str, torch.Tensor]:
        x_ = torch.log1p(x)

        sample_index_cf = sample_index if cf_sample is None else cf_sample
        zsample = self.sample_embeddings(sample_index_cf.long().squeeze(-1))
        zsample_ = zsample
        if mc_samples >= 2:
            zsample_ = zsample[None].expand(mc_samples, *zsample.shape)

        nuisance_oh = []
        for dim in range(categorical_nuisance_keys.shape[-1]):
            nuisance_oh.append(
                one_hot(
                    categorical_nuisance_keys[:, [dim]],
                    self.n_cats_per_nuisance_keys[dim],
                )
            )
        nuisance_oh = torch.cat(nuisance_oh, dim=-1)

        x_feat = self.x_featurizer(x_)
        x_feat = self.bnn(x_feat, sample_index)
        x_feat = self.x_featurizer2(x_feat)
        x_feat = self.bnn2(x_feat, sample_index)
        if x_.ndim != zsample_.ndim:
            x_feat_ = x_feat[None].expand(mc_samples, *x_feat.shape)
            nuisance_oh = nuisance_oh[None].expand(mc_samples, *nuisance_oh.shape)
        else:
            x_feat_ = x_feat

        inputs = torch.cat([x_feat_, zsample_], -1)
        # inputs = x_feat_
        qu = self.qu(inputs)
        if use_mean:
            u = qu.loc
        else:
            u = qu.rsample()

        if self.linear_decoder_uz:
            z = self.pz(u, sample_index_cf)
        else:
            inputs = torch.cat([u, zsample_], -1)
            z = self.pz(inputs)
        library = torch.log(x.sum(1)).unsqueeze(1)

        return dict(
            qu=qu,
            u=u,
            z=z,
            zsample=zsample,
            library=library,
            nuisance_oh=nuisance_oh,
        )

    def get_z(
        self,
        u: torch.Tensor,
        zsample: torch.Tensor | None = None,
        sample_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sample_index is not None:
            zsample = self.sample_embeddings(sample_index.long().squeeze(-1))
            zsample = zsample
        else:
            zsample_ = zsample
        inputs = torch.cat([u, zsample_], -1)
        z = self.pz(inputs)
        return z

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        res = dict(
            z=inference_outputs["z"],
            library=inference_outputs["library"],
            nuisance_oh=inference_outputs["nuisance_oh"],
        )

        return res

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        nuisance_oh: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = torch.concat([z, nuisance_oh], dim=-1)
        px = self.px(inputs, size_factor=library.exp())
        h = px.mu / library.exp()

        pu = db.Normal(0, 1)
        return dict(px=px, pu=pu, h=h)

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        reconstruction_loss = (
            -generative_outputs["px"].log_prob(tensors[REGISTRY_KEYS.X_KEY]).sum(-1)
        )
        kl_u = kl(inference_outputs["qu"], generative_outputs["pu"]).sum(-1)
        kl_local_for_warmup = kl_u

        weighted_kl_local = kl_weight * kl_local_for_warmup
        loss = torch.mean(reconstruction_loss + weighted_kl_local)

        kl_local = torch.tensor(0.0)
        kl_global = torch.tensor(0.0)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local=kl_local,
            kl_global=kl_global,
        )
