import torch
import torch.distributions as db
import torch.nn as nn
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import one_hot
from torch.distributions import kl_divergence as kl
import numpy as np

from ._utils import ConditionalBatchNorm1d, NormalNN, ResnetFC

import pdb

DEFAULT_PX_HIDDEN = 32
DEFAULT_PZ_LAYERS = 1
DEFAULT_PZ_HIDDEN = 32


class ExpActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class DecoderZX(nn.Module):
    """Parameterizes the counts likelihood for the data given the latent variables."""

    def __init__(
        self,
        n_in,
        n_out,
        n_nuisance,
        linear_decoder,
        n_hidden=128,
        activation="softmax",
    ):
        super().__init__()
        if activation == "softmax":
            activation_ = nn.Softmax(-1)
        elif activation == "softplus":
            activation_ = nn.Softplus()
        elif activation == "exp":
            activation_ = ExpActivation()
        elif activation == "sigmoid":
            activation_ = nn.Sigmoid()
        else:
            raise ValueError("activation must be one of 'softmax' or 'softplus'")
        self.linear_decoder = linear_decoder
        self.n_nuisance = n_nuisance
        self.n_latent = n_in - n_nuisance
        if linear_decoder:
            self.amat = nn.Linear(self.n_latent, n_out, bias=False)
            self.amat_site = nn.Parameter(
                torch.randn(self.n_nuisance, self.n_latent, n_out)
            )
            self.offsets = nn.Parameter(torch.randn(self.n_nuisance, n_out))
            self.dropout_ = nn.Dropout(0.1)
            self.activation_ = activation_

        else:
            self.px_mean = ResnetFC(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                activation=activation_,
            )
        self.px_r = nn.Parameter(torch.randn(n_out))

    def forward(self, z, size_factor):
        if self.linear_decoder:
            nuisance_oh = z[..., -self.n_nuisance :]
            z0 = z[..., : -self.n_nuisance]
            x1 = self.amat(z0)

            nuisance_ids = torch.argmax(nuisance_oh, -1)
            As = self.amat_site[nuisance_ids]
            z0_detach = self.dropout_(z0.detach())[..., None]
            x2 = (As * z0_detach).sum(-2)
            offsets = self.offsets[nuisance_ids]
            mu = x1 + x2 + offsets
            mu = self.activation_(mu)
        else:
            mu = self.px_mean(z)
        mu = mu * size_factor
        return NegativeBinomial(mu=mu, theta=self.px_r.exp())


class LinearDecoderUZ(nn.Module):
    def __init__(
        self,
        n_latent,
        n_donors,
        n_out,
        scaler=False,
        scaler_n_hidden=32,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_donors = n_donors
        self.n_out = n_out

        # self.amat_sample = nn.Linear(self.n_latent, n_out, bias=False)
        self.amat_sample = nn.Parameter(torch.randn(n_donors, self.n_latent, n_out))
        self.offsets = nn.Parameter(torch.randn(self.n_donors, n_out))

        self.n_donors = n_donors
        self.scaler = None
        if scaler:
            self.scaler = nn.Sequential(
                nn.Linear(n_latent + n_donors, scaler_n_hidden),
                nn.LayerNorm(scaler_n_hidden),
                nn.ReLU(),
                nn.Linear(scaler_n_hidden, 1),
                nn.Sigmoid(),
            )

    def forward(self, u, donor_id):
        donor_id_ = donor_id.long().squeeze()
        As = self.amat_sample[donor_id_]

        u_detach = u.detach()[..., None]
        z2 = (As * u_detach).sum(-2)
        offsets = self.offsets[donor_id_]
        delta = z2 + offsets
        if self.scaler is not None:
            donor_oh = one_hot(donor_id, self.n_donors)
            if u.ndim != donor_oh.ndim:
                donor_oh = donor_oh[None].expand(u.shape[0], *donor_oh.shape)
            inputs = torch.cat([u.detach(), donor_oh], -1)
            delta = delta * self.scaler(inputs)
        return u + delta


class DecoderUZ(nn.Module):
    def __init__(
        self,
        n_latent,
        n_latent_donor,
        n_out,
        unnormalized_scaler=False,
        dropout_rate=0.0,
        n_layers=1,
        n_hidden=128,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_latent_donor = n_latent_donor
        self.n_in = n_latent + n_latent_donor
        self.n_out = n_out
        self.unnormalized_scaler = unnormalized_scaler

        arch_mod = self.construct_arch(self.n_in, n_hidden, n_layers, dropout_rate) + [
            nn.Linear(n_hidden, self.n_out, bias=False)
        ]
        self.mod = nn.Sequential(*arch_mod)

        arch_scaler = self.construct_arch(
            self.n_latent, n_hidden, n_layers, dropout_rate
        ) + [nn.Linear(n_hidden, 1)]
        self.scaler = nn.Sequential(*arch_scaler)
        if not self.unnormalized_scaler:
            self.scaler.append(nn.Sigmoid())
        else:
            self.scaler.append(nn.Softplus())

    @staticmethod
    def construct_arch(n_inputs, n_hidden, n_layers, dropout_rate):
        """Initializes MLP architecture"""

        block_inputs = [
            nn.Linear(n_inputs, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
        ]

        block_inner = n_layers * [
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        ]
        return block_inputs + block_inner

    def forward(self, u):
        u_ = u.clone()
        if u_.dim() == 3:
            n_samples, n_cells, n_features = u_.shape
            u0_ = u_[:, :, : self.n_latent].reshape(-1, self.n_latent)
            u_ = u_.reshape(-1, n_features)
            pred_ = self.mod(u_).reshape(n_samples, n_cells, -1)
            scaler_ = self.scaler(u0_).reshape(n_samples, n_cells, -1)
        else:
            pred_ = self.mod(u)
            scaler_ = self.scaler(u[:, : self.n_latent])
        if self.unnormalized_scaler:
            pred_ = F.normalize(pred_, p=2, dim=-1)
        mean = u[..., : self.n_latent] + scaler_ * pred_
        return mean


class MrVAE(BaseModuleClass):
    def __init__(
        self,
        n_input,
        n_batch,
        n_obs_per_batch,
        n_cats_per_nuisance_keys,
        n_cats_per_bio_keys,
        library_log_means,
        library_log_vars,
        n_latent=10,
        n_latent_donor=2,
        observe_library_sizes=True,
        linear_decoder_zx=True,
        linear_decoder_uz=True,
        linear_decoder_uz_scaler=False,
        linear_decoder_uz_scaler_n_hidden=32,
        unnormalized_scaler=False,
        px_kwargs=None,
        pz_kwargs=None,
        max_batches_comp=30,
    ):
        super().__init__()
        px_kwargs = dict(n_hidden=DEFAULT_PX_HIDDEN)
        if px_kwargs is not None:
            px_kwargs.update(px_kwargs)
        pz_kwargs = dict(n_layers=DEFAULT_PZ_LAYERS, n_hidden=DEFAULT_PZ_HIDDEN)
        if pz_kwargs is not None:
            pz_kwargs.update(pz_kwargs)

        self.n_cats_per_nuisance_keys = n_cats_per_nuisance_keys
        self.n_cats_per_bio_keys = n_cats_per_bio_keys
        self.n_batch = n_batch
        self.max_batches_comp = np.minimum(max_batches_comp, n_batch)
        assert n_latent_donor != 0
        self.donor_embeddings = nn.Embedding(n_batch, n_latent_donor)

        self.register_buffer(
            "library_log_means", torch.from_numpy(library_log_means).float()
        )
        self.register_buffer(
            "library_log_vars", torch.from_numpy(library_log_vars).float()
        )
        n_nuisance = sum(self.n_cats_per_nuisance_keys)
        # Generative model
        self.px = DecoderZX(
            n_latent + n_nuisance,
            n_input,
            n_nuisance=n_nuisance,
            linear_decoder=linear_decoder_zx,
            **px_kwargs,
        )
        self.qu = NormalNN(128 + n_latent_donor, n_latent, n_categories=1)
        self.ql = NormalNN(n_input, 1, n_categories=1)

        self.linear_decoder_uz = linear_decoder_uz
        if linear_decoder_uz:
            self.pz = LinearDecoderUZ(
                n_latent,
                self.n_batch,
                n_latent,
                scaler=linear_decoder_uz_scaler,
                scaler_n_hidden=linear_decoder_uz_scaler_n_hidden,
            )
        else:
            self.pz = DecoderUZ(
                n_latent,
                n_latent_donor,
                n_latent,
                unnormalized_scaler=unnormalized_scaler,
                **pz_kwargs,
            )
        self.n_obs_per_batch = nn.Parameter(n_obs_per_batch, requires_grad=False)
        self.observe_library_sizes = observe_library_sizes

        self.x_featurizer = nn.Sequential(nn.Linear(n_input, 128), nn.ReLU())
        self.bnn = ConditionalBatchNorm1d(128, n_batch)
        self.x_featurizer2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.bnn2 = ConditionalBatchNorm1d(128, n_batch)

    def _get_inference_input(self, tensors, **kwargs):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        categorical_nuisance_keys = tensors["categorical_nuisance_keys"]
        return dict(
            x=x,
            batch_index=batch_index,
            categorical_nuisance_keys=categorical_nuisance_keys,
        )

    @auto_move_data
    def inference(
        self,
        x,
        batch_index,
        categorical_nuisance_keys,
        n_samples=1,
        cf_batch=None,
        use_mean=False,
    ):
        x_ = torch.log1p(x)

        batch_index_cf = batch_index if cf_batch is None else cf_batch
        zdonor = self.donor_embeddings(batch_index_cf.long().squeeze(-1))
        zdonor_ = zdonor
        if n_samples >= 2:
            zdonor_ = zdonor[None].expand(n_samples, *zdonor.shape)
        qzdonor = None

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
        x_feat = self.bnn(x_feat, batch_index)
        x_feat = self.x_featurizer2(x_feat)
        x_feat = self.bnn2(x_feat, batch_index)
        if x_.ndim != zdonor_.ndim:
            x_feat_ = x_feat[None].expand(n_samples, *x_feat.shape)
            nuisance_oh = nuisance_oh[None].expand(n_samples, *nuisance_oh.shape)
        else:
            x_feat_ = x_feat

        inputs = torch.cat([x_feat_, zdonor_], -1)
        # inputs = x_feat_
        qu = self.qu(inputs)
        if use_mean:
            u = qu.loc
        else:
            u = qu.rsample()

        if torch.isinf(u).any():
            pdb.set_trace()
        if self.linear_decoder_uz:
            z = self.pz(u, batch_index_cf)
        else:
            inputs = torch.cat([u, zdonor_], -1)
            z = self.pz(inputs)
        if self.observe_library_sizes:
            library = torch.log(x.sum(1)).unsqueeze(1)
            ql = None
        else:
            ql = self.ql(x_)
            library = ql.rsample()

        return dict(
            qu=qu,
            qzdonor=qzdonor,
            ql=ql,
            u=u,
            z=z,
            zdonor=zdonor,
            library=library,
            nuisance_oh=nuisance_oh,
        )

    def get_z(self, u, zdonor=None, batch_index=None):
        if batch_index is not None:
            zdonor = self.donor_embeddings(batch_index.long().squeeze(-1))
            zdonor_ = zdonor
        else:
            zdonor_ = zdonor
        inputs = torch.cat([u, zdonor_], -1)
        z = self.pz(inputs)
        return z

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        categorical_nuisance_keys = tensors["categorical_nuisance_keys"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        res = dict(
            z=inference_outputs["z"],
            zdonor=inference_outputs["zdonor"],
            library=inference_outputs["library"],
            batch_index=batch_index,
            categorical_nuisance_keys=categorical_nuisance_keys,
            nuisance_oh=inference_outputs["nuisance_oh"],
        )

        return res

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        zdonor,
        categorical_nuisance_keys,
        nuisance_oh,
    ):

        pzdonor = None
        inputs = torch.concat([z, nuisance_oh], dim=-1)
        px = self.px(inputs, size_factor=library.exp())
        h = px.mu / library.exp()

        (
            local_library_log_means,
            local_library_log_vars,
        ) = self._compute_local_library_params(batch_index)
        pl = (
            None
            if self.observe_library_sizes
            else db.Normal(local_library_log_means, local_library_log_vars.sqrt())
        )

        pu = db.Normal(0, 1)
        return dict(pzdonor=pzdonor, px=px, pl=pl, pu=pu, h=h)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        lambd=None,
        do_comp=False,
    ):

        reconstruction_loss = (
            -generative_outputs["px"].log_prob(tensors[REGISTRY_KEYS.X_KEY]).sum(-1)
        )
        kl_u = kl(inference_outputs["qu"], generative_outputs["pu"]).sum(-1)
        kl_local_for_warmup = kl_u

        if self.observe_library_sizes:
            kl_local_no_warmup = 0.0
        else:
            kl_local_no_warmup = kl(
                inference_outputs["ql"], generative_outputs["pl"]
            ).sum(-1)

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        loss = torch.mean(reconstruction_loss + weighted_kl_local)

        pen = torch.tensor(0.0, device=self.device)
        if (lambd is not None) and do_comp:
            log_qz = (
                inference_outputs["qu"]
                .log_prob(inference_outputs["u"][:, None])
                .sum(-1)
            )
            samples = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze()
            unique_samples = torch.unique(samples).cpu().numpy()
            n_batches_to_take = np.minimum(
                unique_samples.shape[0], self.max_batches_comp
            )
            unique_samples = np.random.choice(
                unique_samples, size=n_batches_to_take, replace=False
            )
            for batch_index in unique_samples:
                set_s = samples == batch_index
                set_ms = samples != batch_index
                log_qz_foreg = (
                    torch.logsumexp(log_qz[set_s][:, set_s], 1) - set_s.sum().log()
                )
                log_qz_backg = (
                    torch.logsumexp(log_qz[set_s][:, set_ms], 1) - set_ms.sum().log()
                )
                pen += (log_qz_foreg - log_qz_backg).sum()
            pen = lambd * pen / unique_samples.shape[0]
        loss += pen

        kl_local = torch.tensor(0.0)
        kl_global = torch.tensor(0.0)
        return LossRecorder(
            loss,
            reconstruction_loss,
            kl_local,
            kl_global,
            pen=pen,
        )

    def _compute_local_library_params(self, batch_index):
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars
