import torch
import torch.nn as nn
from scvi.distributions import NegativeBinomial
from scvi.nn import one_hot

from ._utils import ResnetFC


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
        n_sample,
        n_out,
        scaler=False,
        scaler_n_hidden=32,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_sample = n_sample
        self.n_out = n_out

        self.amat_sample = nn.Parameter(torch.randn(n_sample, self.n_latent, n_out))
        self.offsets = nn.Parameter(torch.randn(n_sample, n_out))

        self.scaler = None
        if scaler:
            self.scaler = nn.Sequential(
                nn.Linear(n_latent + n_sample, scaler_n_hidden),
                nn.LayerNorm(scaler_n_hidden),
                nn.ReLU(),
                nn.Linear(scaler_n_hidden, 1),
                nn.Sigmoid(),
            )

    def forward(self, u, sample_id):
        sample_id_ = sample_id.long().squeeze()
        As = self.amat_sample[sample_id_]

        u_detach = u.detach()[..., None]
        z2 = (As * u_detach).sum(-2)
        offsets = self.offsets[sample_id_]
        delta = z2 + offsets
        if self.scaler is not None:
            sample_oh = one_hot(sample_id, self.n_sample)
            if u.ndim != sample_oh.ndim:
                sample_oh = sample_oh[None].expand(u.shape[0], *sample_oh.shape)
            inputs = torch.cat([u.detach(), sample_oh], -1)
            delta = delta * self.scaler(inputs)
        return u + delta


class DecoderUZ(nn.Module):
    def __init__(
        self,
        n_latent,
        n_latent_sample,
        n_out,
        dropout_rate=0.0,
        n_layers=1,
        n_hidden=128,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_latent_sample = n_latent_sample
        self.n_in = n_latent + n_latent_sample
        self.n_out = n_out

        arch_mod = self.construct_arch(self.n_in, n_hidden, n_layers, dropout_rate) + [
            nn.Linear(n_hidden, self.n_out, bias=False)
        ]
        self.mod = nn.Sequential(*arch_mod)

        arch_scaler = self.construct_arch(
            self.n_latent, n_hidden, n_layers, dropout_rate
        ) + [nn.Linear(n_hidden, 1)]
        self.scaler = nn.Sequential(*arch_scaler)
        self.scaler.append(nn.Sigmoid())

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
        mean = u[..., : self.n_latent] + scaler_ * pred_
        return mean
