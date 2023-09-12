import logging

import torch
import torch.distributions as db
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResnetFC(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=128,
        activation=nn.Softmax(-1),
    ):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        )
        self.module2 = nn.Sequential(
            nn.Linear(n_hidden, n_out),
            nn.BatchNorm1d(n_out),
        )
        if n_in != n_hidden:
            self.id_map1 = nn.Linear(n_in, n_hidden)
        else:
            self.id_map1 = None
        self.activation = activation

    def forward(self, inputs):
        need_reshaping = False
        if inputs.ndim == 3:
            n_d1, nd2 = inputs.shape[:2]
            inputs = inputs.reshape(n_d1 * nd2, -1)
            need_reshaping = True
        h = self.module1(inputs)
        if self.id_map1 is not None:
            h = h + self.id_map1(inputs)
        h = self.module2(h)
        if need_reshaping:
            h = h.view(n_d1, nd2, -1)
        if self.activation is not None:
            return self.activation(h)
        return h


class _NormalNN(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=128,
        n_layers=1,
        use_batch_norm=True,
        use_layer_norm=False,
        do_orthogonal=False,
    ):
        super().__init__()
        self.n_layers = n_layers

        self.hidden = ResnetFC(n_in, n_out=n_hidden, activation=nn.ReLU())
        self._mean = nn.Linear(n_hidden, n_out)
        self._var = nn.Sequential(nn.Linear(n_hidden, n_out), nn.Softplus())

    def forward(self, inputs):
        if self.n_layers >= 1:
            h = self.hidden(inputs)
            mean = self._mean(h)
            var = self._var(h)
        else:
            mean = self._mean(inputs)
            k = mean.shape[0]
            var = self._var[None].expand(k, -1)
        return mean, var


class NormalNN(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_categories,
        n_hidden=128,
        n_layers=1,
        use_batch_norm=True,
        use_layer_norm=False,
    ):
        super().__init__()
        nn_kwargs = dict(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.n_out = n_out
        self._mymodules = nn.ModuleList(
            [_NormalNN(**nn_kwargs) for _ in range(n_categories)]
        )

    def forward(self, inputs, categories=None):
        means = []
        vars = []
        for idx, module in enumerate(self._mymodules):
            _means, _vars = module(inputs)
            means.append(_means[..., None])
            vars.append(_vars[..., None])
        means = torch.cat(means, -1)
        vars = torch.cat(vars, -1)
        if categories is not None:
            # categories  (minibatch, 1)
            n_batch = categories.shape[0]
            cat_ = categories.unsqueeze(-1).long().expand(n_batch, self.n_out, 1)
            if means.ndim == 4:
                d1, n_batch, _, _ = means.shape
                cat_ = (
                    categories[None, :, None].long().expand(d1, n_batch, self.n_out, 1)
                )
            means = torch.gather(means, -1, cat_)
            vars = torch.gather(vars, -1, cat_)
        means = means.squeeze(-1)
        vars = vars.squeeze(-1)
        return db.Normal(means, vars + 1e-5)


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        need_reshaping = False
        if x.ndim == 3:
            n_d1, nd2 = x.shape[:2]
            x = x.view(n_d1 * nd2, -1)
            need_reshaping = True

            y = y[None].expand(n_d1, nd2, -1)
            y = y.contiguous().view(n_d1 * nd2, -1)

        out = self.bn(x)
        gamma, beta = self.embed(y.squeeze(-1).long()).chunk(2, 1)
        out = gamma * out + beta

        if need_reshaping:
            out = out.view(n_d1, nd2, -1)

        return out
