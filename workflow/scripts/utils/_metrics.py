import torch
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from joblib import Parallel, delayed


def smooth_distance(sq_dists, mask_farther_than_k=False):
    n_donors = sq_dists.shape[-1]
    k = int(np.sqrt(n_donors))
    topk_vals, topk_idx = per_cell_knn(sq_dists, k=k)
    bandwidth_idx = k // 3
    bandwidth_vals = topk_vals[:, :, bandwidth_idx].unsqueeze(
        -1
    )  # n_cells x n_donors x 1
    w_mtx = torch.exp(-sq_dists / bandwidth_vals)  # n_cells x n_donors x n_donors

    if mask_farther_than_k:
        masked_w_mtx = torch.zeros_like(w_mtx)
        masked_w_mtx = masked_w_mtx.scatter(
            -1, topk_idx, w_mtx
        )  # n_cells x n_donors x n_donors
        w_mtx = masked_w_mtx

    return w_mtx


@torch.no_grad()
def per_cell_knn(dists, k):
    # Given n_cells x n_donors x n_donors returns n_cells x n_donors x k
    # tensor of dists and indices of the k nearest neighbors for each donor in each cell
    topkp1 = torch.topk(dists, k + 1, dim=-1, largest=False, sorted=True)
    topk_values, topk_indices = topkp1.values[:, :, 1:], topkp1.indices[:, :, 1:]
    return topk_values, topk_indices


@torch.no_grad()
def compute_geary(xs, donor_labels):
    oh_feats = OneHotEncoder(sparse=False).fit_transform(
        donor_labels[:, None]
    )  # n_donors x n_labels
    w_mat = torch.tensor(oh_feats @ oh_feats.T, device="cuda")  # n_donors x n_donors
    xs = xs.unsqueeze(-2)  # n_cells x n_donors x 1 x n_donor_latent
    sq_dists = ((xs - xs.transpose(-2, -3)) ** 2).sum(
        -1
    )  # n_cells x n_donors x n_donors
    scores_ = (sq_dists * w_mat).sum([-1, -2]) / w_mat.sum([-1, -2])
    var_estim = xs.var(1).sum(-1).squeeze()
    scores_ = scores_ / (2.0 * var_estim)
    return scores_.cpu().numpy()


@torch.no_grad()
def compute_hotspot_morans(xs, donor_labels):
    oh_feats = OneHotEncoder(sparse=False).fit_transform(donor_labels[:, None])
    like_label_mtx = torch.tensor(oh_feats @ oh_feats.T, device="cuda")
    xx_mtx = (like_label_mtx * 2) - 1  # n_donors x n_donors

    xs = xs.unsqueeze(-2)  # n_cells x n_donors x 1 x n_donor_latent
    sq_dists = ((xs - xs.transpose(-2, -3)) ** 2).sum(
        -1
    )  # n_cells x n_donors x n_donors
    w_mtx = smooth_distance(sq_dists)
    w_norm_mtx = w_mtx / w_mtx.sum(-1, keepdim=True)  # n_cells x n_donors x n_donors

    scores_ = (w_norm_mtx * xx_mtx).sum([-1, -2])  # n_cells
    return scores_.cpu().numpy()


@torch.no_grad()
def compute_cramers(xs, donor_labels):
    oh_feats = OneHotEncoder(sparse=False).fit_transform(
        donor_labels[:, None]
    )  # n_donors x n_labels
    oh_feats = torch.tensor(oh_feats, device="cuda").float()  # n_donors x n_labels

    xs = xs.unsqueeze(-2)  # n_cells x n_donors x 1 x n_donor_latent
    sq_dists = ((xs - xs.transpose(-2, -3)) ** 2).sum(
        -1
    )  # n_cells x n_donors x n_donors
    w_mtx = smooth_distance(sq_dists, mask_farther_than_k=True)

    c_ij = w_mtx @ oh_feats  # n_cells x n_donors x n_labels
    contingency_X = c_ij.transpose(-1, -2) @ oh_feats  # n_cells x n_labels x n_labels

    scores = []
    sig_scores = []
    for i in range(contingency_X.shape[0]):
        contingency_Xi = contingency_X[i].cpu().numpy()
        chi_sq, sig_score, _, _ = stats.chi2_contingency(contingency_Xi)
        n = np.sum(contingency_Xi)
        min_dim = contingency_Xi.shape[0] - 1
        scores.append(np.sqrt(chi_sq / (n * min_dim)))
        sig_scores.append(sig_score)
    return np.array(scores), np.array(sig_scores)


def _random_subsample(d1, n_mc_samples):
    n_minibatch = d1.shape[0]
    if d1.shape[0] >= n_mc_samples:
        d1_ = torch.zeros(n_minibatch, n_mc_samples)
        for i in range(n_minibatch):
            rdm_idx1 = torch.randperm(d1.shape[0])[:n_mc_samples]
            d1_[i] = d1[i][rdm_idx1]
    return d1


@torch.no_grad()
def compute_manova(
    xs,
    donor_labels,
):
    target = pd.Series(donor_labels, dtype="category").cat.codes
    target = sm.add_constant(target)

    def _compute(xi):
        try:
            res = (
                MANOVA(endog=xi, exog=target)
                .mv_test()
                .results["x1"]["stat"]
                .loc["Wilks' lambda", ["F Value", "Pr > F"]]
                .values
            )
        except Exception:
            return np.array([1000.0, 1.0])
        return res

    all_res = np.array(
        Parallel(n_jobs=10)(delayed(_compute)(xi) for xi in xs), dtype=np.float32
    )
    if isinstance(xs, np.ndarray):
        xs = torch.from_numpy(xs).to(torch.float32)
    all_res = np.array(
        Parallel(n_jobs=10)(delayed(_compute)(xi.cpu().numpy()) for xi in xs),
        dtype=np.float32,
    )
    stats, pvals = all_res[:, 0], all_res[:, 1]
    return stats, pvals


def compute_ks(
    xs, donor_labels, n_mc_samples=5000, alternative="two-sided", do_smoothing=False
):
    n_minibatch = xs.shape[0]
    oh_feats = OneHotEncoder(sparse=False).fit_transform(
        donor_labels[:, None]
    )  # n_donors x n_labels
    oh_feats = torch.tensor(oh_feats, device="cuda").float()  # n_donors x n_labels
    wmat = oh_feats @ oh_feats.T
    wmat = wmat - torch.eye(wmat.shape[0], device="cuda").float()
    is_off_diag = 1.0 - torch.eye(wmat.shape[0], device="cuda").float()

    xs = xs.unsqueeze(-2)
    sq_dists = ((xs - xs.transpose(-2, -3)) ** 2).sum(
        -1
    )  # n_cells x n_donors x n_donors
    if do_smoothing:
        sq_dists = smooth_distance(sq_dists)
    sq_dists = sq_dists.reshape(n_minibatch, -1)

    d1 = sq_dists[:, wmat.reshape(-1).bool()]
    d1 = _random_subsample(d1, n_mc_samples).cpu().numpy()
    d2 = sq_dists[:, is_off_diag.reshape(-1).bool()]
    d2 = _random_subsample(d2, n_mc_samples).cpu().numpy()

    ks_results = stats.ttest_ind(d1, d2, 1, equal_var=False)
    effect_sizes = ks_results.statistic
    p_values = ks_results.pvalue
    return effect_sizes, p_values
