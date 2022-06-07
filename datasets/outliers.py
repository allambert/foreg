import torch
from sklearn.gaussian_process.kernels import RBF
import numpy as np

from . import synthetic_func_or as synth


def add_local_outliers(X, Xeval=None, freq_sample=1., freq_loc=0.1, intensity=0.5, seed=453):
    n, m = X.shape
    res = X.detach().clone()
    if Xeval is not None:
        res_eval = Xeval.copy()
    a_max = (X.abs().max()) * intensity
    n_contaminated = int(freq_sample * n)
    np.random.seed(seed)
    contaminated_inds = np.random.choice(np.arange(n), n_contaminated, replace=False)
    for i in contaminated_inds:
        m_contaminated = int(freq_loc * m)
        contaminated_locs = np.random.choice(np.arange(m), m_contaminated, replace=False)
        noise = np.random.uniform(- a_max.item(), a_max.item(), m_contaminated)
        res[i, contaminated_locs] = torch.from_numpy(noise)
        if Xeval is not None:
            res_eval[i, contaminated_locs] = noise
    if Xeval is not None:
        return res, res_eval
    else:
        return res, None


def add_label_noise(X, Xeval=None, freq_sample=0.02, seed=443, coef=-1., alternate_coef=False):
    n, m = X.shape
    res = X.detach().clone()
    if Xeval is not None:
        res_eval = Xeval.copy()
    n_contaminated = int(freq_sample * n)
    np.random.seed(seed)
    contaminated_inds = torch.from_numpy(np.random.choice(np.arange(n), n_contaminated, replace=False))
    ind_shift = contaminated_inds.flip(0)
    # ind_shift = contaminated_inds.shift(1)
    if alternate_coef :
        coef = torch.tensor([(-1) ** i * coef for i in range(n_contaminated)]).unsqueeze(1)
    res[contaminated_inds] = coef * X[ind_shift]
    if Xeval is not None:
        res_eval[contaminated_inds] = Xeval[ind_shift.numpy()]
        return res, res_eval
    else:
        return res, None


def add_gp_outliers(X, Xeval=None, freq_sample=0.02, seed=443, seed_gps=56, covs_params=(0.01, 0.05, 1, 4), scale=2, intensity=2.5, additive=True):
    n, m = X.shape
    res = X.detach().clone()
    if Xeval is not None:
        res_eval = Xeval.copy()
    theta = torch.linspace(0, 1, m)
    gp_outliers =  synth.SyntheticGPmixture(len(covs_params), (covs_params, covs_params), noise=(None, None), scale=scale)
    gp_outliers.drawGP(theta, seed_gp=seed_gps)
    n_contaminated = int(freq_sample * n)
    _, drawns_gps = gp_outliers.sample(n_contaminated, new_GP=False, seed_gp=seed_gps, seed_coefs=seed)
    np.random.seed(seed)
    contaminated_inds = torch.from_numpy(np.random.choice(np.arange(n), n_contaminated, replace=False))
    if additive:
        res[contaminated_inds] += intensity * drawns_gps
        # return intensity * drawns_gps
        if Xeval is not None:
            res_eval[contaminated_inds] += intensity * drawns_gps.numpy()
            return res, res_eval
        else:
            return res, None
    else:
        res[contaminated_inds] = intensity * drawns_gps
        # return intensity * drawns_gps
        if Xeval is not None:
            res_eval[contaminated_inds] = intensity * drawns_gps.numpy()
            return res, res_eval
        else:
            return res, None


def add_local_gnoise(X, Xeval=None, freq_sample=1., freq_loc=0.2, intensity=0.1, seed=4453):
    n, m = X.shape
    res = X.detach().clone()
    np.random.seed(seed)
    n_contaminated = int(freq_sample * n)
    contaminated_inds = torch.from_numpy(np.random.choice(np.arange(n), n_contaminated, replace=False))
    for i in contaminated_inds:
        m_contaminated = int(freq_loc * m)
        contaminated_locs = np.random.choice(np.arange(m), m_contaminated, replace=False)
        noise = np.random.normal(0, intensity, m_contaminated)
        res[i, contaminated_locs] += torch.from_numpy(noise)
    return res, None