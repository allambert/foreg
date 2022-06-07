import torch


def proj_matrix_2(alpha, kappa):
    m = torch.tensor(float(alpha.shape[1]))
    norm = (1 / torch.sqrt(m)) * torch.sqrt(torch.sum(alpha ** 2, axis=1))
    mask = torch.where(norm > kappa)
    alpha[mask] *= kappa / norm[mask].reshape((-1, 1))
    return alpha


def proj_matrix_inf(alpha, kappa):
    norm = torch.abs(alpha)
    return torch.where(norm > kappa, kappa * alpha/norm, alpha)


def proj_vect_2(alpha, kappa):
    norm = torch.sqrt(torch.sum(alpha**2))
    if norm > kappa:
        alpha *= kappa / norm
    return(alpha)


def proj_vect_inf(alpha, kappa):
    norm = torch.abs(alpha)
    mask = torch.where(norm > kappa)
    alpha[mask] *= kappa / norm[mask].reshape((-1, 1))
    return alpha


def bst_matrix(alpha, tau):
    norm = (alpha**2).sum(1).sqrt()
    mask_st = torch.where(norm >= tau)
    mask_ze = torch.where(norm < tau)
    alpha[mask_st] = alpha[mask_st] - alpha[mask_st] / \
        norm[mask_st].reshape((-1, 1)) * tau
    alpha[mask_ze] = 0
    return(alpha)


def bst_vector(alpha, tau):
    norm = (alpha**2).sum().sqrt()
    if norm > tau:
        alpha -= alpha/norm * tau
    else:
        alpha = 0
    return(alpha)


def st(alpha, tau):
    return torch.where(alpha.abs() - tau < 0,
                       torch.zeros_like(alpha),
                       alpha.abs() - tau) * torch.sign(alpha)
