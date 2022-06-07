import torch
from .utils import rbf_kernel
from sklearn.metrics.pairwise import manhattan_distances

torch.set_default_dtype(torch.float64)


class Gaussian(object):

    def __init__(self, gamma):
        self.gamma = gamma
        self.is_learnable = False

    def compute_gram(self, X, Y=None):
        return rbf_kernel(X, Y, self.gamma)


class Laplacian(object):

    def __init__(self, gamma):
        self.gamma = gamma

    def compute_gram(self, X, Y=None):
        if Y is None:
            Y = X
        # pairwise = torch.from_numpy((1 / X.shape[1]) * manhattan_distances(X.numpy(), Y.numpy()))
        pairwise = torch.from_numpy(manhattan_distances(X.numpy(), Y.numpy()))
        return torch.exp(- self.gamma * pairwise)