from abc import ABC, abstractmethod

from scipy.interpolate import interp1d
import torch

torch.set_default_dtype(torch.float64)


class Decomposable(ABC):

    def __init__(self, kernel_input, kernel_output, subsample=1):
        self.kernel_input = kernel_input
        self.kernel_output = kernel_output
        self.subsample = subsample
        self.thetas = None
        self.alpha = None
        self.x_train = None
        self.y_train = None
        self.G_x = None
        self.G_t = None
        self.n = None
        self.m = None

    def compute_gram_train(self, G_x=None, G_t=None):
        if not hasattr(self, 'x_train'):
            raise Exception('No training data provided')
        if G_x is None:
            self.G_x = self.kernel_input.compute_gram(self.x_train)
        else:
            self.G_x = G_x
        if G_t is None:
            self.G_t = self.kernel_output.compute_gram(self.thetas[::self.subsample])
        else:
            self.G_t = G_t
    
    @abstractmethod
    def initialise_specifics(warm_start):
        pass

    def initialise(self, x, y, thetas, warm_start, G_x=None, G_t=None):
        self.x_train = x
        self.n = x.shape[0]
        self.y_train = y[:, ::self.subsample]
        self.thetas = thetas
        self.compute_gram_train(G_x=G_x, G_t=G_t)
        self.m = thetas[::self.subsample].shape[0]
        self.initialise_specifics(warm_start)

    @abstractmethod
    def forward(self, x, G_x=None):
        pass


class DecomposableIdentityScalar(Decomposable):

    def __init__(self, kernel_input, kernel_output, subsample):
        super().__init__(kernel_input, kernel_output, subsample)

    def forward(self, x, G_x=None):
        if not hasattr(self, 'x_train'):
            raise Exception('No training anchors provided to the model')
        if G_x is None:
            G_x = self.kernel_input.compute_gram(x, self.x_train)
        pred = G_x @ self.alpha @ self.G_t
        if self.subsample != 1:
            interp_func = interp1d(self.thetas[::self.subsample, 0].numpy(), pred.numpy(), fill_value="extrapolate")
            return torch.from_numpy(interp_func(self.thetas[:, 0].numpy()))
        else:
            return pred

    def initialise_specifics(self, warm_start):
        if self.alpha is None or not warm_start:
            self.alpha = torch.randn((self.n, self.m), requires_grad=False)
        if warm_start and len(self.alpha) != self.n:
            self.alpha = torch.randn((self.n, self.m), requires_grad=False)


class DecomposableIntOp(Decomposable):

    def __init__(self, kernel_input, kernel_output, n_eig, subsample):
        super().__init__(kernel_input, kernel_output, subsample)
        self.n_eig = n_eig
        self.R = None
        self.eig_vals = None
        self.eig_vecs = None

    def compute_eigen_output(self):
        eig_vecs, eig_vals, _ = torch.svd(self.G_t)
        self.eig_vecs = eig_vecs[:, :self.n_eig].T
        self.eig_vals = eig_vals[:self.n_eig]

    def compute_R(self, y):
        self.R = y @ self.eig_vecs.T

    def initialise_specifics(self, warm_start):
        self.compute_eigen_output()
        self.compute_R(self.y_train)
        if self.alpha is None or not warm_start:
            self.alpha = torch.randn((self.n, self.n_eig), requires_grad=False)
        if warm_start and len(self.alpha) != self.n:
            self.alpha = torch.randn((self.n, self.n_eig), requires_grad=False)

    def forward(self, x, G_x=None):
        if not hasattr(self, 'x_train'):
            raise Exception('No training anchors provided to the model')
        if G_x is None:
            G_x = self.kernel_input.compute_gram(x, self.x_train)
        pred = G_x @ self.alpha @ torch.diag(self.eig_vals) @ self.eig_vecs
        if self.subsample != 1:
            interp_func = interp1d(self.thetas[::self.subsample, 0].numpy(), pred.numpy(), fill_value="extrapolate")
            return torch.from_numpy(interp_func(self.thetas[:, 0].numpy()))
        else:
            return pred