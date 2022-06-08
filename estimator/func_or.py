from abc import ABC, abstractmethod
import torch
import numpy as np
from slycot import sb04qd
from sklearn.exceptions import ConvergenceWarning
from .proxs import (proj_matrix_2,proj_matrix_inf, bst_matrix, st)



class FOR(ABC):

    def __init__(self, model, lbda, alpha_scaling="discrete"):
        self.model = model
        self.lbda = lbda
        self.alpha_scaling = alpha_scaling
        self.losses = None

    @abstractmethod
    def primal_loss(self, alpha, rescale_alpha=True):
        pass

    @abstractmethod
    def dual_loss_diff(self, alpha):
        pass

    # @abstractmethod
    # def dual_loss_full(self, alpha):
    #     pass

    @abstractmethod
    def dual_grad(self, alpha):
        pass

    @abstractmethod
    def prox_step(self, alpha, gamma=None):
        pass

    def get_rescale_cste(self):
        if self.alpha_scaling == "discrete":
            return self.lbda * self.model.n * self.model.m
        else:
            return self.lbda * self.model.n * self.model.m

    def prox_lsearch(self, alpha, grad, beta=0.2):
        t = 1
        stop = False
        while not stop:
            direction = (1 / t) * (alpha - self.prox_step(alpha - t * grad, t))
            term1 = self.dual_loss_diff(alpha - t * direction)
            term2 = self.dual_loss_diff(alpha) - t * (grad * direction).sum() + 0.5 * t * (direction * direction).sum()
            if term1 > term2:
                t *= beta
            else:
                stop = True
        return t

    def model_init(self, x, y, thetas, warm_start, reinit_losses=True, G_x=None, G_t=None):
        self.model.initialise(x, y, thetas, warm_start=warm_start, G_x=G_x, G_t=G_t)
        # Losses initialization
        if self.losses is None:
            self.losses = []
        if reinit_losses:
            self.losses = []
        cste = self.get_rescale_cste()
        return self.model.alpha.detach().clone() * cste

    def monitor_loss(self, alpha, mode):
        # Monitor loss
        if mode == "dual":
            self.losses.append(self.dual_loss_full(alpha))
        elif mode == "dual_diff":
            self.losses.append(self.dual_loss_diff(alpha))
        elif mode == "primal":
            self.losses.append(self.primal_loss(alpha, rescale_alpha=True))
        else:
            raise ValueError("Unknown loss monitoring mode")

    def get_loss(self, alpha, mode):
        # Monitor loss
        if mode == "dual":
            return self.dual_loss_full(alpha)
        elif mode == "dual_diff":
            return self.dual_loss_diff(alpha)
        elif mode == "primal":
            return self.primal_loss(alpha, rescale_alpha=True)
        else:
            raise ValueError("Unknown loss monitoring mode")

    def fit_prox_gd(self, x, y, thetas, n_epoch=2000, warm_start=True, tol=1e-8, beta=0.5,
                    monitor_loss="dual_diff", reinit_losses=True, G_x=None, G_t=None):
        # Initializations + Scaling of alpha to match computations
        alpha = self.model_init(x, y, thetas, warm_start, reinit_losses=reinit_losses, G_x=G_x, G_t=G_t)
        # Grad initialization
        grad = self.dual_grad(alpha)
        for epoch in range(n_epoch):
            step_size = self.prox_lsearch(alpha, grad, beta)
            alpha -= step_size * self.dual_grad(alpha)
            alpha = self.prox_step(alpha, step_size)
            # Monitor loss
            self.monitor_loss(alpha, monitor_loss)
            # Stopping criterion
            if epoch > 1:
                diff = (self.losses[-2].item() - self.losses[-1].item()) / self.losses[-2].abs().item()
                print(diff)
                if diff < tol:
                    break
            grad = self.dual_grad(alpha)
        # Scale alpha to match prediction formula
        cste = self.get_rescale_cste()
        self.model.alpha = alpha / cste

    def acc_prox_lsearch(self, t0, alpha_v, grad_v, beta=0.2):
        t = t0
        stop = False
        while not stop:
            alpha_plus = self.prox_step(alpha_v - t * grad_v, t)
            term1 = self.dual_loss_diff(alpha_plus)
            term21 = self.dual_loss_diff(alpha_v)
            term22 = (grad_v * (alpha_plus - alpha_v)).sum()
            term23 = 0.5 * (1 / t) * ((alpha_plus - alpha_v) ** 2).sum()
            term2 = term21 + term22 + term23
            if term1 > term2:
                t *= beta
            else:
                stop = True
        return t

    def fit_acc_prox_gd(self, x, y, thetas, n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
                        monitor_loss=None, reinit_losses=True, d=20, G_x=None, G_t=None, verbose=True):
        alpha = self.model_init(x, y, thetas, warm_start, reinit_losses=reinit_losses, G_x=G_x, G_t=G_t)
        alpha_minus1 = alpha
        alpha_minus2 = alpha
        step_size = 1
        converged = False
        for epoch in range(0, n_epoch):
            if verbose:
                print("Iteration: " + str(epoch))
            acc_cste = epoch / (epoch + 1 + d)
            alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
            grad_v = self.dual_grad(alpha_v)
            step_size = self.acc_prox_lsearch(step_size, alpha_v, grad_v, beta)
            alpha = self.prox_step(alpha_v - step_size * grad_v, step_size)
            if monitor_loss:
                self.monitor_loss(alpha, monitor_loss)
            diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
            if verbose:
                print("Normalized distance between iterates: " + str(diff))
            if diff < tol:
                converged = True
                break
            alpha_minus2 = alpha_minus1.detach().clone()
            alpha_minus1 = alpha.detach().clone()
        if not converged:
            raise ConvergenceWarning("Maximum number of iteration reached")
        # Scale alpha to match prediction formula
        cste = self.get_rescale_cste()
        self.model.alpha = alpha / cste

    def fit_acc_prox_restart_gd(self, x, y, thetas, n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
                                monitor_loss=None, reinit_losses=True, d=20, G_x=None, G_t=None, verbose=True):
        alpha = self.model_init(x, y, thetas, warm_start, reinit_losses=reinit_losses, G_x=G_x, G_t=G_t)
        alpha_minus1 = alpha
        alpha_minus2 = alpha
        step_size = 1
        epoch_restart = 0
        converged = False
        for epoch in range(0, n_epoch):
            if verbose:
                print("Iteration: " + str(epoch))
            acc_cste = epoch_restart / (epoch_restart + 1 + d)
            alpha_v = alpha_minus1 + acc_cste * (alpha_minus1 - alpha_minus2)
            grad_v = self.dual_grad(alpha_v)
            step_size = self.acc_prox_lsearch(step_size, alpha_v, grad_v, beta)
            alpha_tentative = self.prox_step(alpha_v - step_size * grad_v, step_size)
            if ((alpha_v - alpha_tentative) * (alpha_tentative - alpha_minus1)).sum() > 0:
                if verbose:
                    print("\n RESTART \n")
                epoch_restart = 0
                grad_v = self.dual_grad(alpha_minus1)
                step_size = self.acc_prox_lsearch(step_size, alpha_minus1, grad_v, beta)
                alpha = self.prox_step(alpha_minus1 - step_size * grad_v, step_size)
            else:
                alpha = alpha_tentative
            if monitor_loss:
                self.monitor_loss(alpha, monitor_loss)
            diff = (alpha - alpha_minus1).norm() / alpha_minus1.norm()
            if verbose:
                print("Normalized distance between iterates: " + str(diff))
            if diff < tol:
                converged = True
                break
            alpha_minus2 = alpha_minus1.detach().clone()
            alpha_minus1 = alpha.detach().clone()
            epoch_restart += 1
        # Scale alpha to match prediction formula
        cste = self.get_rescale_cste()
        self.model.alpha = alpha / cste
        if not converged:
            raise ConvergenceWarning("Maximum number of iteration reached")
    
    @abstractmethod
    def fit_sylvester(x, y, thetas, G_x=None, G_t=None):
        pass

    def fit(self, x, y, thetas, solver="acc_proxgd_restart", n_epoch=20000, warm_start=True, tol=1e-6, beta=0.8,
            monitor_loss=None, reinit_losses=True, d=20, G_x=None, G_t=None, verbose=True, sylvester_init=False):
        if solver == "acc_proxgd_restart":
            if sylvester_init: 
                self.fit_sylvester(x, y, thetas, G_x=G_x, G_t=G_t)
                warm_start = True
            self.fit_acc_prox_restart_gd(x, y, thetas, n_epoch, warm_start, tol, beta, monitor_loss, reinit_losses, d, G_x, G_t, verbose)
        elif solver == "acc_proxgd":
            if sylvester_init: 
                self.fit_sylvester(x, y, thetas, G_x=G_x, G_t=G_t)
                warm_start = True
            self.fit_acc_prox_gd(x, y, thetas, n_epoch, warm_start, tol, beta, monitor_loss, reinit_losses, d, G_x, G_t, verbose)
        elif solver == "Sylvester":
            if isinstance(self, (RobustFOREig, RobustFORSpl, SparseFOREig, SparseFORSpl)):
                raise ValueError("Sylvester solver can only be used for the square loss")
            self.fit_sylvester(x, y, thetas, G_x=G_x, G_t=G_t)
        else:
            raise ValueError("Unknown solver")


class FORSpl(FOR):

    def __init__(self, model, lbda):
        super().__init__(model, lbda, alpha_scaling="discrete")

    def primal_loss(self, alpha, rescale_alpha=True):
        if rescale_alpha:
            cste = 1 / (self.lbda * self.model.n * self.model.m)
            alpha_sc = alpha * cste
        else:
            alpha_sc = alpha
        pred = self.model.G_x @ alpha_sc @ self.model.G_t
        return torch.mean((pred - self.model.y_train) ** 2)

    def dual_loss_diff(self, alpha):
        A = 0.5 * alpha @ alpha.T
        B = - alpha @ self.model.y_train.T
        cste = 0.5 / (self.lbda * self.model.n * self.model.m)
        C = cste * self.model.G_x @ alpha @ self.model.G_t @ alpha.T
        return torch.trace(A + B + C)

    # def dual_loss_full(self, alpha):
    #     return self.dual_loss_diff(alpha)

    def dual_grad(self, alpha):
        A = alpha
        B = - self.model.y_train
        cste = 1 / (self.lbda * self.model.n * self.model.m)
        C = cste * self.model.G_x @ alpha @ self.model.G_t
        return A + B + C

    def prox_step(self, alpha, gamma=None):
        return alpha

    def fit_sylvester(self, x, y, thetas, G_x=None, G_t=None):
        alpha = self.model_init(x, y, thetas, warm_start=False, reinit_losses=False, G_x=G_x, G_t=G_t)
        alpha = sb04qd(self.model.n, self.model.m, 
                       self.model.G_x.numpy() / (self.lbda * self.model.n * self.model.m), 
                       self.model.G_t.numpy(), y.numpy() / (self.lbda * self.model.n * self.model.m))
        self.model.alpha = torch.from_numpy(alpha)


class FOREig(FOR):

    def __init__(self, model, lbda):
        super().__init__(model, lbda, alpha_scaling="eigen")

    def primal_loss(self, alpha, rescale_alpha=True):
        cste = self.get_rescale_cste()
        if rescale_alpha:
            alpha_sc = alpha * (1 / cste)
        else:
            alpha_sc = alpha
        pred = self.model.G_x @ alpha_sc @ torch.diag(self.model.eig_vals) @ self.model.eig_vecs.T
        return torch.mean((pred - self.model.y_train) ** 2)

    def dual_loss_diff(self, alpha):
        A = 0.5 * alpha @ alpha.T
        B = - alpha @ self.model.R.T
        cste = 0.5 / (self.lbda * self.model.n * self.model.m)
        C = cste * self.model.G_x @ alpha @ torch.diag(self.model.eig_vals) @ alpha.T
        return torch.trace(A + B + C)

    # def dual_loss_full(self, alpha):
    #     return self.dual_loss_diff(alpha)

    def dual_grad(self, alpha):
        A = alpha
        B = - self.model.R
        cste = 1 / (self.lbda * self.model.n * self.model.m)
        C = cste * self.model.G_x @ alpha @ torch.diag(self.model.eig_vals)
        return A + B + C

    def prox_step(self, alpha, gamma=None):
        return alpha
    
    def fit_sylvester(self, x, y, thetas, G_x=None, G_t=None):
        alpha = self.model_init(x, y, thetas, warm_start=False, reinit_losses=False, G_x=G_x, G_t=G_t)
        Lambda = torch.diag(self.model.eig_vals)
        alpha = sb04qd(self.model.n, self.model.n_eig, 
                       self.model.G_x.numpy() / (self.lbda * self.model.n * self.model.m), Lambda.numpy(),
                       self.model.R.numpy() / (self.lbda * self.model.n * self.model.m))
        self.model.alpha = torch.from_numpy(alpha)


class RobustFOREig(FOREig):

    def __init__(self, model, lbda, loss_param=0.1):
        super().__init__(model, lbda)
        self.loss_param = loss_param

    def prox_step(self, alpha, gamma=None):
        return proj_matrix_2(alpha, self.loss_param)

    def get_kappa_max(self, alpha, rescale_alpha=False):
        n, m = self.model.alpha.shape[0], self.model.alpha.shape[1]
        if rescale_alpha:
            cste = 1 / (self.lbda * self.model.n * self.model.m)
            alpha_sc = alpha * cste
        else:
            alpha_sc = alpha
        pred = self.model.G_x @ alpha_sc @ torch.diag(self.model.eig_vals) @ self.model.eig_vecs
        return (1 / np.sqrt(m)) * torch.sqrt(((self.model.y_train - pred) ** 2).sum(dim=1)).max()


class SparseFOREig(FOREig):

    def __init__(self, model, lbda, loss_param=0.1):
        super().__init__(model, lbda)
        self.loss_param = loss_param

    def prox_step(self, alpha, gamma=None):
        return bst_matrix(alpha, gamma * np.sqrt(self.model.m) * self.loss_param)

    def get_sparsity_level(self):
        n_zeros = len(torch.where(self.model.alpha == 0)[0])
        return n_zeros / (self.model.n_eig * self.model.n)


class RobustFORSpl(FORSpl):

    def __init__(self, model, lbda, loss_param=0.1, norm='inf'):
        super().__init__(model, lbda)
        self.loss_param = loss_param
        self.norm = norm

    # Indicator function is problematic to implement it
    # def dual_loss_full(self, alpha):
    #     return self.dual_loss_diff(alpha)

    def primal_loss(self, alpha, rescale_alpha=True):
        if rescale_alpha:
            cste = 1 / (self.lbda * self.model.n * self.model.m)
            alpha_sc = alpha * cste
        else:
            alpha_sc = alpha
        pred = self.model.G_x @ alpha_sc @ self.model.G_t
        if self.norm == "2":
            data_fitting = None
        elif self.norm == "inf":
            error = self.model.y_train - pred
            error_norms = (1 / np.sqrt(self.model.m)) * torch.sqrt((error ** 2).sum(dim=1))
            mask_sup_kappa = error_norms > self.loss_param
            mask_inf_kappa = error_norms <= self.loss_param
            term_sup_kappa = (self.loss_param * (mask_sup_kappa * (error_norms - 0.5 * self.loss_param))).sum()
            term_inf_kappa = (0.5 * mask_inf_kappa * error_norms ** 2).sum()
            data_fitting = (1 / self.model.n) * (term_inf_kappa + term_sup_kappa)
        else:
            raise ValueError('Not implemented norm')
        regularization = (0.5 * self.lbda / self.model.m ** 2) \
            * torch.trace(self.model.G_x @ alpha_sc @ self.model.G_t @ alpha_sc.T)
        return data_fitting + regularization

    def prox_step(self, alpha, gamma=None):
        if self.norm == '2':
            return proj_matrix_2(alpha, self.loss_param)
        elif self.norm == 'inf':
            return proj_matrix_inf(alpha, self.loss_param)
        else:
            raise ValueError('Not implemented norm')

    def get_kappa_max(self, alpha, rescale_alpha=True):
        n, m = self.model.alpha.shape[0], self.model.alpha.shape[1]
        if rescale_alpha:
            cste = 1 / (self.lbda * self.model.n * self.model.m)
            alpha_sc = alpha * cste
        else:
            alpha_sc = alpha
        pred = self.model.G_x @ alpha_sc @ self.model.G_t
        if self.norm == '2':
            return (1 / np.sqrt(m)) * torch.sqrt(((self.model.y_train - pred) ** 2).sum(dim=1)).max()
        elif self.norm == 'inf':
            return (self.model.y_train - pred).abs().max()
        else:
            raise ValueError('Not implemented norm')


class SparseFORSpl(FORSpl):

    def __init__(self, model, lbda, loss_param=0.1, norm='inf'):
        super().__init__(model, lbda)
        self.loss_param = loss_param
        self.norm = norm

    # def dual_loss_full(self, alpha):
    #     if self.norm == "2":
    #         dual_penalty = np.sqrt(self.model.m) * self.model.alpha.norm(dim=1).sum()
    #     elif self.norm == "inf":
    #         dual_penalty = alpha.abs().sum()
    #     else:
    #         raise ValueError('Not implemented norm')
    #     return self.dual_loss_diff(alpha) + self.loss_param * dual_penalty

    def primal_loss(self, alpha, rescale_alpha=True):
        if rescale_alpha:
            cste = 1 / (self.lbda * self.model.n * self.model.m)
            alpha_sc = alpha * cste
        else:
            alpha_sc = alpha
        error = self.model.y_train - self.model.G_x @ alpha_sc @ self.model.G_t
        if self.norm == "2":
            error_norms = (1 / np.sqrt(self.model.m)) * torch.sqrt((error ** 2).sum(dim=1))
            data_fitting = error_norms.mean()
        elif self.norm == "inf":
            abs_error = error.abs()
            data_fitting = (torch.maximum(abs_error, torch.tensor(self.loss_param)) ** 2).mean()
        else:
            raise ValueError('Not implemented norm')
        regularization = (0.5 * self.lbda / self.model.m ** 2) \
            * torch.trace(self.model.G_x @ alpha_sc @ self.model.G_t @ alpha_sc.T)
        return data_fitting + regularization

    def prox_step(self, alpha, gamma=None):
        if self.norm == '2':
            return bst_matrix(alpha, gamma * np.sqrt(self.model.m) * self.loss_param)
        elif self.norm == 'inf':
            return st(alpha, gamma * self.loss_param)

    def get_epsilon_min(self, alpha, rescale_alpha=True):
        n, m = self.model.alpha.shape[0], self.model.alpha.shape[1]
        if rescale_alpha:
            cste = 1 / (self.lbda * self.model.n * self.model.m)
            alpha_sc = alpha * cste
        else:
            alpha_sc = alpha
        pred = self.model.G_x @ alpha_sc @ self.model.G_t
        if self.norm == '2':
            return (1 / np.sqrt(m)) * torch.sqrt(((self.model.y_train - pred) ** 2).sum(dim=1)).min()
        elif self.norm == 'inf':
            return (self.model.y_train - pred).abs().min()
        else:
            raise ValueError('Not implemented norm')

    def get_sparsity_level(self):
        n_zeros = len(torch.where(self.model.alpha == 0)[0])
        return n_zeros / (self.model.m * self.model.n)
