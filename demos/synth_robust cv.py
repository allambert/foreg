import os
import sys
import torch
import pathlib
import itertools

torch.set_default_dtype(torch.float64)

# Set right path for local imports and dataset finding
try:
    exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    code_path = exec_path.parent.parent
except NameError:
    exec_path = pathlib.Path(os.getcwd())
    code_path = exec_path
sys.path.append(str(code_path))


# Local imports
from datasets import add_gp_outliers, load_gp_dataset
from estimator import RobustFORSpl, FOREig, RobustFOREig
from kernel import Gaussian
from model import DecomposableIdentityScalar, DecomposableIntOp
from model_selection import model_selection as msel

seed1 = 344
seed2 = 452

if __name__ == "__main__":

    # Load data
    Xtrain, Ytrain, Xtest, Ytest = load_gp_dataset(seed1, seed2)
    theta = torch.linspace(0, 1, 100)
    # Corrupt data set with GP (type 2) outliers
    Ytrain_corr, _ = add_gp_outliers(Ytrain, Xeval=None, freq_sample=0.1, seed=443, seed_gps=56, covs_params=(0.01, 0.05, 1, 4), scale=2, intensity=3)

    # Define kernels
    kin = Gaussian(0.01)
    kout = Gaussian(100)
    # Precompute Gram matrices
    G_x = kin.compute_gram(Xtrain, Xtrain)
    G_t = kout.compute_gram(theta.view(-1, 1), theta.view(-1, 1))

    # Parameters grids
    # Regularization
    lbda_grid = torch.logspace(-6, -3, 10)
    # Huber loss' kappa
    loss_params = torch.logspace(-3, -1, 10)


    # ###################### SQUARE LOSS ###############################################################################
    # Number of eigenfunctions for eigen solver
    n_eig = 25

    # Huber 2 loss estimators to consider in the tuning
    estis_sqr = [FOREig(DecomposableIntOp(kin, kout, n_eig, subsample=1),
                  lbda) for lbda in lbda_grid]
    
    # Cross validate estimators with Huber 2 loss and return best one
    best_esti_sqr, _ = msel.tune_estis(estis_sqr, Xtrain, Ytrain_corr, theta.view(-1, 1), G_x=G_x, G_t=G_t,
                                       n_splits=5, reduce_stat="median", random_state=342, 
                                       solver="Sylvester", n_jobs=-1, fit_best=True)
    
    # Use the selected estimator to predict on test set and display corresponding score
    pred_sqr = best_esti_sqr.model.forward(Xtest)
    score_sqr = torch.mean((pred_sqr - Ytest) ** 2)
    print("Score Square loss: " + str(score_sqr))


    # ###################### HUBER 2 ###############################################################################
    # Number of eigenfunctions for eigen solver
    n_eig = 25

    # Huber 2 loss estimators to consider in the tuning
    estis_hub2 = [RobustFOREig(DecomposableIntOp(kin, kout, n_eig, subsample=1),
                  lbda, loss_param=kappa) for lbda, kappa in itertools.product(lbda_grid, loss_params)]
    
    # Cross validate estimators with Huber 2 loss and return best one
    best_esti_hub2, _ = msel.tune_estis(estis_hub2, Xtrain, Ytrain_corr, theta.view(-1, 1), G_x=G_x, G_t=G_t,
                                        n_splits=5, reduce_stat="median", random_state=342, 
                                        solver="acc_proxgd_restart", n_epoch=20000, warm_start=True,
                                        tol=1e-8, beta=0.8, d=20, sylvester_init=True, 
                                        n_jobs=-1, fit_best=True)
    
    # Use the selected estimator to predict on test set and display corresponding score
    pred_hub2 = best_esti_hub2.model.forward(Xtest)
    score_hub2 = torch.mean((pred_hub2 - Ytest) ** 2)
    print("Score Huber 2 loss: " + str(score_hub2))


    # ###################### HUBER 1 ###############################################################################
    # Huber 1 loss estimators to consider in the tuning
    estis_hub1 = [RobustFORSpl(DecomposableIdentityScalar(kin, kout, subsample=1),
                  lbda, loss_param=kappa) for lbda, kappa in itertools.product(lbda_grid, loss_params)]
    
    # Cross validate estimators with Huber 2 loss and return best one
    best_esti_hub1, _ = msel.tune_estis(estis_hub1, Xtrain, Ytrain_corr, theta.view(-1, 1), G_x=G_x, G_t=G_t,
                                           n_splits=5, reduce_stat="median", random_state=342, 
                                           solver="acc_proxgd_restart", n_epoch=20000, warm_start=True,
                                           tol=1e-8, beta=0.8, d=20, sylvester_init=True, 
                                           n_jobs=-1, fit_best=True)
    
    # Use the selected estimator to predict on test set and display corresponding score
    pred_hub1 = best_esti_hub1.model.forward(Xtest)
    score_hub1 = torch.mean((pred_hub1 - Ytest) ** 2)
    print("Score Huber 1 loss: " + str(score_hub1))