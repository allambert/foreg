import os
import sys
import torch
import pathlib

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
from estimator import RobustFORSpl, SparseFORSpl, FOREig, SparseFOREig, RobustFOREig
from kernel import Gaussian
from model import DecomposableIdentityScalar, DecomposableIntOp

seed1 = 344
seed2 = 452


if __name__ == "__main__":

    # Load data
    Xtrain, Ytrain, Xtest, Ytest = load_gp_dataset(seed1, seed2)
    theta = torch.linspace(0, 1, 100)

    # Define kernels
    kernel_input = Gaussian(0.01)
    kernel_output = Gaussian(100)

    # Regularization parameter
    lbda = 1e-5


    # ######################### UNCORRUPTED DATA #######################################
    # Estimator with square loss
    model_sqreig = DecomposableIntOp(kernel_input, kernel_output, n_eig=25, subsample=1)
    esti_sqreig = FOREig(model_sqreig, lbda)
    # Use solver Sylvester to compute close form, much faster
    esti_sqreig.fit(Xtrain, Ytrain, theta.view(-1, 1), solver="Sylvester")
    pred_sqreig = model_sqreig.forward(Xtest)
    score_sqreig = torch.mean((pred_sqreig - Ytest) ** 2)
    print("Score square loss: " + str(score_sqreig))


    # Sparse FOR with 2 norm
    model_eps2 = DecomposableIntOp(kernel_input, kernel_output, n_eig=25, subsample=1)
    eps = 1e-1
    esti_eps2 = SparseFOREig(model_eps2, lbda, loss_param=eps)
    esti_eps2.fit(Xtrain, Ytrain, theta.view(-1, 1), tol=1e-6)
    pred_eps2 = model_eps2.forward(Xtest)
    score_eps2 = torch.mean((pred_eps2 - Ytest) ** 2)
    print("Score epsilon 2 loss: " + str(score_eps2))
    print("Sparsity epsilon 2 loss: " + str(esti_eps2.get_sparsity_level()))


    # Sparse FOR with infinite norm
    model_epsinf = DecomposableIdentityScalar(kernel_input, kernel_output, subsample=1)
    eps = 1e-1
    esti_epsinf = SparseFORSpl(model_epsinf, lbda, loss_param=eps, norm="inf")
    esti_epsinf.fit(Xtrain, Ytrain, theta.view(-1, 1), tol=1e-6)
    pred_epsinf = model_epsinf.forward(Xtest)
    score_epsinf = torch.mean((pred_epsinf - Ytest) ** 2)
    print("Score epsilon infinite loss: " + str(score_epsinf))
    print("Sparsity epsilon infinite loss: " + str(esti_epsinf.get_sparsity_level()))


    # ######################### CORRUPTED DATA ###########################################
    # Corrupt data set with GP (type 2) outliers
    Ytrain_corr, _ = add_gp_outliers(Ytrain, Xeval=None, freq_sample=0.1, seed=443, 
                                     seed_gps=56, covs_params=(0.01, 0.05, 1, 4), 
                                     scale=2, intensity=3)

    # Estimator with square loss
    model_sqreig = DecomposableIntOp(kernel_input, kernel_output, n_eig=25, subsample=1)
    esti_sqreig = FOREig(model_sqreig, lbda)
    esti_sqreig.fit(Xtrain, Ytrain_corr, theta.view(-1, 1), solver="Sylvester")
    pred_sqreig = model_sqreig.forward(Xtest)
    score_sqreig = torch.mean((pred_sqreig - Ytest) ** 2)
    print("Score square loss: " + str(score_sqreig))

    # Robust FOR with 2 norm
    model_hub2 = DecomposableIntOp(kernel_input, kernel_output, n_eig=25, subsample=1)
    kappa = 5e-2
    esti_hub2 = RobustFOREig(model_hub2, lbda, loss_param=kappa)
    esti_hub2.fit(Xtrain, Ytrain_corr, theta.view(-1, 1), tol=1e-6)
    pred_hub2 = model_hub2.forward(Xtest)
    score_hub2 = torch.mean((pred_hub2 - Ytest) ** 2)
    print("Score Huber 2 loss: " + str(score_hub2))

    # Robust FOR with infinite norm
    model_hubinf = DecomposableIdentityScalar(kernel_input, kernel_output, subsample=1)
    kappa = 5e-2
    esti_hubinf = RobustFORSpl(model_hubinf, lbda, loss_param=kappa, norm="inf")
    esti_hubinf.fit(Xtrain, Ytrain_corr, theta.view(-1, 1), tol=1e-6)
    pred_hubinf = model_hubinf.forward(Xtest)
    score_hubinf = torch.mean((pred_hubinf - Ytest) ** 2)
    print("Score Huber 1 loss: " + str(score_hubinf))