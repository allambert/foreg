# Functional output regression with Huber and epsilon insensitive losses

## I / Dependencies 
This code was developped using Python 3.8, the required packages are listed in the file **requirements.txt**. 

*Remark: **slycot** is easily installed using conda whereas using pip can raise issues:*

```conda install -c conda-forge slycot```

## II / Estimators and Models
This package implements estimators for functional output regression using reproducing kernel Hilbert spaces (RKHS) of function-valued functions as hypothesis class for various losses promoting either sparsity (epsilon insensitive losses) or robustness (Huber losses). The proposed solvers are based on dualization. 

### A / Models
The base models are built using identity separable kernels modelling the output functions in a RKHS of scalar-valued functions. We propose two implementations based on different representation schemes for the (functional) dual variables in `model\decomposable.py`
- The class `DecomposableIdentityScalar` uses linear splines.
- The class `DecomposableIntOp` uses a truncated (approximate) set of eigenfunctions associated to the integral operator of the output RKHS.

### B / Estimators
To estimate those models, we propose estimators which correspond to different losses in `estimator\func_or.py`. Each estimator is representation dependent and can therefore only be used in conjuction with its appropriate model.

- The class `FORSpl` implements functional output regression with square loss. It must be used with a `DecomposableIdentityScalar` model.

- The class `FOREig` implements functional output regression with square loss. It must be used with a `DecomposableIntOp` model.

- The class `SparseFORSpl` implements functional output regression with epsilon insensitive loss (using either the 2 norm or the infinite norm). It must be used with a `DecomposableIdentityScalar` model.

- The class `SparseFOREig` implements functional output regression with epsilon insensitive loss (2 norm only). It must be used with a `DecomposableIntOp` model.

- The class `RobustFORSpl` implements functional output regression with Huber loss (using either the 2 norm or the infinite norm). It must be used with a `DecomposableIdentityScalar` model.

- The class `RobustFOREig` implements functional output regression with epsilon insensitive loss (2 norm only). It must be used with a `DecomposableIntOp` model.

### C / Solvers
Several solvers are proposed. They can be chosen when fitting the estimators

- For epsilon insensitive and Huber losses, either *FISTA* or *FISTA with restarts* can be used.

- For the square loss, an additional solver based on Sylvester equation solver is proposed and is much faster. 

### D / Overview of relevant parameters
We give a brief overview of some key parameters, we refer to section III and the corresponding script for examples.

#### `DecomposableIdentityScalar` 
- **kernel_input**: Kernel for the input data, see `kernel/kernels.py` for examples of valid kernels
- **kernel_output**: Kernel of the output RKHS, see `kernel/kernels.py` for examples of valid kernels
- **subsample (int)**: Whether to downsample the output function by a factor 1 / subsample

#### `DecomposableIntOp` 
- **kernel_input**: Kernel for the input data, see `kernel/kernels.py` for examples of valid kernels
- **kernel_output**: Kernel of the output RKHS, see `kernel/kernels.py` for examples of valid kernels
- **n_eig (int)**: Number of eigenfunctions to use in the approximation of the output functions
- **subsample (int)** : Whether to downsample the output function by a factor 1 / subsample

#### `FORSpl`and `FOREig`
- **lbda (float)**: Regularization parameter

#### `SparseFORSpl` and `RobustFORSpl`
- **lbda (float)**: Regularization parameter
- **loss_param (float)**: Parameter for the loss, either epsilon for epsilon-insensitive losses or kappa for Huber losses
- **norm (str)**: Norm to use, either {"2", "inf"}. For Huber loss, the "inf" corresponds to the Huber 1 loss.

#### `SparseFOREig` and `RobustFOREig`
- **lbda (float)**: Regularization parameter
- **loss_param (float)**: Parameter for the loss, either epsilon for epsilon-insensitive losses or kappa for Huber losses

#### `fit` method of all estimators
- **x (torch.Tensor)**: Input data
- **y (torch.Tensor)**: Output data
- **thetas (torch.Tensor)**: Sampling locations of the observations in y
- **solver (str)**: The solver to use, must be in {"FISTA", "FISTA_restart", "Sylvester"}. Note that Sylvester is valid only for the square loss and will raise an Error if passed for other losses. Default is "FISTA_restart".
- **n_epochs (int)**: Maximum number of iterations
- **warm_start (bool)**: Should the estimator be re-initialized
- **tol (float)**: Stopping criterion. Iterative procedure stops when the normalized distance between consecutive iterates gets below this parameter
- **beta (float)**: Parameter for line search, must have 0 < beta < 1
- **sylvester_init (bool)**: Should initialization with close form for the square loss computed with Sylvester solver be used as initialization
- **verbose (bool)**: Should details of iterations be displayed during fitting

## III / Quick start with simple examples
We provide a tool for generating a synthetic function to function regression dataset based on Gaussian processes in `datasets`.

Simple examples running the proposed approaches on such synthetic dataset can be found in `demos`.

- In the script `demos\synth.py`, we demonstrate basic usage of the models and estimators and give a brief overview of the most relevant parameters.

- In the script `demos\synth_robust_cv.py`, we demonstrate how to select estimators based on cross-validation using the functions provided in `model_selection`. 


## Cite
The losses and the corresponding estimators are described extensively in the paper https://allambert.github.io/files/pdf/robust_for.pdf which will appear soon in the PMLR proceedings of ICML 2022. 

 Please cite it when using this library
```
@inproceedings{brault2019infinite,
  title={Functional Output Regression with Infimal Convolution:
Exploring the Huber and $\epsilon$-insensitive Losses},
  author={Lambert, Alex and Bouche, Dimitri and Szab{\'o}, Zolt{\'a}n and d’Alch{\'e}-Buc, Florence},
  booktitle={The 39th International Conference on Machine Learning (ICML 2022)},
  year={2022},
  organization={PMLR}
}
```
