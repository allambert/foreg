import os
import sys
import torch
import pathlib


exec_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(exec_path))
sys.path.append(str(exec_path.parent))


from .synthetic_func_or import SyntheticGPmixture
from . import config_synth


if config_synth.DEFAULT_TORCH_DTYPE == "float64":
    torch.set_default_dtype(torch.float64)


def load_gp_dataset(train_coefs_seed, test_coefs_seed):
    n_samples = config_synth.N_SAMPLES
    gamma_cov = torch.Tensor([config_synth.STDS_GPS_IN, config_synth.STDS_GPS_OUT]).numpy()
    n_atoms = len(config_synth.STDS_GPS_IN)
    data_gp = SyntheticGPmixture(n_atoms=n_atoms, gamma_cov=gamma_cov, scale=config_synth.GP_SCALE)
    theta = torch.linspace(0, 1, config_synth.N_THETA)
    data_gp.drawGP(theta, seed_gp=config_synth.SEED_GPS)
    X_train, Y_train = data_gp.sample(n_samples, seed_coefs=train_coefs_seed)
    n_test = config_synth.N_TEST
    X_test, Y_test = data_gp.sample(n_test, seed_coefs=test_coefs_seed)
    return X_train, Y_train, X_test, Y_test




