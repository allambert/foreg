from typing import Iterable
import torch
import numpy as np
from sklearn.model_selection import KFold
import os
from joblib import delayed, Parallel, parallel_backend
from multiprocessing import cpu_count


def create_folder(folder):
    folder_split = str(folder).split("/")
    path = ""
    for fold in folder_split:
        path += "/" + fold
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

def interpret_n_jobs(n_jobs):
    if isinstance(n_jobs, str):
        if n_jobs == "max":
            n_jobs = -1
        elif "max/" in n_jobs:
            tot_cpus = cpu_count()
            sp = n_jobs.split("/")
            n_jobs = tot_cpus // int(sp[1])
    return n_jobs


def get_score(pred, Y, partial=False, metric_p=2):
    if partial:
        return np.mean(np.nansum(np.abs(pred.numpy() - Y) ** metric_p, axis=1))
    else:
        return torch.mean(torch.abs(pred - Y) ** metric_p).item()


def test_esti(esti, X, Y, G_x=None, partial=False, Ymean=None):
    pred = esti.model.forward(X, G_x)
    if Ymean is not None:
        if isinstance(Ymean, np.ndarray):
            pred += torch.from_numpy(Ymean)
        else:
            pred += Ymean
    return get_score(pred, Y, partial)


def cv_esti(esti, X, Y, thetas, G_x=None, G_t=None, Yeval=None,
            n_splits=5, metric_p=2, reduce_stat="median", random_state=342, solver="FISTA_restart",
            n_epoch=10000, warm_start=True, tol=1e-8, beta=0.8, monitor_loss=None, d=20, sylvester_init=True):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mse_local = []
    count = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        if G_x is not None:
            G_x_train = G_x[train_index, :][:, train_index]
            G_x_test = G_x[test_index, :][:, train_index]
        else:
            G_x_train, G_x_test = None, None
        esti.fit(X_train, Y_train, thetas, solver=solver, n_epoch=n_epoch, warm_start=warm_start,
                 tol=tol, beta=beta, monitor_loss=monitor_loss, d=d, G_x=G_x_train, G_t=G_t, verbose=False, sylvester_init=sylvester_init)
        pred_test = esti.model.forward(X_test, G_x_test)
        if Yeval is not None:
            mse_local.append(get_score(pred_test, Yeval[test_index], partial=True, metric_p=metric_p))
        else:
            mse_local.append(get_score(pred_test, Y_test, partial=False, metric_p=metric_p))
        # print("Done for fold number " + str(count))
        count += 1
    try:
        print("Done for lambda=" + str(esti.lbda) + " and loss_param=" + str(esti.loss_param))
    except AttributeError:
        try:
            print("Done for lambda=" + str(esti.lbda))
        except AttributeError:
            print("Done for loss_param=" + str(esti.loss_param))
    if reduce_stat == "mean":
        return torch.tensor(mse_local).mean()
    else:
        return torch.tensor(mse_local).quantile(0.5)


def tune_estis(estis, X, Y, thetas, G_x=None, G_t=None, Yeval=None,
                    n_splits=5, metric_p=2, reduce_stat="median", 
                    random_state=342, solver="FISTA_restart", n_epoch=20000, warm_start=True,
                    tol=1e-8, beta=0.8, monitor_loss=None, d=20, sylvester_init=True, n_jobs=-1, fit_best=True):
    n_jobs = interpret_n_jobs(n_jobs)
    with parallel_backend("loky"):
        mses = Parallel(n_jobs=n_jobs)(
            delayed(cv_esti)(esti, X, Y, thetas, G_x, G_t, Yeval, n_splits=n_splits,
                             metric_p=metric_p, reduce_stat=reduce_stat,
                             random_state=random_state, solver=solver,
                             n_epoch=n_epoch, warm_start=warm_start, tol=tol, beta=beta,
                             monitor_loss=monitor_loss, d=d, sylvester_init=sylvester_init)
            for esti in estis)
    mses = torch.tensor(mses)
    esti_argmin = mses.argmin()
    best_esti = estis[esti_argmin]
    if fit_best:
        best_esti.fit(X, Y, thetas, solver=solver, n_epoch=n_epoch, warm_start=warm_start,
                 tol=tol, beta=beta, monitor_loss=monitor_loss, d=d, G_x=G_x, G_t=G_t, verbose=False, 
                 sylvester_init=sylvester_init)
    return best_esti, mses