import os
import pickle
import time
from pathlib import Path

import mdtraj
import ml_confs
import numpy as np
import torch
from falkon import FalkonOptions
from falkon.kernels import GaussianKernel
from nystrom_helpers import *
from tqdm import tqdm

exp_path = Path(__file__).parent.parent
configs = ml_confs.from_file(
    exp_path / "nystrom/configs.yaml", register_jax_pytree=False
)


def tensor_size(t: torch.Tensor) -> int:
    # Returns the size of a tensor in gigabytes
    nbytes = t.element_size() * t.numel()
    return nbytes / 1e9


def get_data(num_data_pts: int):
    pwised = np.load(exp_path / "data/CLN025_heavy_pwisedist.npy")
    num_data_pts = min(num_data_pts, pwised.shape[0] - 1)
    pwised = torch.from_numpy(pwised)
    # Data Normalization
    feature_norm = torch.linalg.norm(pwised, dim=0, keepdim=True)
    feature_mean = torch.mean(pwised, dim=0, keepdim=True)
    pwised -= feature_mean
    pwised /= feature_norm

    rnd_ids = np.random.choice(
        np.arange(pwised.shape[0] - 1), num_data_pts, replace=False
    )
    X = pwised[rnd_ids]
    Y = pwised[rnd_ids + 1]
    del pwised
    print(f"Shape {X.shape} Dtype {X.dtype} - {tensor_size(X):.1f} GB")
    return X, Y


if __name__ == "__main__":
    model_kw = {
        "pcr": {
            "num_components": configs.feature_dim,
            "M": configs.inducing_points,
            "kernel": GaussianKernel(configs.ls, opt=FalkonOptions(use_cpu=True)),
            "svd_solver": "full",
        },
        "rrr": {
            "penalty": 1e-9,
            "num_components": configs.feature_dim,
            "M": configs.inducing_points,
            "kernel": GaussianKernel(configs.ls, opt=FalkonOptions(use_cpu=True)),
        },
    }
    X, Y = get_data(num_data_pts=configs.num_data_pts)
    print("Data Loaded")
    # Data shuffling
    rng = np.random.default_rng()
    # Train Nystrom
    for model in model_kw.keys():
        print(f"Training {model.upper()}")
        start = time.time()
        est = train_est(X, Y, kind=model, **model_kw[model])
        t_elapsed = time.time() - start
        print(f"Training took {t_elapsed:.1f} seconds")
        v, lf, rf = evals, efun_left, efun_right = est.eigenfunctions()
        report = {
            "model": model,
            "fit_time": t_elapsed,
            "eigenvalues": v.detach().resolve_conj().numpy(),
            "estimator": est,
        }
        with open(exp_path / f"ckpt/nystrom_{model}.pkl", "wb") as f:
            pickle.dump(report, f)
