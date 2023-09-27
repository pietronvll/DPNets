import os
import pickle
import time
from pathlib import Path
from typing import Optional

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


def load_data():
    pwised = np.load(exp_path / "data/CLN025_heavy_pwisedist.npy")
    pwised = torch.from_numpy(pwised)
    # Data Normalization
    feature_mean = pwised.mean(dim=0, keepdim=True)
    feature_std = pwised.std(dim=0, keepdim=True)
    rsqrt_feature_dim = (pwised.shape[1]) ** -0.5
    pwised = rsqrt_feature_dim * (pwised - feature_mean) / feature_std
    return pwised


def make_dataset(train_pts: Optional[int] = None):
    pwised = load_data()
    if train_pts is None:
        train_pts = pwised.shape[0] - 1
    train_pts = min(train_pts, pwised.shape[0] - 1)
    rnd_ids = np.random.choice(np.arange(pwised.shape[0] - 1), train_pts, replace=False)
    X = pwised[rnd_ids]
    Y = pwised[rnd_ids + 1]
    del pwised
    print(f"Shape {X.shape} Dtype {X.dtype} - {tensor_size(X):.1f} GB")
    return X, Y


def fit_and_evd(X, Y, kind: str, **kw):
    print(f"Training {kind.upper()}")
    start = time.time()
    est = train_est(X, Y, kind=kind, **kw)
    fit_time = time.time() - start
    print(f"Training took {fit_time:.1f} seconds")
    print("Computing EVD")
    data = load_data()
    start = time.time()
    evals, _left_fn, _right_fn = est.eigenfunctions()
    evals = evals.resolve_conj().numpy()
    efun_left = _left_fn(data).resolve_conj().numpy()
    efun_right = _right_fn(data).resolve_conj().numpy()
    evd_time = time.time() - start
    print(f"EVD took {evd_time:.1f} seconds")

    report = {
        "model": model,
        "fit_time": fit_time,
        "evd_time": evd_time,
        "eigenvalues": evals,
        "left_eigenfunctions": efun_left,
        "right_eigenfunctions": efun_right,
    }
    data_path = exp_path / "data/nystrom_evd"
    if not data_path.exists():
        data_path.mkdir(parents=True)
    with open(data_path / f"{model.upper()}_report.pkl", "wb") as f:
        pickle.dump(report, f)
    del report
    del data


def median_heuristic(num_sample_points: int = 10000):
    data = load_data()
    rnd_ids = np.random.choice(
        np.arange(data.shape[0] - 1), num_sample_points, replace=False
    )
    data = data[rnd_ids]
    dists = torch.cdist(data, data)
    i, j = torch.triu_indices(dists.shape[0], dists.shape[1])
    dists = dists[i, j]
    return dists.median().item()


if __name__ == "__main__":
    ls = median_heuristic()
    print(f"Length-scale from the Median Heuristics: {ls:.3f}")
    model_kw = {
        "pcr": {
            "num_components": configs.feature_dim,
            "M": configs.inducing_points,
            "kernel": GaussianKernel(ls, opt=FalkonOptions(use_cpu=True)),
            "svd_solver": "full",
        },
        "rrr": {
            "penalty": 1e-9,
            "num_components": configs.feature_dim,
            "M": configs.inducing_points,
            "kernel": GaussianKernel(ls, opt=FalkonOptions(use_cpu=True)),
        },
    }
    # Train Nystrom
    for model in model_kw.keys():
        kw = model_kw[model]
        fit_and_evd(*make_dataset(configs.num_data_pts), **kw)
