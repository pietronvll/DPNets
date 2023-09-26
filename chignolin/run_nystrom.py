import os
import pickle
import time
from pathlib import Path

import mdtraj
import ml_confs
import numpy as np
import torch

os.environ["CUDA_PATH"] = "/home/novelli/anaconda3/envs/nyskoop"
from falkon import FalkonOptions
from falkon.kernels import GaussianKernel
from tqdm import tqdm

import sys  # isort:skip

sys.path.append("/home/novelli/koopmannystroem/nyskoop")
from nyskoop.data.DESRES_proteins import load_traj_metadata  # noqa: E402
from nystrom_helpers import *  # noqa: E402

# Common Paths
experiment_path = Path(__file__).parent
configs = ml_confs.from_file(
    experiment_path / "nystrom_cfg.yaml", register_jax_pytree=False
)
desres_data_path = "/home/novelli/data/md_datasets/DESRES_folding_trajs"
traj_meta = load_traj_metadata(
    protein_id=configs.protein_id, base_path=desres_data_path
)


def compute_pwise_distances(traj: mdtraj.Trajectory) -> np.ndarray:
    from itertools import combinations

    sel = traj.top.select("symbol != H")
    atom_pairs = list(combinations(sel, 2))
    return mdtraj.compute_distances(traj, atom_pairs)


def tensor_size(t: torch.Tensor) -> int:
    # Returns the size of a tensor in gigabytes
    nbytes = t.element_size() * t.numel()
    return nbytes / 1e9


def get_data(num_data_pts: int):
    flist = traj_meta["trajectory_files"]
    top = traj_meta["topology_path"]

    pwised = []
    for chunk in tqdm(flist, unit="chunk", desc="Load+Preprocess trajectory"):
        traj = mdtraj.load(chunk, top=top)
        dists = compute_pwise_distances(traj)
        pwised.append(dists)

    pwised = np.concatenate(pwised, axis=0)
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
        with open(experiment_path / f"ckpt/nystrom_{model}.pkl", "wb") as f:
            pickle.dump(report, f)
