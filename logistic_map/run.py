import argparse
import functools
import logging
import pickle
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Union

import lightning
import ml_confs
import numpy as np
import optuna
import scipy.special
import torch
from kooplearn._src.metrics import directed_hausdorff_distance
from kooplearn._src.utils import topk
from kooplearn.abc import FeatureMap
from kooplearn.datasets import LogisticMap
from kooplearn.models.feature_maps import ConcatenateFeatureMaps
from kooplearn.nn.data import TrajToContextsDataset
from lightning.pytorch.callbacks import EarlyStopping
from scipy.integrate import romb
from torch.utils.data import DataLoader

# General definitions
experiment_path = Path(__file__).parent
data_path = experiment_path / "data"
ckpt_path = experiment_path / "ckpt"
results_path = experiment_path / "results"
configs = ml_confs.from_file(
    experiment_path / "configs.yaml", register_jax_pytree=False
)

# Logging
logger = logging.getLogger("logistic_map")
logger.setLevel(logging.INFO)
log_file = experiment_path / "logs/run.log"  # Specify the path to your log file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Model Initialization
logistic = LogisticMap(N=configs.N, rng_seed=0)  # Reproducibility
# Data pipeline
sample_traj = logistic.sample(
    0.5, configs.num_train + configs.num_val + configs.num_test
)

dataset = {
    "train": sample_traj[: configs.num_train],
    "validation": sample_traj[configs.num_train : configs.num_train + configs.num_val],
    "test": sample_traj[configs.num_train + configs.num_val :],
}

train_data = torch.from_numpy(dataset["train"]).float()
val_data = torch.from_numpy(dataset["validation"]).float()

train_ds = TrajToContextsDataset(train_data)
val_ds = TrajToContextsDataset(val_data)

train_dl = DataLoader(train_ds, batch_size=configs.batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

# Init report dict
dl_reports = {}

opt = torch.optim.Adam

trainer_kwargs = {
    "accelerator": "cpu",
    "devices": 1,
    "max_epochs": configs.max_epochs,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "enable_checkpointing": False,
    "logger": False,
}


# Adapted from https://realpython.com/python-timer/#creating-a-python-timer-decorator
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        value = func(*args, **kwargs)
        toc = perf_counter()
        elapsed_time = toc - tic
        return value, elapsed_time

    return wrapper_timer


def stack_reports(reports):
    stacked = {}
    for key in [
        "hausdorff-distance",
        "optimality-gap",
        "feasibility-gap",
        "fit-time",
        "time_per_epoch",
    ]:
        if key in reports[0]:
            stacked[key] = np.mean([r[key] for r in reports])
            key_std = key + "_std"
            stacked[key_std] = np.std([r[key] for r in reports])
    for key in ["estimator-eigenvalues", "covariance-eigenvalues"]:
        stacked[key] = np.stack([r[key] for r in reports])
    return stacked


def sanitize_filename(filename):
    # Define a set of characters that are not allowed in file names on most systems
    illegal_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    # Replace any illegal characters with an underscore
    for char in illegal_chars:
        filename = filename.replace(char, "_")
    # Remove leading and trailing spaces
    filename = filename.strip()
    # Remove dots and spaces from the beginning and end of the filename
    filename = filename.strip(". ")
    # Ensure the filename is not empty and is not just composed of dots
    if not filename:
        filename = "unnamed"
    # Limit the filename length to a reasonable number of characters
    max_length = 255  # Max file name length on most systems
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename


def population_covs(feature_map: FeatureMap, pow_of_two_k: int = 12):
    """Computes the population covariance and cross-covariance"""
    x = np.linspace(0, 1, 2**pow_of_two_k + 1)[:, None]
    vals, lv = logistic.eig(eval_left_on=x)
    perron_eig_idx = np.argmax(np.abs(vals))
    pi = lv[:, perron_eig_idx]
    assert np.isreal(pi).all()
    pi = pi.real
    pi = pi / romb(pi, dx=1 / 2**pow_of_two_k)  # Normalization of π
    # Evaluating the feature map
    phi = feature_map(x)  # [2**pow_of_two_k + 1, d]
    # Covariance
    cov_unfolded = (
        phi.reshape(2**pow_of_two_k + 1, -1, 1)
        * phi.reshape(2**pow_of_two_k + 1, 1, -1)
        * pi.reshape(-1, 1, 1)
    )
    cov = romb(cov_unfolded, dx=1 / 2**pow_of_two_k, axis=0)
    # Cross-covariance
    alphas = np.stack(
        [logistic.noise_feature_composed_map(x, n) for n in range(logistic.N + 1)],
        axis=1,
    )
    betas = np.stack(
        [logistic.noise_feature(x, n) for n in range(logistic.N + 1)], axis=1
    )

    cov_alpha_unfolded = (
        phi.reshape(2**pow_of_two_k + 1, -1, 1)
        * alphas.reshape(2**pow_of_two_k + 1, 1, -1)
        * pi.reshape(-1, 1, 1)
    )
    cov_beta_unfolded = phi.reshape(2**pow_of_two_k + 1, -1, 1) * betas.reshape(
        2**pow_of_two_k + 1, 1, -1
    )

    cov_alpha = romb(cov_alpha_unfolded, dx=1 / 2**pow_of_two_k, axis=0)
    cov_beta = romb(cov_beta_unfolded, dx=1 / 2**pow_of_two_k, axis=0)

    cross_cov = cov_alpha @ (cov_beta.T)
    return cov, cross_cov


def evaluate_representation(feature_map: FeatureMap):
    report = {}
    # Compute OLS estimator
    cov, cross_cov = population_covs(feature_map)
    OLS_estimator = np.linalg.solve(cov, cross_cov)
    # Eigenvalue estimation
    OLS_eigs = np.linalg.eigvals(OLS_estimator)
    top_eigs = topk(np.abs(OLS_eigs), 3)
    OLS_eigs = OLS_eigs[top_eigs.indices]

    report["hausdorff-distance"] = directed_hausdorff_distance(OLS_eigs, logistic.eig())
    # VAMP2-score
    M = np.linalg.multi_dot(
        [
            np.linalg.pinv(cov, hermitian=True),
            cross_cov,
            np.linalg.pinv(cov, hermitian=True),
            cross_cov.T,
        ]
    )
    feature_dim = cov.shape[0]
    report["optimality-gap"] = np.sum(logistic.svals()[:feature_dim] ** 2) - np.trace(M)
    # Feasibility
    report["feasibility-gap"] = np.linalg.norm(cov - np.eye(feature_dim), ord=2)
    report["estimator-eigenvalues"] = OLS_eigs
    report["covariance-eigenvalues"] = np.linalg.eigvalsh(cov)
    return report


def tune_learning_rate(
    trainer: lightning.Trainer,
    model: FeatureMap,
    train_dataloader: DataLoader,
):
    # See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner.lr_find for an explanation of how the lr is chosen
    tuner = lightning.pytorch.tuner.Tuner(trainer)
    # Run learning rate finder
    lr_finder = tuner.lr_find(
        model.lightning_module,
        train_dataloaders=train_dataloader,
        min_lr=1e-6,
        max_lr=1e-4,
        num_training=50,
        early_stop_threshold=None,
        update_attr=True,
    )
    return lr_finder.suggestion()


# Models
class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Assuming x is in [0, 1]
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class SimpleMLP(torch.nn.Module):
    def __init__(
        self, feature_dim: int, layer_dims: list[int], activation=torch.nn.LeakyReLU
    ):
        super().__init__()
        self.activation = activation
        lin_dims = (
            [2] + layer_dims + [feature_dim]
        )  # The 2 is for the sinusoidal embedding

        layers = []
        for layer_idx in range(len(lin_dims) - 2):
            layers.append(
                torch.nn.Linear(
                    lin_dims[layer_idx], lin_dims[layer_idx + 1], bias=False
                )
            )
            layers.append(activation())
        layers.append(torch.nn.Linear(lin_dims[-2], lin_dims[-1], bias=True))
        self.layers = torch.nn.ModuleList(layers)
        self.sin_embedding = SinusoidalEmbedding()

    def forward(self, x):
        # Sinusoidal embedding
        x = self.sin_embedding(x)
        # MLP
        for layer in self.layers:
            x = layer(x)
        return x


def kaiming_init(model):
    for p in model.parameters():
        psh = p.shape
        if len(psh) == 2:  # Linear layers
            _, in_shape = psh
            if in_shape == 2:  # Initial layer
                torch.nn.init.uniform_(p, -1, 1)
            else:
                acname = model.activation.__name__.lower()
                if acname == "leakyrelu":
                    acname = "leaky_relu"
                torch.nn.init.kaiming_uniform_(p, a=1, nonlinearity=acname)
        else:  # Bias
            torch.nn.init.zeros_(p)


# Runners
def run_VAMPNets(feature_dim: int):
    logger.info(f"VAMPNets::START::FeatureDim {feature_dim}")
    reports = []
    for rng_seed in range(configs.num_rng_seeds):
        logger.info(f"VAMPNets::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
        result = _run_VAMPNets(rng_seed, feature_dim)
        reports.append(result)
    full_report = stack_reports(reports)
    logger.info(f"VAMPNets::END")
    return full_report


def _run_VAMPNets(rng_seed: int, feature_dim: int):
    from kooplearn.models.feature_maps import VAMPNet

    trainer = lightning.Trainer(**trainer_kwargs)
    net_kwargs = {"feature_dim": feature_dim, "layer_dims": configs.layer_dims}
    # Defining the model
    vamp_fmap = VAMPNet(
        SimpleMLP,
        opt,
        {"lr": 5e-5},
        trainer,
        net_kwargs,
        lobe_timelagged=SimpleMLP,
        lobe_timelagged_kwargs=net_kwargs,
        center_covariances=False,
        seed=rng_seed,
    )
    # Init
    torch.manual_seed(rng_seed)
    kaiming_init(vamp_fmap.lightning_module.lobe)
    kaiming_init(vamp_fmap.lightning_module.lobe_timelagged)
    best_lr = tune_learning_rate(trainer, vamp_fmap, train_dl)
    assert vamp_fmap.lightning_module.lr == best_lr
    _, fit_time = timer(vamp_fmap.fit)(train_dl)
    report = evaluate_representation(vamp_fmap)
    report["time_per_epoch"] = fit_time / configs.max_epochs
    return report


def _run_DPNets(
    relaxed: bool, metric_deformation_coeff: float, rng_seed: int, feature_dim: int
):
    from kooplearn.models.feature_maps import DPNet

    trainer = lightning.Trainer(**trainer_kwargs)
    net_kwargs = {"feature_dim": feature_dim, "layer_dims": configs.layer_dims}
    # Defining the model
    dpnet_fmap = DPNet(
        SimpleMLP,
        opt,
        {"lr": 5e-5},
        trainer,
        use_relaxed_loss=relaxed,
        metric_deformation_loss_coefficient=metric_deformation_coeff,
        encoder_kwargs=net_kwargs,
        encoder_timelagged=SimpleMLP,
        encoder_timelagged_kwargs=net_kwargs,
        center_covariances=False,
        seed=rng_seed,
    )
    # Init
    torch.manual_seed(rng_seed)
    kaiming_init(dpnet_fmap.lightning_module.encoder)
    kaiming_init(dpnet_fmap.lightning_module.encoder_timelagged)
    best_lr = tune_learning_rate(trainer, dpnet_fmap, train_dl)
    assert dpnet_fmap.lightning_module.lr == best_lr
    _, fit_time = timer(dpnet_fmap.fit)(train_dl)
    report = evaluate_representation(dpnet_fmap)
    report["time_per_epoch"] = fit_time / configs.max_epochs
    return report


def _tune_DPNets_metric_deformation(relaxed: bool, rng_seed: int, feature_dim: int):
    def objective(trial):
        metric_deformation = trial.suggest_float(
            "metric_deformation", 1e-2, 1, log=True
        )
        report = _run_DPNets(relaxed, metric_deformation, rng_seed, feature_dim)
        return report["optimality-gap"] + report["feasibility-gap"]

    sampler = optuna.samplers.TPESampler(seed=0)  # Reproductibility
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=configs.trial_budget)
    return study.best_params["metric_deformation"]


def run_DPNets(feature_dim: int):
    logger.info(f"DPNets::START::FeatureDim {feature_dim}")
    relaxed = False
    report = []
    tuner_rng_seed = 171192  # Reproductibility
    # metric_deformation = _tune_DPNets_metric_deformation(
    #     relaxed, tuner_rng_seed, feature_dim
    # )
    metric_deformation = 1.0
    logger.info(f"DPNets::Tuned metric deformation: {metric_deformation}")
    for rng_seed in range(configs.num_rng_seeds):
        logger.info(f"DPNets::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
        result = _run_DPNets(relaxed, metric_deformation, rng_seed, feature_dim)
        report.append(result)
    full_report = stack_reports(report)
    logger.info(f"DPNets::END::FeatureDim {feature_dim}")
    return full_report


def run_DPNets_relaxed(feature_dim: int):
    logger.info(f"DPNets-relaxed::START::FeatureDim {feature_dim}")
    relaxed = True
    report = []
    tuner_rng_seed = 171192  # Reproductibility
    # metric_deformation = _tune_DPNets_metric_deformation(
    #     relaxed, tuner_rng_seed, feature_dim
    # )
    metric_deformation = 1.0
    logger.info(f"DPNets-relaxed::Tuned metric deformation: {metric_deformation}")
    for rng_seed in range(configs.num_rng_seeds):
        logger.info(f"DPNets::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
        result = _run_DPNets(relaxed, metric_deformation, rng_seed, feature_dim)
        report.append(result)
    full_report = stack_reports(report)
    logger.info(f"DPNets-relaxed::END::FeatureDim {feature_dim}")
    return full_report


def run_ChebyT(feature_dim: int):
    logger.info(f"ChebyT::START::FeatureDim {feature_dim}")

    def ChebyT(feature_dim: int = 3):
        def scaled_chebyt(n, x):
            return scipy.special.eval_chebyt(n, 2 * x - 1)

        fn_list = [partial(scaled_chebyt, n) for n in range(feature_dim)]
        return ConcatenateFeatureMaps(fn_list)

    full_report = evaluate_representation(ChebyT(feature_dim))
    logger.info(f"ChebyT::END::FeatureDim {feature_dim}")
    return full_report


def run_NoiseKernel(feature_dim: int):
    import scipy.special

    logger.info(f"NoiseKernel::START::FeatureDim {feature_dim}")

    def NoiseKernel(order: int = 3):
        binom_coeffs = [scipy.special.binom(configs.N, i) for i in range(configs.N + 1)]
        sorted_coeffs = np.argsort(binom_coeffs)

        def noise_feat(n, x):
            return logistic.noise_feature(x, n)

        fn_list = [partial(noise_feat, n) for n in sorted_coeffs[:order]]

        return ConcatenateFeatureMaps(fn_list)

    full_report = evaluate_representation(NoiseKernel(feature_dim))
    logger.info(f"NoiseKernel::END::FeatureDim {feature_dim}")
    return full_report


AVAIL_MODELS = {
    "VAMPNets": run_VAMPNets,
    "DPNets": run_DPNets,
    "DPNets-relaxed": run_DPNets_relaxed,
    "Cheby-T": run_ChebyT,
    "NoiseKernel": run_NoiseKernel,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the experiment on a specific model."
    )
    parser.add_argument("--model", help="Specify the model to run.")
    parser.add_argument(
        "--fdim",
        type=int,
        default=3,
        help="Specify the feature dimension.",
    )

    args = parser.parse_args()

    if args.model:
        if args.model in AVAIL_MODELS:
            results = AVAIL_MODELS[args.model](args.fdim)
            results["name"] = args.model
            fname = sanitize_filename(args.model) + f"_results_{args.fdim}.pkl"
            if not results_path.exists():
                results_path.mkdir()
            with open(results_path / fname, "wb") as f:
                pickle.dump(results, f)
        else:
            print(f"Model '{args.model}' is not available. Available models:")
            for model_name in AVAIL_MODELS:
                print(f"- {model_name}")
    else:
        print("Available models:")
        for model_name in AVAIL_MODELS:
            print(f"- {model_name}")


if __name__ == "__main__":
    main()
