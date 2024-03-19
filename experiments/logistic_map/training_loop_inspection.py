import argparse
import functools
import logging
import pickle
from pathlib import Path
from time import perf_counter

import lightning
import ml_confs
import numpy as np
import torch
from kooplearn._src.metrics import directed_hausdorff_distance
from kooplearn._src.utils import topk
from kooplearn.abc import FeatureMap
from kooplearn.datasets import LogisticMap
from kooplearn.nn.data import TrajToContextsDataset
from lightning.pytorch.callbacks import Callback
from scipy.integrate import romb
from torch.utils.data import DataLoader

# General definitions
experiment_path = Path(__file__).parent
ckpt_path = experiment_path / "ckpt"
results_path = experiment_path / "results_training_loop_inspection"
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
    "enable_progress_bar": True,
    "enable_model_summary": False,
    "enable_checkpointing": False,
    "logger": False,
}


# Evaluation callback
class HausdorffCB(Callback):
    def __init__(self):
        super().__init__()
        self.distance = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        fmap = pl_module._kooplearn_feature_map_weakref()
        if (trainer.global_step - 1) % 10 == 0:
            d_H = evaluate_representation(fmap)
            self.distance.append(d_H)


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
    # Compute OLS estimator
    cov, cross_cov = population_covs(feature_map)
    OLS_estimator = np.linalg.solve(cov, cross_cov)
    # Eigenvalue estimation
    OLS_eigs = np.linalg.eigvals(OLS_estimator)
    top_eigs = topk(np.abs(OLS_eigs), 3)
    OLS_eigs = OLS_eigs[top_eigs.indices]

    true_eigs = logistic.eig()
    top_eigs = topk(np.abs(true_eigs), 3)
    true_eigs = true_eigs[top_eigs.indices]

    return directed_hausdorff_distance(OLS_eigs, true_eigs)


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
def run_VAMPNets(feature_dim: int, rng_seed: int):
    logger.info(f"VAMPNets::START::FeatureDim {feature_dim}")
    logger.info(f"VAMPNets::Seed {rng_seed + 1}")
    result = _run_VAMPNets(rng_seed, feature_dim)
    logger.info("VAMPNets::END")
    return result


def _run_VAMPNets(rng_seed: int, feature_dim: int):
    from kooplearn.models.feature_maps import VAMPNet

    hausdorff_cb = HausdorffCB()
    trainer = lightning.Trainer(**trainer_kwargs, callbacks=[hausdorff_cb])
    net_kwargs = {"feature_dim": feature_dim, "layer_dims": configs.layer_dims}
    # Defining the model
    vamp_fmap = VAMPNet(
        SimpleMLP,
        opt,
        trainer,
        encoder_kwargs=net_kwargs,
        optimizer_kwargs={"lr": 5e-5},
        encoder_timelagged=SimpleMLP,
        encoder_timelagged_kwargs=net_kwargs,
        center_covariances=False,
        seed=rng_seed,
    )
    # Init
    torch.manual_seed(rng_seed)
    kaiming_init(vamp_fmap.lightning_module.encoder)
    kaiming_init(vamp_fmap.lightning_module.encoder_timelagged)
    best_lr = tune_learning_rate(trainer, vamp_fmap, train_dl)
    assert vamp_fmap.lightning_module.lr == best_lr
    vamp_fmap.fit(train_dl)
    return hausdorff_cb.distance


def _run_DPNets(
    relaxed: bool, metric_deformation_coeff: float, rng_seed: int, feature_dim: int
):
    from kooplearn.models.feature_maps import DPNet

    hausdorff_cb = HausdorffCB()
    trainer = lightning.Trainer(**trainer_kwargs, callbacks=[hausdorff_cb])
    net_kwargs = {"feature_dim": feature_dim, "layer_dims": configs.layer_dims}
    # Defining the model
    dpnet_fmap = DPNet(
        SimpleMLP,
        opt,
        trainer,
        use_relaxed_loss=relaxed,
        metric_deformation_loss_coefficient=metric_deformation_coeff,
        encoder_kwargs=net_kwargs,
        optimizer_kwargs={"lr": 5e-5},
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
    dpnet_fmap.fit(train_dl)
    return hausdorff_cb.distance


def run_DPNets(feature_dim: int, rng_seed: int):
    logger.info(f"DPNets::START::FeatureDim {feature_dim}")
    relaxed = False
    metric_deformation = 1.0
    logger.info(f"DPNets::Tuned metric deformation: {metric_deformation}")
    logger.info(f"DPNets::Seed {rng_seed + 1}")
    result = _run_DPNets(relaxed, metric_deformation, rng_seed, feature_dim)
    logger.info(f"DPNets::END::FeatureDim {feature_dim}")
    return result


def run_DPNets_relaxed(feature_dim: int, rng_seed: int):
    logger.info(f"DPNets-relaxed::START::FeatureDim {feature_dim}")
    relaxed = True
    metric_deformation = 1.0
    logger.info(f"DPNets-relaxed::Tuned metric deformation: {metric_deformation}")
    logger.info(f"DPNets::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
    result = _run_DPNets(relaxed, metric_deformation, rng_seed, feature_dim)
    logger.info(f"DPNets-relaxed::END::FeatureDim {feature_dim}")
    return result


AVAIL_MODELS = {
    "VAMPNets": run_VAMPNets,
    "DPNets": run_DPNets,
    "DPNets-relaxed": run_DPNets_relaxed,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the experiment on a specific model."
    )
    parser.add_argument("--model", help="Specify the model to run.")
    parser.add_argument(
        "--rngseed",
        type=int,
        default=0,
        help="Specify the rng seed",
    )

    args = parser.parse_args()

    if args.model:
        if args.model in AVAIL_MODELS:
            results = AVAIL_MODELS[args.model](configs.feature_dim, args.rngseed)
            fname = sanitize_filename(args.model) + f"_{args.rngseed}.pkl"
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
