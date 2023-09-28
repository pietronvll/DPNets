import argparse
import functools
import logging
import pickle
from pathlib import Path
from time import perf_counter
from typing import Union

import lightning
import ml_confs
import numpy as np
import torch
from kooplearn.abc import BaseModel, TrainableFeatureMap
from kooplearn.data import traj_to_contexts
from kooplearn.nn.data import TrajToContextsDataset

# Paths and configs
experiment_path = Path(__file__).parent
data_path = experiment_path / "data"
ckpt_path = experiment_path / "ckpt"
results_path = experiment_path / "results"
configs = ml_confs.from_file(
    experiment_path / "configs.yaml", register_jax_pytree=False
)

# Logging
logger = logging.getLogger("fluid_flow")
logger.setLevel(logging.INFO)
log_file = experiment_path / "logs/run.log"  # Specify the path to your log file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

opt = torch.optim.Adam

# Trainer Configuration
trainer_kwargs = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": configs.max_epochs,
    "enable_progress_bar": False,
    "enable_checkpointing": False,
    "logger": False,
}


class MLP(torch.nn.Module):
    def __init__(self, feature_dim, layer_widths: list[int], activation=torch.nn.ReLU):
        super().__init__()
        self.activation = activation
        lin_dims = [feature_dim] + layer_widths
        layers = []
        for i in range(len(layer_widths)):
            layers.append(torch.nn.Linear(lin_dims[i], lin_dims[i + 1], bias=True))
            layers.append(activation())
        # Remove last activation
        layers.pop()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        # Flatten input
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x


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


class MLP_r(torch.nn.Module):
    def __init__(
        self,
        feature_dim,
        layer_widths: list[int],
        input_dims: tuple,
        activation=torch.nn.ReLU,
    ):
        super().__init__()
        self.activation = activation
        lin_dims = np.flip([feature_dim] + layer_widths).tolist()
        layers = []
        for i in range(len(layer_widths)):
            layers.append(torch.nn.Linear(lin_dims[i], lin_dims[i + 1], bias=True))
            layers.append(activation())
        # Remove last activation
        layers.pop()
        self.layers = torch.nn.ModuleList(layers)
        self.input_dims = input_dims

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # Restore dimensions
        x = x.view(x.shape[0], *self.input_dims)
        return x


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


def tune_learning_rate(
    trainer: lightning.Trainer,
    model: Union[BaseModel, TrainableFeatureMap],
    train_dataloader: torch.utils.data.DataLoader,
):
    # See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner.lr_find for an explanation of how the lr is chosen
    tuner = lightning.pytorch.tuner.Tuner(trainer)
    # Run learning rate finder
    lr_finder = tuner.lr_find(
        model.lightning_module,
        train_dataloaders=train_dataloader,
        min_lr=1e-6,
        max_lr=1e-3,
        num_training=50,
        early_stop_threshold=None,
        update_attr=True,
    )
    return lr_finder.suggestion()


# Init fn
def kaiming_init(model, trail_dims):
    feature_dim = np.prod(trail_dims)
    for p in model.parameters():
        psh = p.shape
        if len(psh) == 2:  # Linear layers
            _, in_shape = psh
            if in_shape == feature_dim:  # Initial layer
                torch.nn.init.uniform_(p, -1, 1)
            else:
                acname = model.activation.__name__.lower()
                if acname == "leakyrelu":
                    acname = "leaky_relu"
                torch.nn.init.kaiming_uniform_(p, a=1, nonlinearity=acname)
        else:  # Bias
            torch.nn.init.zeros_(p)


def svd_init(model, trail_dims, Vh):
    feature_dim = np.prod(trail_dims)
    for p in model.parameters():
        psh = p.shape
        if len(psh) == 2:  # Linear layers
            out_shape, in_shape = psh
            if in_shape == feature_dim:  # Initial layer
                p.data = torch.from_numpy(Vh[:out_shape])
            else:
                acname = model.activation.__name__.lower()
                if acname == "leakyrelu":
                    acname = "leaky_relu"
                torch.nn.init.kaiming_uniform_(p, a=1, nonlinearity=acname)
        else:  # Bias
            torch.nn.init.zeros_(p)


def merge_channels(data_dict, split="train", subsample: int = 1) -> np.ndarray:
    channels = [data_dict[key][:, :, None, :] for key in ["U", "V", "P", "C"]]
    channels = np.concatenate(channels, axis=2).transpose((3, 0, 1, 2))  # BHWC
    channels = channels[:, ::subsample, ::subsample, :]
    # Standardize each channel independently
    mean = channels.mean(axis=(0, 1, 2), keepdims=True)
    std = channels.std(axis=(0, 1, 2), keepdims=True)
    channels = (channels - mean) / std
    if split == "train":
        return channels[:160], (mean, std)
    elif split == "test":
        return channels[160:], (mean, std)
    else:
        raise ValueError(f"Unknown split {split}")


def load_data(experiment_path: Path, configs: ml_confs.Configs):
    with np.load(experiment_path / "data/cylinder2d_remesh.npz") as f:
        full_data = {k: np.asarray(v) for k, v in f.items()}
    train_ds, (mean, std) = merge_channels(
        full_data, split="train", subsample=configs.subsample
    )
    test_ds, _ = merge_channels(full_data, split="test", subsample=configs.subsample)
    return train_ds, test_ds, np.squeeze(mean), np.squeeze(std)


def channel_from_state(snapshots, mean, std, channel_idx=-1):
    return snapshots[..., channel_idx] * std[channel_idx] + mean[channel_idx]


def evaluate_model(model: BaseModel, experiment_path: Path, configs: ml_confs.Configs):
    train_ds, test_ds, mean, std = load_data(experiment_path, configs)

    lookback_len = model.lookback_len
    # Initial condition - last training point
    init = np.expand_dims(train_ds[-1:], 1)
    init = np.concatenate([init for _ in range(lookback_len)], axis=1)

    def obs_fn(x):
        return channel_from_state(x, mean, std)

    # Predict
    ref = channel_from_state(test_ds, mean, std)
    # MSE
    mse = []
    for t in range(len(test_ds)):
        pred = model.predict(init, t=t, observables=obs_fn)
        mse.append(np.mean((pred - ref[t]) ** 2))
    return np.sqrt(mse)


# Data
# Data
train_ds, test_ds, mean, std = load_data(experiment_path, configs)
np_train = traj_to_contexts(train_ds).astype(np.float32)
# Torch
torch_train = TrajToContextsDataset(train_ds.astype(np.float32))
torch_dl = torch.utils.data.DataLoader(
    torch_train, batch_size=len(torch_train), shuffle=True
)
# Data info
trail_dims = torch_train.contexts.shape[2:]


# Runners
def run_VAMPNets(rng_seed: int):
    from kooplearn.models import DeepEDMD

    logger.info(f"VAMPNets::Seed {rng_seed + 1}")
    fmap = _run_VAMPNets(rng_seed)
    rank = configs.layer_widths[-1]  # Full rank OLS
    model = DeepEDMD(fmap, rank=rank, reduced_rank=False)
    model.fit(np_train, verbose=False)
    return evaluate_model(model, experiment_path, configs)


def _run_VAMPNets(rng_seed: int):
    from kooplearn.models.feature_maps import VAMPNet

    trainer = lightning.Trainer(**trainer_kwargs)
    net_kwargs = {
        "feature_dim": np.prod(trail_dims),
        "layer_widths": configs.layer_widths,
    }
    # Defining the model
    vamp_fmap = VAMPNet(
        MLP,
        opt,
        {"lr": 1e-3},
        trainer,
        net_kwargs,
        lobe_timelagged=MLP,
        lobe_timelagged_kwargs=net_kwargs,
        center_covariances=False,
        seed=rng_seed,
    )
    # Init
    _, _, Vh = np.linalg.svd(
        train_ds.reshape(train_ds.shape[0], -1), full_matrices=False
    )
    Vh = Vh.astype(np.float32)
    svd_init(vamp_fmap.lightning_module.encoder, trail_dims, Vh)
    svd_init(vamp_fmap.lightning_module.encoder_timelagged, trail_dims, Vh)

    best_lr = tune_learning_rate(trainer, vamp_fmap, torch_dl)
    assert vamp_fmap.lightning_module.lr == best_lr
    vamp_fmap.fit(torch_dl)

    return vamp_fmap


def _run_DPNets(
    relaxed: bool,
    rng_seed: int,
):
    from kooplearn.models.feature_maps import DPNet

    trainer = lightning.Trainer(**trainer_kwargs)
    net_kwargs = {
        "feature_dim": np.prod(trail_dims),
        "layer_widths": configs.layer_widths,
    }
    # Defining the model
    dpnet_fmap = DPNet(
        MLP,
        opt,
        {"lr": 1e-3},
        trainer,
        use_relaxed_loss=relaxed,
        encoder_kwargs=net_kwargs,
        encoder_timelagged=MLP,
        encoder_timelagged_kwargs=net_kwargs,
        center_covariances=False,
        seed=rng_seed,
    )
    # Init
    _, _, Vh = np.linalg.svd(
        train_ds.reshape(train_ds.shape[0], -1), full_matrices=False
    )
    Vh = Vh.astype(np.float32)
    svd_init(dpnet_fmap.lightning_module.encoder, trail_dims, Vh)
    svd_init(dpnet_fmap.lightning_module.encoder_timelagged, trail_dims, Vh)

    best_lr = tune_learning_rate(trainer, dpnet_fmap, torch_dl)
    assert dpnet_fmap.lightning_module.lr == best_lr
    dpnet_fmap.fit(torch_dl)

    return dpnet_fmap


def run_DPNets(rng_seed: int):
    from kooplearn.models import DeepEDMD

    relaxed = False
    logger.info(f"DPNets::Seed {rng_seed + 1}")
    fmap = _run_DPNets(relaxed, rng_seed)
    rank = configs.layer_widths[-1]  # Full rank OLS
    model = DeepEDMD(fmap, rank=rank, reduced_rank=False)
    model.fit(np_train, verbose=False)
    return evaluate_model(model, experiment_path, configs)


def run_DPNets_relaxed(rng_seed: int):
    from kooplearn.models import DeepEDMD

    relaxed = True
    logger.info(f"DPNets::Seed {rng_seed + 1}")
    fmap = _run_DPNets(relaxed, rng_seed)
    rank = configs.layer_widths[-1]  # Full rank OLS
    model = DeepEDMD(fmap, rank=rank, reduced_rank=False)
    model.fit(np_train, verbose=False)
    return evaluate_model(model, experiment_path, configs)


def run_DynamicalAE(rng_seed: int):
    from kooplearn.models import DynamicAE

    logger.info(f"DynamicalAE::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
    trainer = lightning.Trainer(**trainer_kwargs)

    model = DynamicAE(
        MLP,
        MLP_r,
        configs.layer_widths[-1],
        opt,
        {"lr": 1e-3},
        trainer,
        encoder_kwargs={
            "feature_dim": np.prod(trail_dims),
            "layer_widths": configs.layer_widths,
        },
        decoder_kwargs={
            "feature_dim": np.prod(trail_dims),
            "layer_widths": configs.layer_widths,
            "input_dims": trail_dims,
        },
        seed=configs.rng_seed,
    )

    # Choose learning rate
    lr = tune_learning_rate(trainer, model, torch_dl)
    assert model.lightning_module.lr == lr
    model.fit(torch_dl)
    return evaluate_model(model, experiment_path, configs)


def run_ConsistentAE(rng_seed: int):
    from kooplearn.models import ConsistentAE

    logger.info(f"ConsistentAE::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
    trainer = lightning.Trainer(**trainer_kwargs)

    model = ConsistentAE(
        MLP,
        MLP_r,
        configs.layer_widths[-1],
        opt,
        {"lr": 1e-3},
        trainer,
        encoder_kwargs={
            "feature_dim": np.prod(trail_dims),
            "layer_widths": configs.layer_widths,
        },
        decoder_kwargs={
            "feature_dim": np.prod(trail_dims),
            "layer_widths": configs.layer_widths,
            "input_dims": trail_dims,
        },
        seed=configs.rng_seed,
    )

    # Data for this
    cae_train = TrajToContextsDataset(train_ds.astype(np.float32), context_window_len=3)
    cae_dl = torch.utils.data.DataLoader(
        cae_train, batch_size=len(cae_train), shuffle=True
    )

    # Choose learning rate
    lr = tune_learning_rate(trainer, model, cae_dl)
    assert model.lightning_module.lr == lr
    model.fit(cae_dl)
    return evaluate_model(model, experiment_path, configs)


AVAIL_MODELS = {
    "VAMPNets": run_VAMPNets,
    "DPNets": run_DPNets,
    "DPNets-relaxed": run_DPNets_relaxed,
    "DynamicalAE": run_DynamicalAE,
    "ConsistentAE": run_ConsistentAE,
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
            results = AVAIL_MODELS[args.model](args.rngseed)
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
