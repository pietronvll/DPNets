import argparse
import logging
import pickle
import sys
from functools import partial
from pathlib import Path

import lightning
import ml_confs
import numpy as np
import torch
from kooplearn.data import traj_to_contexts
from kooplearn.models import Kernel
from kooplearn.nn.data import collate_context_dataset
from sklearn.metrics.pairwise import polynomial_kernel

main_path = Path(__file__).parent.parent.parent
sys.path.append(str(main_path))
from experiments.utils import sanitize_filename, tune_learning_rate

# Paths and configs
experiment_path = Path(__file__).parent
data_path = experiment_path / "data"
ckpt_path = experiment_path / "ckpt"
results_path = experiment_path / "results"
configs = ml_confs.from_file(experiment_path / "configs.yaml")

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

device = "cpu"
# Trainer Configuration
trainer_kwargs = {
    "accelerator": device,
    "devices": 1,
    "max_epochs": configs.max_epochs,
    "enable_progress_bar": True,
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


def evaluate_model(model):
    true = np_test.lookforward(1)
    true_C = channel_from_state(true, mean, std)

    pred = np.concatenate(
        [model.predict(np_test[0], t=t) for t in range(1, len(np_test) + 1)], axis=0
    )
    pred_C = channel_from_state(pred, mean, std)

    rMSE_error = np.sqrt(np.mean((true_C - pred_C) ** 2, axis=(2, 3)))[:, 0]
    return rMSE_error


# Data
train_ds, test_ds, mean, std = load_data(experiment_path, configs)
np_train = traj_to_contexts(
    train_ds,
)
np_test = traj_to_contexts(
    test_ds,
)
# Torch
train_ctxs = traj_to_contexts(
    train_ds, backend="torch", device=device, dtype=torch.float32
)
torch_dl = torch.utils.data.DataLoader(
    train_ctxs,
    batch_size=len(train_ctxs),
    shuffle=True,
    collate_fn=collate_context_dataset,
)
# Data info
trail_dims = train_ctxs.data.shape[2:]


# Runners
def _run_NNFeatureMap(rng_seed: int, loss_fn, loss_kwargs):
    from kooplearn.models import Nonlinear
    from kooplearn.models.feature_maps import NNFeatureMap

    logger.info(f"{loss_fn.__class__.__name__}::Seed {rng_seed + 1}")
    trainer = lightning.Trainer(**trainer_kwargs)
    net_kwargs = {
        "feature_dim": np.prod(trail_dims),
        "layer_widths": configs.layer_widths,
    }
    # Defining the model
    fmap = NNFeatureMap(
        MLP,
        loss_fn,
        opt,
        trainer,
        lagged_encoder=MLP,
        encoder_kwargs=net_kwargs,
        loss_kwargs=loss_kwargs,
        lagged_encoder_kwargs=net_kwargs,
        seed=rng_seed,
    )
    # Init
    _, _, Vh = np.linalg.svd(
        train_ds.reshape(train_ds.shape[0], -1), full_matrices=False
    )
    Vh = Vh.astype(np.float32)
    svd_init(fmap.lightning_module.encoder, trail_dims, Vh)
    svd_init(fmap.lightning_module.lagged_encoder, trail_dims, Vh)

    best_lr = tune_learning_rate(trainer, fmap, torch_dl)
    assert fmap.lightning_module.lr == best_lr
    fmap.fit(torch_dl)

    rank = configs.layer_widths[-1]  # Full rank OLS
    model = Nonlinear(fmap, rank=rank, reduced_rank=False)
    try:
        model.fit(
            np_train,
        )
        return evaluate_model(model)
    except Exception as e:
        logger.warn(e)
        return None


def run_VAMPNets(rng_seed: int):
    from kooplearn.nn import VAMPLoss

    return _run_NNFeatureMap(rng_seed, VAMPLoss, {"center_covariances": False})


def run_DPNets(rng_seed: int):
    from kooplearn.nn import DPLoss

    return _run_NNFeatureMap(
        rng_seed, DPLoss, {"center_covariances": False, "relaxed": False}
    )


def run_DPNets_relaxed(rng_seed: int):
    from kooplearn.nn import DPLoss

    return _run_NNFeatureMap(
        rng_seed, DPLoss, {"center_covariances": False, "relaxed": True}
    )


def _run_Kernel(kernel_fn):
    model = Kernel(kernel_fn, reduced_rank=False, rank=32, tikhonov_reg=1e-6)
    model.fit(np_train)
    return evaluate_model(model)


def run_Poly1(rng_seed):
    kernel_fn = partial(polynomial_kernel, degree=1)
    return _run_Kernel(kernel_fn)


def run_Poly3(rng_seed):
    kernel_fn = polynomial_kernel
    return _run_Kernel(kernel_fn)


def run_DynamicalAE(rng_seed: int):
    from kooplearn.models import DynamicAE

    logger.info(f"DynamicalAE::Seed {rng_seed}")
    trainer = lightning.Trainer(**trainer_kwargs)

    model = DynamicAE(
        MLP,
        MLP_r,
        configs.layer_widths[-1],
        opt,
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
        optimizer_kwargs={"lr": 1e-3},
        seed=rng_seed,
    )

    # Choose learning rate
    lr = tune_learning_rate(trainer, model, torch_dl)
    assert model.lightning_module.lr == lr
    model.fit(torch_dl)
    return evaluate_model(model)


def run_ConsistentAE(rng_seed: int):
    from kooplearn.models import ConsistentAE

    logger.info(f"ConsistentAE::Seed {rng_seed}")
    trainer = lightning.Trainer(**trainer_kwargs)

    model = ConsistentAE(
        MLP,
        MLP_r,
        configs.layer_widths[-1],
        opt,
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
        optimizer_kwargs={"lr": 1e-3},
        seed=rng_seed,
    )

    # Data for this
    # Torch
    train_ctxs = traj_to_contexts(
        train_ds,
        context_window_len=3,
        backend="torch",
        device=device,
        dtype=torch.float32,
    )
    cae_dl = torch.utils.data.DataLoader(
        train_ctxs, batch_size=64, shuffle=True, collate_fn=collate_context_dataset
    )

    # Choose learning rate
    lr = tune_learning_rate(trainer, model, cae_dl)
    assert model.lightning_module.lr == lr
    model.fit(cae_dl)
    return evaluate_model(model)


AVAIL_MODELS = {
    "VAMPNets": run_VAMPNets,
    "DPNets": run_DPNets,
    "DPNets-relaxed": run_DPNets_relaxed,
    "DynamicalAE": run_DynamicalAE,
    "ConsistentAE": run_ConsistentAE,
    "Poly3": run_Poly3,
    "Poly1": run_Poly1,
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
