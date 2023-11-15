import argparse
import functools
import logging
import pickle
from pathlib import Path
from time import perf_counter
from typing import Union

import datasets
import lightning
import ml_confs
import numpy as np
import torch
from data_pipeline import ClassifierFeatureMap, ClassifierModule, CNNDecoder, CNNEncoder
from datasets import load_from_disk
from kooplearn.abc import BaseModel, TrainableFeatureMap
from kooplearn.data import traj_to_contexts
from kooplearn.nn.data import TrajToContextsDataset
from torch.utils.data import DataLoader

# Paths and configs
experiment_path = Path(__file__).parent
data_path = experiment_path / "data"
ckpt_path = experiment_path / "ckpt"
results_path = experiment_path / "results"
configs = ml_confs.from_file(
    experiment_path / "configs.yaml", register_jax_pytree=False
)

# Logging
logger = logging.getLogger("ordered_MNIST")
logger.setLevel(logging.INFO)
log_file = experiment_path / "logs/run.log"  # Specify the path to your log file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Trainer Configuration
trainer_kwargs = {
    "accelerator": "mps",
    "devices": 1,
    "max_epochs": configs.max_epochs,
    "enable_progress_bar": True,
    "enable_checkpointing": False,
    "logger": False,
}


# Utility functions
def stack_reports(reports):
    # Save the 'image' and 'label' keys of the report with the best accuracy, and save an average and std of the rest.
    stacked = {}
    stacked["accuracy"] = np.mean([r["accuracy"] for r in reports], axis=0)
    stacked["accuracy_std"] = np.std([r["accuracy"] for r in reports], axis=0)
    stacked["fit_time"] = np.mean([r["fit_time"] for r in reports])
    stacked["fit_time_std"] = np.std([r["fit_time"] for r in reports])
    # If the model is a neural network, save the time per epoch
    if "time_per_epoch" in reports[0]:
        stacked["time_per_epoch"] = np.mean([r["time_per_epoch"] for r in reports])
        stacked["time_per_epoch_std"] = np.std([r["time_per_epoch"] for r in reports])
    best_accuracy_idx = np.argmax([np.mean(r["accuracy"]) for r in reports])
    stacked["image"] = reports[best_accuracy_idx]["image"]
    stacked["label"] = reports[best_accuracy_idx]["label"]
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


def evaluate_model(
    model: BaseModel, oracle: TrainableFeatureMap, test_data: datasets.Dataset
):
    from kooplearn.models.ae.consistent import (
        ConsistentAE,  # Need to import the class from the true path, otherwise isinstance will fail (?!)
    )

    cap_saved_imgs = 10  # Deflate report size
    assert model.is_fitted
    test_labels = test_data["label"]
    test_images = test_data["image"]
    test_images = np.expand_dims(test_images, 1)  # Add context dimension
    if isinstance(model, ConsistentAE):
        # Pad with zeros
        padding = np.zeros_like(test_images)
        test_images = np.concatenate(
            [padding] * (model.lookback_len - 1) + [test_images], axis=1
        )
    report = {"accuracy": [], "label": [], "image": []}
    for t in range(1, configs.eval_up_to_t + 1):
        pred = model.predict(test_images, t=t)
        pred_labels = oracle(pred).argmax(axis=1)
        accuracy = (pred_labels == (test_labels + t) % configs.classes).mean()
        report["accuracy"].append(accuracy)
        report["image"].append(pred[:cap_saved_imgs])
        report["label"].append(pred_labels[:cap_saved_imgs])
    return report


def load_oracle():
    oracle = ClassifierFeatureMap.load(ckpt_path / "oracle")
    return oracle


def load_data(torch: bool = False, ctx_len: int = 2):
    ordered_MNIST = load_from_disk(str(data_path))
    # Creating a copy of the dataset in numpy format
    np_ordered_MNIST = ordered_MNIST.with_format(
        type="numpy", columns=["image", "label"]
    )
    if torch:
        train_ds = TrajToContextsDataset(
            ordered_MNIST["train"]["image"], context_window_len=ctx_len
        )
        val_ds = TrajToContextsDataset(
            ordered_MNIST["validation"]["image"], context_window_len=ctx_len
        )
        # Dataloaders
        train_data = DataLoader(train_ds, batch_size=configs.batch_size, shuffle=True)
        val_data = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    else:
        train_data = traj_to_contexts(
            np_ordered_MNIST["train"]["image"], context_window_len=ctx_len
        )
        train_data = train_data.copy()  # Avoid numpy to torch read-only errors
        val_data = np_ordered_MNIST["validation"]
    return train_data, val_data, np_ordered_MNIST["test"]


def tune_learning_rate(
    trainer: lightning.Trainer,
    model: Union[BaseModel, TrainableFeatureMap],
    train_dataloader: DataLoader,
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


# Model Runners
def run_OracleFeatures():
    from kooplearn.models import DeepEDMD

    logger.info("OracleFeatures::START")
    oracle = load_oracle()
    train_data, _, test_data = load_data()
    classifier_model = DeepEDMD(oracle, reduced_rank=False, rank=configs.classes)
    _, fit_time = timer(classifier_model.fit)(train_data)
    report = evaluate_model(classifier_model, oracle, test_data)
    report["fit_time"] = fit_time
    logger.info("OracleFeatures::END")
    return report


def run_DMD():
    from kooplearn.models import DMD

    logger.info("DMD::START")
    train_data, _, test_data = load_data()
    dmd_model = DMD(reduced_rank=configs.reduced_rank, rank=configs.classes)
    _, fit_time = timer(dmd_model.fit)(train_data)
    oracle = load_oracle()
    report = evaluate_model(dmd_model, oracle, test_data)
    report["fit_time"] = fit_time
    logger.info("DMD::END")
    return report


def _run_kDMD(kernel):
    from kooplearn.models import KernelDMD

    train_data, _, test_data = load_data()
    kernel_model = KernelDMD(
        kernel=kernel,
        reduced_rank=configs.reduced_rank,
        rank=configs.classes,
        svd_solver="arnoldi",
    )
    _, fit_time = timer(kernel_model.fit)(train_data)
    oracle = load_oracle()
    report = evaluate_model(kernel_model, oracle, test_data)
    report["fit_time"] = fit_time
    return report


def _tune_kDMD_lengthscale(kernel_class, **kw):
    import optuna
    from kooplearn.models import KernelDMD
    from scipy.spatial.distance import pdist

    oracle = load_oracle()
    train_data, val_data, _ = load_data()
    flattened_train_data = (train_data[:, 0, ...]).reshape(train_data.shape[0], -1)
    median_heuristic = np.median(pdist(flattened_train_data))

    # Length-scale optimization with Optuna
    def objective(trial):
        ls = trial.suggest_float("ls", 0.1 * median_heuristic, 10 * median_heuristic)
        kernel = kernel_class(length_scale=ls, **kw)
        model = KernelDMD(
            kernel=kernel,
            reduced_rank=configs.reduced_rank,
            rank=configs.classes,
            svd_solver="arnoldi",
        ).fit(train_data)
        report = evaluate_model(model, oracle, val_data)
        return np.mean(report["accuracy"])

    sampler = optuna.samplers.TPESampler(seed=0)  # Reproductibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=configs.trial_budget)
    return study.best_params["ls"]


def run_RBF_DMD():
    from sklearn.gaussian_process.kernels import RBF

    logger.info("RBF_DMD::START")
    logger.info("RBF_DMD::Tuning length-scale")
    ls = _tune_kDMD_lengthscale(RBF)
    logger.info(f"RBF_DMD::Tuned length-scale: {ls}")
    report = _run_kDMD(RBF(length_scale=ls))
    logger.info("RBF_DMD::END")
    return report


def run_Poly3_DMD():
    # Poly3 Kernel DMD
    from sklearn.metrics.pairwise import polynomial_kernel

    logger.info("Poly3_DMD::START")
    report = _run_kDMD(polynomial_kernel)
    logger.info("Poly3_DMD::END")
    return report


def run_AbsExp_DMD():
    # Absolute Exponential a.k.a. Matern 0.5 Kernel DMD
    from sklearn.gaussian_process.kernels import Matern

    logger.info("AbsExp_DMD::START")
    logger.info("AbsExp_DMD::Tuning length-scale")
    ls = _tune_kDMD_lengthscale(Matern, nu=0.5)
    logger.info(f"AbsExp_DMD::Tuned length-scale: {ls}")
    report = _run_kDMD(Matern(length_scale=ls, nu=0.5))
    logger.info("AbsExp_DMD::END")
    return report


def run_VAMPNets():
    from kooplearn.models import DeepEDMD
    from kooplearn.models.feature_maps import VAMPNet

    logger.info("VAMPNets::START")
    oracle = load_oracle()
    train_dl, _, _ = load_data(torch=True)
    results = []
    for rng_seed in range(configs.num_rng_seeds):  # Reproducibility
        logger.info(f"VAMPNets::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
        trainer = lightning.Trainer(**trainer_kwargs)
        # Defining the model
        feature_map = VAMPNet(
            CNNEncoder,
            torch.optim.Adam,
            trainer,
            encoder_kwargs={"num_classes": configs.classes},
            optimizer_kwargs={"lr": 1e-4},
            center_covariances=False,
            seed=rng_seed,
        )
        best_lr = tune_learning_rate(trainer, feature_map, train_dl)
        assert feature_map.lightning_module.lr == best_lr
        _, nn_fit_time = timer(feature_map.fit)(train_dl)
        time_per_epoch = nn_fit_time / trainer.max_epochs
        train_data, _, test_data = load_data()
        VAMPNet_model = DeepEDMD(
            feature_map, reduced_rank=configs.reduced_rank, rank=configs.classes
        )
        _, fit_time = timer(VAMPNet_model.fit)(train_data)
        report = evaluate_model(VAMPNet_model, oracle, test_data)
        report["fit_time"] = fit_time
        report["time_per_epoch"] = time_per_epoch
        results.append(report)
    report = stack_reports(results)
    logger.info("VAMPNets::END")
    return report


def _run_DPNets(
    relaxed: bool, metric_deformation_coeff: float = 1.0, rng_seed: int = 0
):
    from kooplearn.models import DeepEDMD
    from kooplearn.models.feature_maps import DPNet

    train_dl, _, _ = load_data(torch=True)
    trainer = lightning.Trainer(**trainer_kwargs)
    # Defining the model
    feature_map = DPNet(
        CNNEncoder,
        torch.optim.Adam,
        trainer,
        use_relaxed_loss=relaxed,
        metric_deformation_loss_coefficient=metric_deformation_coeff,
        encoder_kwargs={"num_classes": configs.classes},
        optimizer_kwargs={"lr": 1e-4},
        center_covariances=False,
        seed=rng_seed,
    )
    best_lr = tune_learning_rate(trainer, feature_map, train_dl)
    assert feature_map.lightning_module.lr == best_lr
    _, nn_fit_time = timer(feature_map.fit)(train_dl)
    time_per_epoch = nn_fit_time / trainer.max_epochs
    train_data, _, _ = load_data()
    DPNet_model = DeepEDMD(
        feature_map, reduced_rank=configs.reduced_rank, rank=configs.classes
    )
    _, fit_time = timer(DPNet_model.fit)(train_data)
    return DPNet_model, (fit_time, time_per_epoch)


def _tune_DPNets_metric_deformation(relaxed: bool, rng_seed: int = 0):
    import optuna

    oracle = load_oracle()
    _, val_data, _ = load_data()

    # HP Opt with Optuna
    def objective(trial):
        metric_deformation = trial.suggest_float(
            "metric_deformation", 1e-2, 1, log=True
        )
        model, _ = _run_DPNets(relaxed, metric_deformation, rng_seed)
        report = evaluate_model(model, oracle, val_data)
        return np.mean(report["accuracy"])

    sampler = optuna.samplers.TPESampler(seed=0)  # Reproductibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=configs.trial_budget)
    return study.best_params["metric_deformation"]


def run_DPNets():
    logger.info("DPNets::START")
    _, _, test_data = load_data()
    oracle = load_oracle()
    results = []
    relaxed = False
    for rng_seed in range(configs.num_rng_seeds):  # Reproducibility
        logger.info(f"DPNets::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
        metric_deformation = _tune_DPNets_metric_deformation(relaxed, rng_seed)
        logger.info(f"DPNets::Tuned metric deformation: {metric_deformation}")
        model, (fit_time, time_per_epoch) = _run_DPNets(
            relaxed, metric_deformation, rng_seed
        )
        report = evaluate_model(model, oracle, test_data)
        report["fit_time"] = fit_time
        report["time_per_epoch"] = time_per_epoch
        results.append(report)
    report = stack_reports(results)
    logger.info("DPNets::END")
    return report


def run_DPNets_relaxed():
    logger.info("DPNets-relaxed::START")
    _, _, test_data = load_data()
    oracle = load_oracle()
    results = []
    relaxed = True
    for rng_seed in range(configs.num_rng_seeds):  # Reproducibility
        logger.info(f"DPNets-relaxed::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
        metric_deformation = _tune_DPNets_metric_deformation(relaxed, rng_seed)
        logger.info(f"DPNets-relaxed::Tuned metric deformation: {metric_deformation}")
        model, (fit_time, time_per_epoch) = _run_DPNets(
            relaxed, metric_deformation, rng_seed
        )
        report = evaluate_model(model, oracle, test_data)
        report["fit_time"] = fit_time
        report["time_per_epoch"] = time_per_epoch
        results.append(report)
    report = stack_reports(results)
    logger.info("DPNets-relaxed::END")
    return report


def run_DynamicalAE():
    from kooplearn.models import DynamicAE

    logger.info("DynamicalAE::START")
    results = []
    for rng_seed in range(configs.num_rng_seeds):  # Reproducibility
        logger.info(f"DynamicalAE::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
        trainer = lightning.Trainer(**trainer_kwargs)
        train_dl, _, test_data = load_data(torch=True)
        dae_model = DynamicAE(
            CNNEncoder,
            CNNDecoder,
            configs.classes,
            torch.optim.Adam,
            trainer,
            encoder_kwargs={"num_classes": configs.classes},
            decoder_kwargs={"num_classes": configs.classes},
            optimizer_kwargs={"lr": 1e-4},
            seed=rng_seed,
        )
        best_lr = tune_learning_rate(trainer, dae_model, train_dl)
        assert dae_model.lightning_module.lr == best_lr
        _, fit_time = timer(dae_model.fit)(train_dl)
        time_per_epoch = fit_time / trainer.max_epochs
        oracle = load_oracle()
        report = evaluate_model(dae_model, oracle, test_data)
        report["fit_time"] = fit_time
        report["time_per_epoch"] = time_per_epoch
        results.append(report)
    report = stack_reports(results)
    logger.info("DynamicalAE::END")
    return report


def run_ConsistentAE():
    from kooplearn.models import ConsistentAE

    logger.info("ConsistentAE::START")
    results = []
    for rng_seed in range(configs.num_rng_seeds):  # Reproducibility
        logger.info(f"ConsistentAE::Seed {rng_seed + 1}/{configs.num_rng_seeds}")
        trainer = lightning.Trainer(**trainer_kwargs)
        train_dl, _, test_data = load_data(torch=True, ctx_len=3)
        dae_model = ConsistentAE(
            CNNEncoder,
            CNNDecoder,
            configs.classes,
            torch.optim.Adam,
            trainer,
            encoder_kwargs={"num_classes": configs.classes},
            decoder_kwargs={"num_classes": configs.classes},
            optimizer_kwargs={"lr": 1e-4},
            seed=rng_seed,
        )
        best_lr = tune_learning_rate(trainer, dae_model, train_dl)
        assert dae_model.lightning_module.lr == best_lr
        _, fit_time = timer(dae_model.fit)(train_dl)
        time_per_epoch = fit_time / trainer.max_epochs
        oracle = load_oracle()
        report = evaluate_model(dae_model, oracle, test_data)
        report["fit_time"] = fit_time
        report["time_per_epoch"] = time_per_epoch
        results.append(report)
    report = stack_reports(results)
    logger.info("ConsistentAE::END")
    return report


########### MAIN  ###########

AVAIL_MODELS = {
    "Oracle-Features": run_OracleFeatures,
    "DMD": run_DMD,
    "KernelDMD-RBF": run_RBF_DMD,
    "KernelDMD-Poly3": run_Poly3_DMD,
    "KernelDMD-AbsExp": run_AbsExp_DMD,
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
    parser.add_argument(
        "--model", choices=AVAIL_MODELS.keys(), help="Specify the model to run."
    )

    args = parser.parse_args()

    if args.model:
        if args.model in AVAIL_MODELS:
            results = AVAIL_MODELS[args.model]()
            results["name"] = args.model
            fname = sanitize_filename(args.model) + "_results.pkl"
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
