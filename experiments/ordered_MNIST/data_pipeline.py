import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

import lightning
import ml_confs as mlcfg
import numpy as np
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset, load_from_disk
from kooplearn.abc import TrainableFeatureMap
from torch.utils.data import DataLoader

main_path = Path(__file__).parent
data_path = main_path / "data"
ckpt_path = main_path / "ckpt"
configs = mlcfg.from_file(main_path / "configs.yaml")


def make_dataset():
    # Data pipeline
    MNIST = load_dataset("mnist", keep_in_memory=True)
    digit_ds = []
    for i in range(configs.classes):
        digit_ds.append(
            MNIST.filter(
                lambda example: example["label"] == i, keep_in_memory=True, num_proc=8
            )
        )
    ordered_MNIST = DatasetDict()
    # Order the digits in the dataset and select only a subset of the data
    for split in ["train", "test"]:
        ordered_MNIST[split] = interleave_datasets(
            [ds[split] for ds in digit_ds], split=split
        ).select(range(configs[f"{split}_samples"]))
    _tmp_ds = ordered_MNIST["train"].train_test_split(
        test_size=configs.val_ratio, shuffle=False
    )
    ordered_MNIST["train"] = _tmp_ds["train"]
    ordered_MNIST["validation"] = _tmp_ds["test"]
    ordered_MNIST.set_format(type="torch", columns=["image", "label"])
    ordered_MNIST = ordered_MNIST.map(
        lambda example: {"image": example["image"] / 255.0, "label": example["label"]},
        batched=True,
        keep_in_memory=True,
        num_proc=2,
    )
    ordered_MNIST.save_to_disk(data_path)
    configs.to_file(data_path / "configs.yaml")


# CNN Architecture
class CNNEncoder(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNEncoder, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        # fully connected layer, output num_classes classes
        self.out = torch.nn.Sequential(torch.nn.Linear(32 * 7 * 7, num_classes))
        torch.torch.nn.init.orthogonal_(self.out[0].weight)

    def forward(self, X):
        if X.dim() == 3:
            X = X.unsqueeze(1)  # Add a channel dimension if needed
        X = self.conv1(X)
        X = self.conv2(X)
        # Flatten the output of conv2
        X = X.view(X.size(0), -1)
        output = self.out(X)
        return output


# A decoder which is specular to CNNEncoder, starting with a fully connected layer and then reshaping the output to a 2D image
class CNNDecoder(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNNDecoder, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(num_classes, 32 * 7 * 7))

        self.conv1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, 5, 1, 2),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        # Remove the channel dimension
        x = x.squeeze(1)
        return x


class ClassifierModule(lightning.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.encoder = CNNEncoder(num_classes=num_classes)
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Set up storage for metrics
        self.train_acc = []
        self.train_steps = []
        self.val_acc = []
        self.val_steps = []

    def on_fit_start(self):
        self.train_acc = []
        self.train_steps = []
        self.val_acc = []
        self.val_steps = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.encoder(images)
        loss = self.loss_fn(output, labels)
        with torch.no_grad():
            pred_labels = output.argmax(dim=1)
            accuracy = (pred_labels == labels).float().mean()

        # Log metrics
        self.train_acc.append(accuracy.item())
        self.train_steps.append(self.global_step)

        return {"loss": loss, "train/accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.encoder(images)
        pred_labels = output.argmax(dim=1)
        accuracy = (pred_labels == labels).float().mean()  # Scalar

        self.val_acc.append(accuracy.item())
        self.val_steps.append(self.global_step)

        return {"val/accuracy": accuracy}


class ClassifierFeatureMap(TrainableFeatureMap):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        trainer: lightning.Trainer,
        seed: Optional[int] = None,
    ):
        # Set rng seed
        lightning.seed_everything(seed)
        self.seed = seed
        self.lightning_module = ClassifierModule(num_classes, learning_rate)

        # Init trainer
        self.lightning_trainer = trainer
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def lookback_len(self) -> int:
        return 1  # Hardcoding it here, as we are not using lookback windows

    # Not tested
    def save(self, path: os.PathLike):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save the trainer
        torch.save(self.lightning_trainer, path / "lightning_trainer.bin")
        # Save the lightning checkpoint
        ckpt = path / "lightning.ckpt"
        self.lightning_trainer.save_checkpoint(str(ckpt))
        del self.lightning_module
        del self.lightning_trainer
        model = path / "kooplearn_model.pkl"
        with open(model, "wb") as f:
            pickle.dump(self, f)

    # Not tested
    @classmethod
    def load(cls, path: os.PathLike):
        path = Path(path)
        trainer = torch.load(path / "lightning_trainer.bin")
        ckpt = path / "lightning.ckpt"
        with open(path / "kooplearn_model.pkl", "rb") as f:
            restored_obj = pickle.load(f)
        assert isinstance(restored_obj, cls)
        restored_obj.lightning_trainer = trainer
        restored_obj.lightning_module = ClassifierModule.load_from_checkpoint(str(ckpt))
        return restored_obj

    def fit(self, **trainer_fit_kwargs: dict):
        if "model" in trainer_fit_kwargs:
            logging.warn(
                "The 'model' keyword should not be specified in trainer_fit_kwargs. The model is automatically set to the DPNet feature map, and the provided model is ignored."
            )
            trainer_fit_kwargs = trainer_fit_kwargs.copy()
            del trainer_fit_kwargs["model"]
        self.lightning_trainer.fit(model=self.lightning_module, **trainer_fit_kwargs)
        self._is_fitted = True

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(X).float()
        X.to(self.lightning_module.device)
        self.lightning_module.eval()
        with torch.no_grad():
            embedded_X = self.lightning_module.encoder(X)
            embedded_X = embedded_X.detach().cpu().numpy()
        return embedded_X


def train_oracle():
    ordered_MNIST = load_from_disk(str(data_path))
    train_dl = DataLoader(
        ordered_MNIST["train"], batch_size=configs.batch_size, shuffle=True
    )
    val_dl = DataLoader(
        ordered_MNIST["validation"],
        batch_size=len(ordered_MNIST["validation"]),
        shuffle=False,
    )

    trainer_kwargs = {
        "accelerator": "gpu",
        "max_epochs": 20,
        "log_every_n_steps": 2,
        "enable_progress_bar": False,
        "devices": 1,
        "enable_checkpointing": False,
        "logger": False,
    }

    trainer = lightning.Trainer(**trainer_kwargs)

    oracle = ClassifierFeatureMap(
        configs.classes, 1e-2, trainer, seed=0  # Reproducibility
    )

    import warnings

    warnings.filterwarnings(
        "ignore", ".*does not have many workers.*"
    )  # Ignore warnings about num_workers
    oracle.fit(train_dataloaders=train_dl, val_dataloaders=val_dl)
    oracle.save(ckpt_path / "oracle")


if __name__ == "__main__":
    # Check if data_path exists, if not preprocess the data
    if not data_path.exists():
        print("Data directory not found, preprocessing data.")
        make_dataset()
        train_oracle()
    else:
        # Try to load the configs.yaml file and compare with the current one, if different, wipe the data_path and preprocess the data
        _saved_configs = mlcfg.from_file(data_path / "configs.yaml")
        if _saved_configs != configs:
            print("Configs changed, preprocessing data.")
            # Delete the data_path and preprocess the data
            shutil.rmtree(data_path)
            make_dataset()
            shutil.rmtree(ckpt_path / "oracle")
            train_oracle()
        else:
            print("Data already preprocessed.")
