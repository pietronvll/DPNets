import functools
from time import perf_counter
from typing import Union

import lightning
import torch
from kooplearn.abc import BaseModel, TrainableFeatureMap


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
