import logging
import os
from pathlib import Path

import lightning
import ml_confs
import schnetpack
import torch
from input_pipeline import TimeLaggedSampler
from model import GraphDPNet


def get_dataset(db_path, cutoff):
    data_path = db_path.parent
    db_name = db_path.name.split(".")[0]
    cache_path = str(data_path / f"__cache-{db_name}__")
    nb_list_transform = schnetpack.transform.CachedNeighborList(
        cache_path,
        schnetpack.transform.MatScipyNeighborList(cutoff=cutoff),
        keep_cache=True,
    )
    in_transforms = [schnetpack.transform.CastTo32(), nb_list_transform]
    return schnetpack.data.ASEAtomsData(str(db_path), transforms=in_transforms)


def main():
    main_path = Path(__file__).parent
    configs = ml_confs.from_file(main_path / "configs.yaml")
    # Seed everything
    lightning.pytorch.seed_everything(configs.seed)
    # Loading the database
    db_name = os.path.split(configs.topology_path)[-1].split(".")[0]
    slug = configs.selection.replace(" ", "_")
    db_name = f"{db_name}_{slug}"
    db_path = main_path / "data" / f"{db_name}.db"

    # Loading the dataset
    dataset = get_dataset(db_path, configs.cutoff)
    batch_sampler = TimeLaggedSampler(
        dataset, batch_size=configs.batch_size, lagtime=configs.lagtime, shuffle=True
    )
    dataloader = schnetpack.data.AtomsLoader(
        dataset, batch_sampler=batch_sampler, num_workers=20, persistent_workers=True
    )

    n_atoms = dataset[0][schnetpack.properties.n_atoms].item()
    model = GraphDPNet(
        configs,
        n_atoms,
        torch.optim.Adam,
        use_relaxed_loss=configs.use_relaxed_loss,
        metric_deformation_loss_coefficient=configs.metric_loss,
        optimizer_kwargs={"lr": 1e-2},
    )

    train_logger = lightning.pytorch.loggers.WandbLogger(
        project="GraphDPNet-chignolin", entity="csml"
    )

    lr_finder_cb = lightning.pytorch.callbacks.LearningRateFinder(max_lr=1e-3)

    ckpt_path = db_path.parent.parent / "ckpt" / str(train_logger.experiment.name)
    ckpt_cb = lightning.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_path)

    trainer = lightning.Trainer(
        accelerator="gpu",
        logger=train_logger,
        max_epochs=configs.max_epochs,
        callbacks=[ckpt_cb],
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
