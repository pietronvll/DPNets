import logging
from copy import deepcopy

import ml_confs
import schnetpack
import torch
from kooplearn.nn.functional import (
    log_fro_metric_deformation_loss,
    relaxed_projection_score,
    vamp_score,
)
from lightning import LightningModule


class DeepGraphKernel(LightningModule):
    def __init__(
        self,
        configs: ml_confs.Configs,
        n_atoms: int,
        optimizer: torch.optim.Optimizer,
        use_relaxed_loss: bool = False,
        metric_deformation_loss_coefficient: float = 1.0,  # That is, the parameter γ in the paper.
        optimizer_kwargs={},
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=["encoder", "optimizer_fn", "kooplearn_feature_map_weakref"]
        )
        _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
        if "lr" in _tmp_opt_kwargs:  # For Lightning's LearningRateFinder
            self.lr = _tmp_opt_kwargs.pop("lr")
            self.opt_kwargs = _tmp_opt_kwargs
        else:
            self.lr = 1e-3
            self.opt_kwargs = {}
            logging.warning(
                "No learning rate specified. Using default value of 1e-3. You can specify the learning rate by passing it to the optimizer_kwargs argument."
            )

        self.schnet = SchNet(configs)
        self.n_atoms = n_atoms
        self.configs = configs
        self.optimizer = optimizer

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.schnet(inputs)

    def training_step(self, batch, batch_idx):
        features = self.schnet(batch)["scalar_representation"]
        features = features.reshape(-1, self.n_atoms, self.configs.n_final_features)
        features = features.mean(dim=1)  # Linear kernel mean embedding
        encoded_X = features[::2]
        encoded_Y = features[1::2]

        _norm = torch.rsqrt(torch.tensor(encoded_X.shape[0]))
        encoded_X *= _norm
        encoded_Y *= _norm

        cov_X = torch.mm(encoded_X.T, encoded_X)
        cov_Y = torch.mm(encoded_Y.T, encoded_Y)
        cov_XY = torch.mm(encoded_X.T, encoded_Y)

        metrics = {}
        # Compute the losses
        if self.hparams.use_relaxed_loss:
            svd_loss = -1 * relaxed_projection_score(cov_X, cov_Y, cov_XY)
            metrics["train/relaxed_projection_score"] = -1.0 * svd_loss.item()
        else:
            svd_loss = -1 * vamp_score(cov_X, cov_Y, cov_XY, schatten_norm=2)
            metrics["train/projection_score"] = -1.0 * svd_loss.item()
        if self.hparams.metric_deformation_loss_coefficient > 0.0:
            metric_deformation_loss = 0.5 * (
                log_fro_metric_deformation_loss(cov_X)
                + log_fro_metric_deformation_loss(cov_Y)
            )
            metric_deformation_loss *= self.hparams.metric_deformation_loss_coefficient
            metrics["train/metric_deformation_loss"] = metric_deformation_loss.item()
            svd_loss += metric_deformation_loss
        metrics["train/total_loss"] = svd_loss.item()
        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True)
        return svd_loss

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self._optimizer(self.parameters(), **kw)


class SchNet(schnetpack.model.AtomisticModel):
    def __init__(self, configs: ml_confs.Configs):
        super().__init__(
            input_dtype_str="float32",
            postprocessors=None,
            do_postprocessing=False,
        )
        self.cutoff = configs.cutoff
        self.pwise_dist = schnetpack.atomistic.PairwiseDistances()
        self.radial_basis = schnetpack.nn.GaussianRBF(
            n_rbf=configs.n_rbf, cutoff=self.cutoff
        )
        self.net = schnetpack.representation.SchNet(
            n_atom_basis=configs.n_atom_basis,
            n_interactions=configs.n_interactions,
            radial_basis=self.radial_basis,
            cutoff_fn=schnetpack.nn.CosineCutoff(self.cutoff),
        )
        self.final_lin = torch.nn.Linear(configs.n_atom_basis, configs.n_final_features)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = self.pwise_dist(inputs)
        inputs = self.net(inputs)
        inputs["scalar_representation"] = self.final_lin(
            inputs["scalar_representation"]
        )
        return inputs
