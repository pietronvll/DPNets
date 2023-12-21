import logging
from copy import deepcopy

import ml_confs
import schnetpack
import torch
from kooplearn.nn.functional import (  # relaxed_projection_score,
    log_fro_metric_deformation_loss,
    vamp_score,
)
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm


def relaxed_projection_score(cov_x, cov_y, cov_xy):
    return (
        2 * torch.linalg.matrix_norm(cov_xy, ord="fro").log()
        - torch.linalg.matrix_norm(cov_x, ord=2).log()
        - torch.linalg.matrix_norm(cov_y, ord=2).log()
    )


def EY_CCA_loss(cov_x, cov_y, cov_xy, cov_x_prime, cov_y_prime, cov_xy_prime):
    symm_xcov = cov_xy + cov_xy.T
    symm_xcov_prime = cov_xy_prime + cov_xy_prime.T
    in_cov = cov_x + cov_y
    in_cov_prime = cov_x_prime + cov_y_prime
    return -torch.trace(symm_xcov + symm_xcov_prime) + torch.trace(
        torch.mm(in_cov_prime, in_cov)
    )


def compute_covs(encoded_X, encoded_Y):
    _norm = torch.rsqrt(torch.tensor(encoded_X.shape[0]))
    encoded_X = _norm * encoded_X
    encoded_Y = _norm * encoded_Y

    cov_X = torch.mm(encoded_X.T, encoded_X)
    cov_Y = torch.mm(encoded_Y.T, encoded_Y)
    cov_XY = torch.mm(encoded_X.T, encoded_Y)
    return cov_X, cov_Y, cov_XY


class GraphDPNet(LightningModule):
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

        self.save_hyperparameters(ignore=["configs", "optimizer"])
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

        encoded_X, encoded_X_prime = torch.tensor_split(features[::2], 2, dim=0)
        encoded_Y, encoded_Y_prime = torch.tensor_split(features[1::2], 2, dim=0)

        cov_X, cov_Y, cov_XY = compute_covs(encoded_X, encoded_Y)
        cov_X_prime, cov_Y_prime, cov_XY_prime = compute_covs(
            encoded_X_prime, encoded_Y_prime
        )

        metrics = {}
        S_loss = -1 * relaxed_projection_score(cov_X, cov_Y, cov_XY)
        EY_loss = EY_CCA_loss(
            cov_X, cov_Y, cov_XY, cov_X_prime, cov_Y_prime, cov_XY_prime
        )
        P_loss = -1 * vamp_score(cov_X, cov_Y, cov_XY, schatten_norm=2)
        metrics["train/relaxed_projection_score"] = (-1.0 * S_loss).exp().item()
        metrics["train/projection_score"] = -1.0 * P_loss.item()
        metrics["train/EY_score"] = -1.0 * EY_loss.item()
        # Compute the losses
        if self.hparams.use_relaxed_loss:
            svd_loss = EY_loss
        else:
            svd_loss = P_loss
        if self.hparams.metric_deformation_loss_coefficient > 0.0:
            metric_deformation_loss = 0.5 * (
                log_fro_metric_deformation_loss(cov_X)
                + log_fro_metric_deformation_loss(cov_Y)
            )
            metric_deformation_loss *= self.hparams.metric_deformation_loss_coefficient
            metrics["train/metric_deformation_loss"] = metric_deformation_loss.item()
            svd_loss += metric_deformation_loss
        # cov_eigs = torch.linalg.eigvalsh(cov_X)
        # top_eigs = torch.topk(cov_eigs, k=5, largest=True).values

        # covXY_svals = torch.linalg.svdvals(cov_XY)
        # top_svals = torch.topk(covXY_svals, k=5, largest=True).values
        # for i, v in enumerate(top_eigs):
        #     metrics[f"train/cov_eig_{i}"] = v.item()
        # for i, v in enumerate(top_svals):
        #     metrics[f"train/covXY_sval_{i}"] = v.item()
        # metrics["train/total_loss"] = svd_loss.item()
        # metrics["train/phi_XY_difference"] = (
        #     torch.linalg.matrix_norm(encoded_X - encoded_Y).item()
        #     / torch.linalg.matrix_norm(encoded_X).item()
        # )
        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True)
        return svd_loss

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self.optimizer(self.parameters(), **kw)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.schnet, norm_type=2)
        self.log_dict(norms)


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
        return inputs
