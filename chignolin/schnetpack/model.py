import ml_confs
import torch
from lightning import LightningModule


class DeepGraphKernel(LightningModule):
    def __init__(
        self,
        configs: ml_confs.Configs,
        n_atoms: int,
        optimizer: torch.optim.Optimizer,
        optim_kargs: dict[str, Any],
    ):
        super().__init__()
        self.schnet = SchNet(configs)
        self.n_atoms = n_atoms
        self.configs = configs
        self.optimizer = optimizer
        self.optim_kwargs = optim_kargs
        self.save_hyperparameters()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.schnet(inputs)

    def training_step(self, batch, batch_idx):
        features = self.schnet(batch)["scalar_representation"]
        batch_size = self.configs.training.batch_size
        phi = features.reshape(
            batch_size * 2, self.n_atoms, self.configs.network.n_final_features
        )
        phi_kme = phi.mean(dim=1)  # Linear kernel mean embedding
        phi_X = phi_kme[::2]
        phi_Y = phi_kme[1::2]

        norm_cov = (batch_size) ** -1.0
        in_cov = norm_cov * (phi_X.T @ phi_X)
        out_cov = norm_cov * (phi_Y.T @ phi_Y)
        cross_cov = norm_cov * (phi_X.T @ phi_Y)

        S = DPNets_S(in_cov, out_cov, cross_cov)
        reg = vn_metric_reg(in_cov, reg=self.configs.training.reg)
        if batch_idx % self.configs.training.log_every == 0:
            P = DPNets_P(in_cov, out_cov, cross_cov, epsilon=0)
            cond = torch.linalg.cond(in_cov)
            logging_dict = {
                "train/S": S,
                "train/P": P,
                "train/cond": cond,
                "train/reg": reg,
            }
            self.log_dict(logging_dict)
        return -S + reg

    def validation_step(self, batch, batch_idx):
        features = self.schnet(batch)["scalar_representation"]
        batch_size = self.configs.training.batch_size
        phi = features.reshape(
            batch_size * 2, self.n_atoms, self.configs.network.n_final_features
        )

        phi_kme = phi.mean(dim=1)  # Linear kernel mean embedding
        phi_X = phi_kme[::2]
        phi_Y = phi_kme[1::2]

        norm_cov = (batch_size) ** -1.0
        in_cov = norm_cov * (phi_X.T @ phi_X)
        out_cov = norm_cov * (phi_Y.T @ phi_Y)
        cross_cov = norm_cov * (phi_X.T @ phi_Y)

        S = DPNets_S(in_cov, out_cov, cross_cov)
        reg = vn_metric_reg(in_cov, reg=self.configs.training.reg)
        P = DPNets_P(in_cov, out_cov, cross_cov, epsilon=0)
        cond = torch.linalg.cond(in_cov)
        logging_dict = {"val/S": S, "val/P": P, "val/cond": cond, "val/reg": reg}
        self.log_dict(logging_dict, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters(), **self.optim_kwargs)
        return optimizer


class SchNet(schnetpack.model.AtomisticModel):
    def __init__(self, configs: JSONNamespace):
        super().__init__(
            input_dtype_str="float32",
            postprocessors=None,
            do_postprocessing=False,
        )
        self.cutoff = configs.data.cutoff
        self.pwise_dist = schnetpack.atomistic.PairwiseDistances()
        self.radial_basis = schnetpack.nn.GaussianRBF(
            n_rbf=configs.network.n_rbf, cutoff=self.cutoff
        )
        self.net = schnetpack.representation.SchNet(
            n_atom_basis=configs.network.n_atom_basis,
            n_interactions=configs.network.n_interactions,
            radial_basis=self.radial_basis,
            cutoff_fn=schnetpack.nn.CosineCutoff(self.cutoff),
        )
        self.final_lin = torch.nn.Linear(
            configs.network.n_atom_basis, configs.network.n_final_features
        )
        self.batch_norm = torch.nn.BatchNorm1d(
            configs.network.n_final_features, affine=False
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = self.pwise_dist(inputs)
        inputs = self.net(inputs)
        inputs["scalar_representation"] = self.final_lin(
            inputs["scalar_representation"]
        )
        # n_atoms = inputs[schnetpack.properties.n_atoms][0] #Assuming constant number of atoms across the batch
        # n_final_features = inputs['scalar_representation'].shape[-1]
        # inputs['scalar_representation'] = self.batch_norm(
        #     inputs['scalar_representation'].view(-1, n_atoms, n_final_features).permute(0, 2, 1)
        # ).permute(0, 2, 1).reshape(-1, n_final_features)
        # inputs['scalar_representation'] =  inputs['scalar_representation']*torch.rsqrt(torch.tensor(n_final_features, dtype=torch.float32))
        return inputs
        return inputs
