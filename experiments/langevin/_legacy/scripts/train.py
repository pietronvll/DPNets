from pathlib import Path
from typing import Callable
import jax
import jax.numpy as jnp
import optax
import wandb
from jaxtyping import Float, Array
from DPNets.jax.scores import DPNets_continuous_P, fro_metric_reg, vn_metric_reg
from models.utils.io import load_json
from models.utils.jax import batch_iterator, init_ckptmanager, training_greeting
from models.utils.misc import rich_pbar
from models.utils.typing import JSONNamespace
from kooplearn.jax.estimators import tikhonov_regression, eig
from models.langevin.utils.io import load_data, load_evd_ref_data
from models.langevin.utils.nn import MLP
from flax.training.train_state import TrainState
from orbax.checkpoint import CheckpointManagerOptions
import numpy as np


CPU_DEVICE = jax.devices('cpu')[0]
precision = jax.lax.Precision.HIGHEST
model_dir = Path(__file__).parent.parent #models/langevin

class AutoResettingIterator:
    def __init__(self, constructor: Callable, *args, **kwargs):
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs
        self.key = kwargs.pop('key', jax.random.PRNGKey(0))
        self.key, _k = jax.random.split(self.key) 
        self.iterator = constructor(*args, **kwargs, key = _k)
        self.epochs = 0
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epochs += 1
            self.key, _k = jax.random.split(self.key)
            self.iterator = self.constructor(*self.args, **self.kwargs, key = _k)
            return next(self.iterator)


def loop(configs: JSONNamespace):
    key = jax.random.PRNGKey(configs.seed)
    #Data loading
    ref_evd = load_evd_ref_data(model_dir)
    train_dataset, metadata = load_data(model_dir, split = 'train')
    val_dataset, _ = load_data(model_dir, split = 'val')
    inv_gamma = 1.0/metadata['gamma']
    kBT = metadata['kBT']
    #Init model and optimizer
    net = MLP(configs.mlp_layers)
    _ref_data = train_dataset[0, 0][None, None]
    net_table = net.tabulate(key, _ref_data)
    variables = net.init(key, _ref_data)
    tx = optax.adam(learning_rate=configs.lr)
    state = TrainState.create(apply_fn=net.apply, params=variables['params'], tx=tx)
    #Init logger
    run = wandb.init(project=configs.project_name, config=configs.to_dict(), entity=configs.entity)
    #Init checkpoint manager
    ckpt_dir = model_dir / 'checkpoints'
    options = CheckpointManagerOptions(max_to_keep=5, best_fn= lambda m: float(m['P']), best_mode='max')
    ckpt_manager = init_ckptmanager(run.name, ckpt_dir, options)

    #Derivatives of the network wrt the inputs
    _dnet = jax.jacfwd(net.apply, argnums=1)
    dnet = jax.vmap(_dnet, in_axes=(None, 0))
    d2net = jax.vmap(jax.jacfwd(_dnet, argnums=1), in_axes=(None, 0))

    #Single step of training
    @jax.jit
    def step(state: TrainState, batch: Float[Array, "n 4"]) -> tuple[float, TrainState]:
        cov_norm = jax.lax.rsqrt(float(batch.shape[0]))
        x = batch[:, 0][:, None]
        dV = batch[:, 1]
        def loss_fn(params: dict):
            variables = {'params': params}
            # x [n, 1]
            psi = state.apply_fn(variables, x)*cov_norm # [n, d]
            dPsi = jnp.squeeze(dnet(variables, x)) # [n, d]
            d2Psi = jnp.squeeze(d2net(variables, x)) # [n, d]
            genPsi = ((kBT*d2Psi.T -(dV*dPsi.T)).T)*inv_gamma*cov_norm
            cov = jnp.matmul(psi.T, psi, precision=precision)
            dCov = jnp.matmul(psi.T, genPsi, precision=precision)
            P = DPNets_continuous_P(cov, dCov) 
            reg = fro_metric_reg(cov, configs.metric_reg)
            loss = -P + reg
            metrics = {'train/loss': loss, 'train/P': P, 'train/reg': reg}
            return loss, metrics
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return (loss, metrics), state
    
    #Evaluation step
    @jax.jit
    def evaluate(state: TrainState, batch: Float[Array, "n 4"]) -> tuple[float]:
        cov_norm = jax.lax.rsqrt(float(batch.shape[0]))
        x = batch[:, 0][:, None]
        dV = batch[:, 1]
        variables = {'params': state.params}
        # Batch [n, 1]
        psi = state.apply_fn(variables, x)*cov_norm # [n, d]
        dPsi = jnp.squeeze(dnet(variables, x)) # [n, d]
        d2Psi = jnp.squeeze(d2net(variables, x)) # [n, d]
        genPsi = ((kBT*d2Psi.T -(dV*dPsi.T)).T)*inv_gamma*cov_norm
        cov = jnp.matmul(psi.T, psi, precision=precision)
        dCov = jnp.matmul(psi.T, genPsi, precision=precision)
        metrics = {'val/P': DPNets_continuous_P(cov, dCov), 'val/reg': fro_metric_reg(cov, configs.metric_reg), 'val/vn_reg': vn_metric_reg(cov, configs.metric_reg)}
        estimator = tikhonov_regression(cov, configs.estimator_reg)
        covs_and_estimator = {'cov': cov, 'dCov': dCov, 'estimator': estimator}
        return metrics, covs_and_estimator

    key, val_key = jax.random.split(key)
    val_sampler = AutoResettingIterator(batch_iterator, len(val_dataset), configs.validation.batch_size, key=val_key, shuffle=configs.validation.shuffle)   
    metrics_schema = {'P': 0, 'reg': 0, 'vn_reg': 0} 
    pbar = rich_pbar(metrics_schema)
    training_greeting(run.name, net_table)
    #Progress bar
    eval_error_log = {'ref_evd': ref_evd}
    with pbar:
        training_task = pbar.add_task(description="Training", total = configs.training.max_epochs, **metrics_schema)
        #Actual training loop
        for epoch in range(configs.training.max_epochs):
            key, tr_key = jax.random.split(key)
            train_sampler = batch_iterator(len(train_dataset), configs.training.batch_size, key=tr_key, shuffle=configs.training.shuffle)
            #Epoch-wise loop
            for batch_idxs in train_sampler:
                train_batch = train_dataset[batch_idxs]
                (loss, metrics), state = step(state, train_batch)
                #Logging      
                if state.step%configs.training.log_every == 0:                      
                    wandb.log(metrics, step = state.step)
                if state.step%configs.validation.log_every == 0:
                    val_batch = val_dataset[next(val_sampler)]
                    val_metrics, covs_and_estimators = evaluate(state, val_batch)
                    wandb.log(val_metrics, step = state.step)
                    evd = get_eigvals(covs_and_estimators)
                    evd['values_abserr'] = jnp.abs(evd['values'] - ref_evd['values'])
                    for k, v in evd.items():
                        if k not in eval_error_log:
                            eval_error_log[k] = v
                        else:
                            eval_error_log[k] = jnp.vstack((eval_error_log[k], v))
                    #Checkpointing
                    ckpt = {'state': state, 'eigenvalues': eval_error_log}
                    #Remove 'val/' prefix from metrics
                    val_metrics = {k[4:]: float(v) for k, v in val_metrics.items()}
                    ckpt_manager.save(state.step, ckpt, metrics = val_metrics)
            #Remove 'train/' prefix from metrics
            metrics = {k[6:]: float(v) for k, v in metrics.items()}
            pbar.update(training_task, advance=1, **metrics)
    wandb.finish()

def get_eigvals(covs_and_estimator: dict) -> dict:
    cov = jnp.linalg.eigvalsh(covs_and_estimator['cov'])
    def sort_and_split(eigs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r_eigs, perm = jax.lax.top_k(eigs.real, eigs.shape[0]) 
        return r_eigs, (eigs.imag)[perm]
    dCov = jnp.linalg.eigvals(jax.device_put(covs_and_estimator['dCov'], CPU_DEVICE))
    dCov = sort_and_split(dCov)
    evd = eig(covs_and_estimator['estimator'], covs_and_estimator['dCov']).values
    evd = sort_and_split(evd)
    return {'cov': cov, 'dCov': dCov[0], 'dCov_imag': dCov[1], 'values': evd[0], 'values_imag': evd[1]}

def wandb_eigs_plot(steps: list, eigs: list, title:str) -> wandb.plot:
    x = np.array(steps).tolist()
    y = ((np.array(eigs).real).T).tolist()
    keys = [f'lambda_{i}' for i in range(len(y))]
    return wandb.plot.line_series(xs = x, ys = y, title=title, keys = keys)

if __name__ == "__main__":
    configs = load_json(model_dir / 'configs.json')
    loop(configs)