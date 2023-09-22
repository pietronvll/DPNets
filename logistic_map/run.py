from pathlib import Path
import ml_confs
from kooplearn.datasets import LogisticMap
import numpy as np
from kooplearn.abc import FeatureMap
from scipy.integrate import romb  
from kooplearn._src.metrics import directed_hausdorff_distance
import torch
import pickle
import argparse
import lightning
import optuna
import scipy.special
from functools import partial
from kooplearn.models.feature_maps import ConcatenateFeatureMaps
from torch.utils.data import DataLoader
from kooplearn.nn.data import TrajToContextsDataset

#General definitions
experiment_path = Path(__file__).parent
data_path = experiment_path / 'data'
ckpt_path = experiment_path / 'ckpt'
results_path = experiment_path / 'results'
configs = ml_confs.from_file(experiment_path / 'configs.yaml', register_jax_pytree=False)

logistic = LogisticMap(N = configs.N, rng_seed=0) #Reproducibility
#Data preparation
sample_traj = logistic.sample(0.5, configs.num_train+configs.num_val+configs.num_test)
dataset = {
    'train': sample_traj[:configs.num_train],
    'validation': sample_traj[configs.num_train:configs.num_train+configs.num_val],
    'test': sample_traj[configs.num_train+configs.num_val:]
}

#Preparing the data
train_data = torch.from_numpy(dataset['train']).float()
val_data = torch.from_numpy(dataset['validation']).float()

train_ds = TrajToContextsDataset(train_data)
val_ds = TrajToContextsDataset(val_data)

train_dl = DataLoader(train_ds, batch_size=configs.batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

#Init report dict
dl_reports = {}

min_lr = 1e-4
max_lr = 1e-2

opt = torch.optim.Adam

trainer_kwargs = {
    'accelerator': 'cpu',
    'devices': 1,
    'max_epochs': configs.max_epochs,
    'enable_checkpointing': False,
    'logger': False  
}

def stack_forest(results):
    out = {}
    for run_res in results:
        for key in run_res:
            if key not in out:
                out[key] = []
            out[key].append(run_res[key])
    #Take the mean and std of each metric
    avgout = {}
    for key in out:
        if key == 'eigenvalues':
            best_hausdorff = np.argmax(out['hausdorff-dist'])
            avgout[key] = out[key][best_hausdorff]
        else:
            avgout[key] = np.mean(out[key])
            avgout[key + '_std'] = np.std(out[key])
    return avgout
def sanitize_filename(filename):
    # Define a set of characters that are not allowed in file names on most systems
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    # Replace any illegal characters with an underscore
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    # Remove leading and trailing spaces
    filename = filename.strip()
    # Remove dots and spaces from the beginning and end of the filename
    filename = filename.strip('. ')
    # Ensure the filename is not empty and is not just composed of dots
    if not filename:
        filename = 'unnamed'
    # Limit the filename length to a reasonable number of characters
    max_length = 255  # Max file name length on most systems
    if len(filename) > max_length:
        filename = filename[:max_length]
    return filename
def population_covs(feature_map: FeatureMap, pow_of_two_k: int = 12):
    """Computes the population covariance and cross-covariance"""
    x = np.linspace(0, 1, 2**pow_of_two_k + 1)[:, None]
    vals, lv = logistic.eig(eval_left_on=x)
    perron_eig_idx = np.argmax(np.abs(vals))
    pi = lv[:, perron_eig_idx]
    assert np.isreal(pi).all()
    pi = pi.real
    pi = pi/romb(pi, dx = 1/2**pow_of_two_k) #Normalization of π
    #Evaluating the feature map
    phi = feature_map(x) # [2**pow_of_two_k + 1, d]
    #Covariance
    cov_unfolded = phi.reshape(2**pow_of_two_k + 1, -1, 1)*phi.reshape(2**pow_of_two_k + 1, 1, -1)*pi.reshape(-1, 1, 1)
    cov = romb(cov_unfolded, dx = 1/2**pow_of_two_k, axis=0)
    #Cross-covariance
    alphas = np.stack([logistic.noise_feature_composed_map(x, n) for n in range(logistic.N + 1)], axis = 1)
    betas = np.stack([logistic.noise_feature(x, n) for n in range(logistic.N + 1)], axis = 1)

    cov_alpha_unfolded = phi.reshape(2**pow_of_two_k + 1, -1, 1)*alphas.reshape(2**pow_of_two_k + 1, 1, -1)*pi.reshape(-1, 1, 1)
    cov_beta_unfolded = phi.reshape(2**pow_of_two_k + 1, -1, 1)*betas.reshape(2**pow_of_two_k + 1, 1, -1)

    cov_alpha = romb(cov_alpha_unfolded, dx = 1/2**pow_of_two_k, axis=0)
    cov_beta = romb(cov_beta_unfolded, dx = 1/2**pow_of_two_k, axis=0)
    
    cross_cov = cov_alpha@(cov_beta.T)
    return cov, cross_cov
def evaluate_representation(feature_map: FeatureMap):
    report = {}
    #Compute OLS estimator
    cov, cross_cov = population_covs(feature_map)
    OLS_estimator = np.linalg.solve(cov, cross_cov)
    #Eigenvalue estimation
    OLS_eigs = np.linalg.eigvals(OLS_estimator)
    report['hausdorff-dist'] = directed_hausdorff_distance(OLS_eigs, logistic.eig())
    #VAMP2-score
    M = np.linalg.multi_dot([np.linalg.pinv(cov, hermitian=True), cross_cov, np.linalg.pinv(cov, hermitian=True), cross_cov.T])
    feature_dim = cov.shape[0]
    report['optimality-gap'] = np.sum(logistic.svals()[:feature_dim]**2) - np.trace(M)
    #Metric distortion
    cov_eigs = np.linalg.eigvalsh(cov)
    report['distortionless-gap'] = np.sqrt(np.mean(cov_eigs - np.ones_like(cov_eigs))**2)
    report['eigenvalues'] = OLS_eigs
    return report
class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        #Assuming x is in [0, 1]
        x = 2*torch.pi*x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
class SimpleMLP(torch.nn.Module):
    def __init__(self, feature_dim: int, layer_dims: list[int], activation = torch.nn.LeakyReLU):
        super().__init__()
        self.activation = activation
        lin_dims = [2] + layer_dims + [feature_dim] #The 2 is for the sinusoidal embedding
        
        layers = []
        for layer_idx in range(len(lin_dims) - 2):
            layers.append(torch.nn.Linear(lin_dims[layer_idx], lin_dims[layer_idx+1], bias=False))
            layers.append(activation())
        layers.append(torch.nn.Linear(lin_dims[-2], lin_dims[-1], bias=True))
        self.layers = torch.nn.ModuleList(layers)
        self.sin_embedding = SinusoidalEmbedding()
    
    def forward(self, x):
        #Sinusoidal embedding
        x = self.sin_embedding(x)
        #MLP
        for layer in self.layers:
            x = layer(x)
        return x
def kaiming_init(model):
    for p in model.parameters():
        psh = p.shape
        if len(psh) == 2: #Linear layers
            _, in_shape = psh
            if in_shape == 2: #Initial layer
                torch.nn.init.uniform_(p, -1, 1)
            else:
                acname = model.activation.__name__.lower()
                if acname == 'leakyrelu':
                    acname = 'leaky_relu'
                torch.nn.init.kaiming_uniform_(p, a= 1, nonlinearity=acname)
        else: #Bias
            torch.nn.init.zeros_(p)

def run_VAMPNets():
    pass
def _base_run_DPNets(relaxed: bool, metric_deformation_coeff: float, rng_seed: int, feature_dim: int, lr:float):
    from kooplearn.models.feature_maps import DPNet
    trainer = lightning.Trainer(**trainer_kwargs)

    net_kwargs = {
        'feature_dim': feature_dim,
        'layer_dims': configs.layer_dims
    }
    opt_args = {'lr': lr}
    #Defining the model
    dpnet_fmap = DPNet(
        SimpleMLP,
        opt,
        opt_args,
        trainer,
        use_relaxed_loss=relaxed,
        metric_deformation_loss_coefficient=metric_deformation_coeff,
        encoder_kwargs=net_kwargs,
        encoder_timelagged=SimpleMLP,
        encoder_timelagged_kwargs=net_kwargs,
        center_covariances=False,
        seed=rng_seed
    )
    #Init
    torch.manual_seed(rng_seed)
    kaiming_init(dpnet_fmap.lightning_module.encoder)
    kaiming_init(dpnet_fmap.lightning_module.encoder_timelagged)
    return evaluate_representation(dpnet_fmap)
def _base_DPNets_HPOPT(relaxed: bool, rng_seed: int, feature_dim: int):
    def objective(trial):
        lr = trial.suggest_float("lr", min_lr, max_lr, log=True)
        metric_deformation = trial.suggest_float("metric_deformation", 1e-2, 1, log=True)
        report = _base_run_DPNets(relaxed, metric_deformation, rng_seed, feature_dim, lr)
        return report['optimality-gap']
    sampler = optuna.samplers.TPESampler(seed=0)  #Reproductibility
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=configs.trial_budget)
    return study.best_params
def run_DPNets():
    full_report = {} 
    for feature_dim in range(2, configs.max_feature_dim + 1):
        report = []
        for rng_seed in range(configs.num_rng_seeds):
            best_params = _base_DPNets_HPOPT(False, rng_seed, feature_dim)
            _res = _base_run_DPNets(False, best_params['metric_deformation'], rng_seed, feature_dim, best_params['lr'])
            report.append(_res)
        full_report[f'{feature_dim}_features'] = stack_forest(report)
    return full_report
def run_DPNets_relaxed():
    full_report = {} 
    for feature_dim in range(2, configs.max_feature_dim + 1):
        report = []
        for rng_seed in range(configs.num_rng_seeds):
            best_params = _base_DPNets_HPOPT(True, rng_seed, feature_dim)
            _res = _base_run_DPNets(True, best_params['metric_deformation'], rng_seed, feature_dim, best_params['lr'])
            report.append(_res)
        full_report[f'{feature_dim}_features'] = stack_forest(report)
    return full_report
def run_ChebyT():
    def ChebyT(order: int = 3):
        def scaled_chebyt(n, x):
            return scipy.special.eval_chebyt(n, 2*x - 1)
        fn_list = [partial(scaled_chebyt, n) for n in range(order + 1)]
        return ConcatenateFeatureMaps(fn_list)   
    full_report = {} 
    for feature_dim in range(2, configs.max_feature_dim + 1):
        full_report[f'{feature_dim}_features'] = evaluate_representation(ChebyT(feature_dim - 1))
    return full_report

AVAIL_MODELS = {
    'VAMPNets': run_VAMPNets,
    'DPNets': run_DPNets,
    'DPNets-relaxed': run_DPNets_relaxed,
    'Cheby-T': run_ChebyT,
}

def main():
    parser = argparse.ArgumentParser(description="Run the experiment on a specific model.")
    parser.add_argument("--model", choices=AVAIL_MODELS.keys(), help="Specify the model to run.")

    args = parser.parse_args()

    if args.model:
        if args.model in AVAIL_MODELS:
            results = AVAIL_MODELS[args.model]()
            results['name'] = args.model
            fname = sanitize_filename(args.model) + '_results.pkl'
            if not results_path.exists():
                results_path.mkdir()
            with open(results_path / fname, 'wb') as f:
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