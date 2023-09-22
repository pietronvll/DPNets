from kooplearn.abc import BaseModel, TrainableFeatureMap
from data_pipeline import CNNEncoder, ClassifierFeatureMap, ClassifierModule
from kooplearn.data import traj_to_contexts
from kooplearn.nn.data import TrajToContextsDataset
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
import ml_confs
import argparse
import pickle
from pathlib import Path
import lightning
import torch

#General definitions
experiment_path = Path(__file__).parent
data_path = experiment_path / 'data'
ckpt_path = experiment_path / 'ckpt'
results_path = experiment_path / 'results'
configs = ml_confs.from_file(experiment_path / 'configs.yaml')

#General Trainer configs
trainer_kwargs = {
    'accelerator': 'gpu',
    'devices': 1,
    'max_epochs': configs.max_epochs,  
    'enable_progress_bar': False,
    'enable_checkpointing': False,
    'logger': False
}
def stack_forest(results):
    acc = [np.mean(r['accuracy']) for r in results]
    out = results[np.argmax(acc)]
    #Replace the accuracy with the mean accuracy
    _accs = np.array([r['accuracy'] for r in results])
    out['accuracy'] = _accs.mean(axis=0)
    out['accuracy_std'] = _accs.std(axis=0)
    return out
  
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
def evaluate_model(model: BaseModel, oracle:TrainableFeatureMap,  test_data):
    cap_saved_imgs = 10
    assert model.is_fitted
    test_labels = test_data['label']
    test_images = test_data['image']
    test_images = np.expand_dims(test_images, 1)
    report = {
        'accuracy': [],
        'label': [],
        'image': [],
        'times': []
    }
    for t in range(1, configs.eval_up_to_t + 1):
        pred = model.predict(test_images, t=t).reshape(-1, 28 ,28)
        pred_labels = oracle(pred)
        pred_labels = pred_labels.argmax(axis=1)
        accuracy = (pred_labels == (test_labels + t)%configs.classes ).mean()
        report['accuracy'].append(accuracy)
        report['image'].append(pred[:cap_saved_imgs])
        report['label'].append(pred_labels[:cap_saved_imgs])
        report['times'].append(t)
    return report
def load_oracle():
    oracle = ClassifierFeatureMap.load(ckpt_path / 'oracle')
    return oracle
def load_data(torch: bool = False):
    ordered_MNIST = load_from_disk(str(data_path))
    #Creating a copy of the dataset in numpy format
    np_ordered_MNIST = ordered_MNIST.with_format(type='numpy', columns=['image', 'label'])
    if torch:
        train_ds = TrajToContextsDataset(ordered_MNIST['train']['image'])
        val_ds = TrajToContextsDataset(ordered_MNIST['validation']['image'])
        #Dataloaders 
        train_data = DataLoader(train_ds, batch_size=configs.batch_size, shuffle=True)
        val_data = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    else:
        train_data = traj_to_contexts(np_ordered_MNIST['train']['image'])
        val_data = np_ordered_MNIST['validation']
    return train_data, val_data, np_ordered_MNIST['test']
########## MODELS ##########
def run_baseline():
    from kooplearn.models import DeepEDMD
    oracle = load_oracle()
    train_data, _, test_data = load_data()
    classifier_model = DeepEDMD(oracle, reduced_rank=False, rank=configs.classes).fit(train_data.copy())
    return evaluate_model(classifier_model, oracle, test_data)
def run_VAMPNets():
    from kooplearn.models.feature_maps import VAMPNet
    from kooplearn.models import DeepEDMD
    from lightning.pytorch.callbacks import LearningRateFinder
    oracle = load_oracle()
    train_dl, val_dl, _ = load_data(torch=True)
    train_data, _, test_data = load_data()   
    #See https://github.com/Lightning-AI/lightning/blob/f1df76ce840119f7baee702ef2df1373e516f12f/src/lightning/pytorch/tuner/lr_finder.py#L176 for an explanation of how the lr is chosen
    results = []
    for rng_seed in range(configs.num_rng_seeds): #Reproducibility
        lr_finder = LearningRateFinder(min_lr=1e-6, max_lr=1e-2, early_stop_threshold=None)
        trainer = lightning.Trainer(**trainer_kwargs, callbacks=[lr_finder])   
        #Defining the model
        feature_map = VAMPNet(
            CNNEncoder,
            torch.optim.Adam,
            {'lr': 1e-4},
            trainer,
            {'num_classes': configs.classes},
            center_covariances=False,
            seed=rng_seed
        )
        feature_map.fit(train_dl)
        VAMPNet_model = DeepEDMD(feature_map, reduced_rank = configs.reduced_rank, rank=configs.classes).fit(train_data)
        results.append(evaluate_model(VAMPNet_model, oracle, test_data))
    return stack_forest(results)

def _base_DPNets(relaxed: bool, metric_deformation_coeff = 1.0, rng_seed: int = 0):
    from kooplearn.models.feature_maps import DPNet
    from kooplearn.models import DeepEDMD
    from lightning.pytorch.callbacks import LearningRateFinder
    train_dl, val_dl, _ = load_data(torch=True)
    train_data, _, test_data = load_data()   
    
    #See https://github.com/Lightning-AI/lightning/blob/f1df76ce840119f7baee702ef2df1373e516f12f/src/lightning/pytorch/tuner/lr_finder.py#L176 for an explanation of how the lr is chosen
    
    lr_finder = LearningRateFinder(min_lr=1e-6, max_lr=1e-2, early_stop_threshold=None)
    trainer = lightning.Trainer(**trainer_kwargs, callbacks=[lr_finder])   
    #Defining the model
    feature_map = DPNet(
        CNNEncoder,
        torch.optim.Adam,
        {'lr': 1e-4},
        trainer,
        use_relaxed_loss=relaxed,
        metric_deformation_loss_coefficient = metric_deformation_coeff,
        encoder_kwargs={'num_classes': configs.classes},
        center_covariances=False,
        seed=rng_seed
    )
    feature_map.fit(train_dl)
    DPNet_model = DeepEDMD(feature_map, reduced_rank = configs.reduced_rank, rank=configs.classes).fit(train_data)
    return DPNet_model

def _base_DPNets_HPOPT(relaxed: bool, rng_seed: int = 0): 
    import optuna    
    oracle = load_oracle()
    _, val_data, _ = load_data()
    #HP Opt with Optuna
    def objective(trial):
        metric_deformation = trial.suggest_float("metric_deformation", 1e-2, 1, log=True)
        model = _base_DPNets(relaxed, metric_deformation, rng_seed)
        report = evaluate_model(model, oracle, val_data)
        return np.mean(report['accuracy']) 
    
    sampler = optuna.samplers.TPESampler(seed=0)  #Reproductibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=configs.trial_budget)
    return study.best_params['metric_deformation']

def run_DPNets():
    _, _ , test_data = load_data()
    oracle = load_oracle()
    results = []
    for rng_seed in range(configs.num_rng_seeds): #Reproducibility
        metric_deformation = _base_DPNets_HPOPT(False, rng_seed)
        model = _base_DPNets(False, metric_deformation, rng_seed)
        results.append(evaluate_model(model, oracle, test_data))
    return stack_forest(results)

def run_DPNets_relaxed():
    _, _ , test_data = load_data()
    oracle = load_oracle()
    results = []
    for rng_seed in range(configs.num_rng_seeds): #Reproducibility
        metric_deformation = _base_DPNets_HPOPT(True, rng_seed)
        model = _base_DPNets(False, metric_deformation, rng_seed)
        results.append(evaluate_model(model, oracle, test_data))
    return stack_forest(results)

def run_DynamicalAE():
    pass
def run_ConsistentAE():
    pass

def run_DMD():
    from kooplearn.models import DMD
    train_data, _, test_data = load_data()
    dmd_model = DMD(reduced_rank = configs.reduced_rank, rank=configs.classes).fit(train_data)
    oracle = load_oracle()
    return evaluate_model(dmd_model, oracle, test_data)
def _base_kernel_DMD(kernel):
    from kooplearn.models import KernelDMD
    train_data, _, test_data = load_data()
    kernel_model = KernelDMD(kernel=kernel, reduced_rank = configs.reduced_rank, rank=configs.classes, svd_solver='arnoldi').fit(train_data)
    oracle = load_oracle()
    return evaluate_model(kernel_model, oracle, test_data)
def _base_kernel_lsOPT(base_kernel, **kwargs):
    from kooplearn.models import KernelDMD
    from scipy.spatial.distance import pdist
    import optuna
    oracle = load_oracle()
    train_data, val_data, test_data = load_data()
    _trimgs = train_data[:, 0, ...]
    _trimgs = _trimgs.reshape(_trimgs.shape[0], -1)
    ls0 = np.median(pdist(_trimgs))
    #HP Opt with Optuna
    def objective(trial):
        ls = trial.suggest_float("ls", 0.1*ls0, 10*ls0)
        kernel = base_kernel(length_scale=ls, **kwargs)
        kernel_model = KernelDMD(kernel=kernel, reduced_rank = configs.reduced_rank, rank=configs.classes, svd_solver='arnoldi').fit(train_data)
        report = evaluate_model(kernel_model, oracle, val_data)
        return np.mean(report['accuracy'])
    
    sampler = optuna.samplers.TPESampler(seed=0)  #Reproductibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=configs.trial_budget)
    return study.best_params['ls']
def run_RBF_DMD():
    from sklearn.gaussian_process.kernels import RBF
    ls = _base_kernel_lsOPT(RBF)
    return _base_kernel_DMD(RBF(length_scale=ls))
def run_Poly3_DMD():
    #Poly3 Kernel DMD
    from sklearn.metrics.pairwise import polynomial_kernel
    return _base_kernel_DMD(polynomial_kernel)
def run_AbsExp_DMD():
    #Absolute Exponential a.k.a. Matern 0.5 Kernel DMD
    from sklearn.gaussian_process.kernels import Matern
    ls = _base_kernel_lsOPT(Matern, nu=0.5)
    return _base_kernel_DMD(Matern(length_scale=ls, nu=0.5))
########### MAIN  ###########

AVAIL_MODELS = {
    'DMD': run_DMD,
    'KernelDMD-RBF': run_RBF_DMD,
    'KernelDMD-Poly3': run_Poly3_DMD,
    'KernelDMD-AbsExp': run_AbsExp_DMD,
    'VAMPNets': run_VAMPNets,
    'Baseline-Classifier': run_baseline,
    'DPNets': run_DPNets,
    'DPNets-relaxed': run_DPNets_relaxed,
    'DynamicalAE': run_DynamicalAE,
    'ConsistentAE': run_ConsistentAE
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