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

#General definitions
experiment_path = Path(__file__).parent
data_path = experiment_path / 'data'
ckpt_path = experiment_path / 'ckpt'
results_path = experiment_path / 'results'
configs = ml_confs.from_file(experiment_path / 'configs.yaml')

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
        val_data = traj_to_contexts(np_ordered_MNIST['validation']['image'])
    return train_data, val_data, np_ordered_MNIST['test']
########## MODELS ##########
def run_DMD():
    from kooplearn.models import DMD
    train_data, val_data, test_data = load_data()
    dmd_model = DMD(reduced_rank = configs.reduced_rank, rank=configs.classes).fit(train_data)
    print(f"Train risk: {dmd_model.risk():.3f}\nValidation risk: {dmd_model.risk(val_data):.3f}")
    oracle = load_oracle()
    return evaluate_model(dmd_model, oracle, test_data)

def kernel_DMD():
    pass
########### MAIN  ###########

AVAIL_MODELS = {
    'DMD': run_DMD,
    'Kernel_DMD': kernel_DMD    
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