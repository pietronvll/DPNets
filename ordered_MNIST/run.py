from kooplearn.abc import BaseModel, TrainableFeatureMap
import numpy as np
import ml_confs
import argparse
from pathlib import Path
import tensorstore

#General definitions
experiment_path = Path(__file__).parent
data_path = experiment_path / 'data'
configs = ml_confs.from_file(experiment_path / 'configs.yaml')

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
def save_report(report, model_name):
    pass
########## MODELS ##########
def run_DMD():
    pass

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
            # to do - write a function to save results
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