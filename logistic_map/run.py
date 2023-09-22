from pathlib import Path
import ml_confs

#General definitions
experiment_path = Path(__file__).parent
data_path = experiment_path / 'data'
ckpt_path = experiment_path / 'ckpt'
results_path = experiment_path / 'results'
configs = ml_confs.from_file(experiment_path / 'configs.yaml')