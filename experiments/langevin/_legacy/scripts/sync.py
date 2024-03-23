from pathlib import Path
from models.utils.misc import sync_wandb
from models.utils.io import load_json
model_path = Path(__file__).parent.parent
if __name__ == '__main__':
    configs = load_json(model_path / 'configs.json')
    ckpt_dir = model_path / 'checkpoints'
    sync_wandb('csml', configs.project_name, ckpt_dir)