from pathlib import Path
import pandas as pd
import jax.numpy as jnp
from jaxtyping import Float, Array

def load_data(model_path: Path, split = 'train') -> tuple[Float[Array, 'n'], dict]:
    data_file = model_path / 'data' / '100k.h5'  
    with pd.HDFStore(data_file) as storedata:
        data = storedata['data'].to_numpy()
        metadata = storedata.get_storer('data').attrs.metadata
    data = jnp.asarray(data)
    length = data.shape[0]
    train_size = int(length*0.7)
    eval_size = int(length*0.1)
    train_idx = jnp.arange(0, train_size)
    eval_idx = jnp.arange(train_size, train_size + eval_size)
    test_idx = jnp.arange(train_size + eval_size, length)
    if split == 'train':
        return data[train_idx], metadata
    elif split == 'val':
        return data[eval_idx], metadata
    elif split == 'test':
        return data[test_idx], metadata
    else:
        raise ValueError(f"split must be one of ['train', 'val', 'test'], got {split}")

def load_evd_ref_data(model_path: Path):
    data_file = model_path / 'data' / 'ref_evd_1025_pts.npz'
    with jnp.load(data_file) as data:
        parsed_data = {'values': data['values'], 'vectors': data['vectors'], 'density': data['density'], 'x': data['domain_sample']}
    return parsed_data