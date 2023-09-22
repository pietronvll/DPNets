# Deep projections - Experiments 

## General notes:
1. Each experiment should be self-contained
2. The folder structure for each experiment should be as standardized as possible:
    - `/data` (ignored by git)
    - `/results`
    - `figures.ipynb` to make figures by loading results
    - `data_pipeline.py` (if needed)
    - `configs.yaml` to be read by `ml_confs`
    - `run.py --model='DPNets'` - Containing sandboxed functions to evaluate the model passed (and read from argparse) based on `configs.yaml`. If called without `--model` just print the available models and do nothing. 

## Noisy Logistic Map


## Ordered MNIST