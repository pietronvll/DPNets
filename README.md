# DPNets (submitted to ICLR 2024) - Experiments 
## Noisy Logistic Map
Two scripts: 
1. `run.py` in which are evaluated the Hausdorff distance, Optimality gap, Feasibility gap and estimator eigenvalues. By default runs with many different rng_seeds (for the initialization of the NNs) and is suited to study what happens by varying the feature dimension.
2. `training_loop_inspection.py` which just evaluates the Hausdorff distance along the training loop. By default runs with a fixed feature dimension and many different rng seeds.
## Ordered MNIST

## Fluid Flow

## Chignolin