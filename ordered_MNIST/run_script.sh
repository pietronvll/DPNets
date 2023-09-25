#!/bin/bash
#PBS -l select=1:ncpus=16:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -N ordered_MNIST-DPNets
#PBS -j oe
#PBS -o /work/pnovelli/dp_examples/ordered_MNIST/logs/output.log

module load miniconda3/mc3-py39

source activate /projects/mlcompchem/mambaforge/envs/kooplearn
# Change to the working directory
cd /work/pnovelli/dp_examples/ordered_MNIST 
#List containing the models to run
MODELS=("")

for model_name in Oracle-Features DMD KernelDMD-RBF KernelDMD-Poly3 KernelDMD-AbsExp VAMPNets DPNets DPNets-relaxed DynamicalAE
do
    python run.py --model="$model_name"
done
# Wait for all instances to finish
wait