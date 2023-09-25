#!/bin/bash
#PBS -l select=1:ncpus=24:mem=32gb
#PBS -l walltime=24:00:00
#PBS -q cpu
#PBS -N logistic_map-DPNets
#PBS -j oe
#PBS -o /work/pnovelli/dp_examples/logistic_map/logs/output.log

module load miniconda3/mc3-py39

source activate /projects/mlcompchem/mambaforge/envs/kooplearn
# Change to the working directory
cd /work/pnovelli/dp_examples/logistic_map 

for model_name in Cheby-T VAMPNets DPNets DPNets-relaxed NoiseKernel; do
    python run.py --model="$model_name"
done
# Wait for all instances to finish
wait