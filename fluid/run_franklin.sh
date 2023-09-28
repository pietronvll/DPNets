#!/bin/bash
#PBS -l select=1:ncpus=128:mem=64gb
#PBS -l walltime=24:00:00
#PBS -q cpu
#PBS -N fluid-DPNets
#PBS -j oe
#PBS -o /work/pnovelli/dp_examples/fluid/logs/output.log
#PBS -J 0-9
#PBS -r y

module load miniconda3/mc3-py39

source activate /projects/mlcompchem/mambaforge/envs/kooplearn
# Change to the working directory
cd /work/pnovelli/dp_examples/fluid 

for model_name in DynamicalAE ConsistentAE; do
    python run.py --model="$model_name" --rngseed=$PBS_ARRAY_INDEX
done
# Wait for all instances to finish
wait