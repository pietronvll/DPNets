#!/bin/bash
#PBS -l select=1:ncpus=32:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -N fluid-DPNets
#PBS -j oe
#PBS -o /work/pnovelli/dp_examples/fluid/logs/output.log
#PBS -J 1-9
#PBS -r y

module load miniconda3/mc3-py39

source activate /projects/mlcompchem/mambaforge/envs/kooplearn
# Change to the working directory
cd /work/pnovelli/dp_examples/fluid 

for model_name in VAMPNets DPNets DPNets-relaxed; do
    python run.py --model="$model_name" --rngseed=$PBS_ARRAY_INDEX
done
# Wait for all instances to finish
wait