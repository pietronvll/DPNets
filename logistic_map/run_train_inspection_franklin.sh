#!/bin/bash
#PBS -l select=1:ncpus=128:mem=64gb
#PBS -l walltime=24:00:00
#PBS -q cpu
#PBS -N logistic_map-DPNets
#PBS -j oe
#PBS -o /work/pnovelli/dp_examples/logistic_map/logs/output.log
#PBS -J 10-19
#PBS -r y

module load miniconda3/mc3-py39

source activate /projects/mlcompchem/mambaforge/envs/kooplearn
# Change to the working directory
cd /work/pnovelli/dp_examples/logistic_map 

for model_name in VAMPNets DPNets DPNets-relaxed; do
    python training_loop_inspection.py --model="$model_name" --rngseed=$PBS_ARRAY_INDEX
done
# Wait for all instances to finish
wait