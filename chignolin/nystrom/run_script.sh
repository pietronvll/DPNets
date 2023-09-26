#!/bin/bash
#PBS -l select=1:ncpus=64:mem=192gb
#PBS -l walltime=24:00:00
#PBS -q cpu
#PBS -N chignolin-nystrom
#PBS -j oe
#PBS -o /work/pnovelli/dp_examples/chignolin/logs/output.log

module load miniconda3/mc3-py39

source activate /projects/mlcompchem/mambaforge/envs/kooplearn

cd /work/pnovelli/dp_examples/chignolin/nystrom