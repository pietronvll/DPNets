#!/bin/bash
#List containing the models to run
MODELS=("Cheby-T", "VAMPNets", "DPNets", "DPNets-relaxed")

for model_name in "${MODELS[@]}"; do
    python run.py --model="$model_name"
done
# Wait for all instances to finish
wait