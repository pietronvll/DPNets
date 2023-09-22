#!/bin/bash
#List containing the models to run
for model_name in Cheby-T DPNets DPNets-relaxed; do
    python run.py --model="$model_name"
done
# Wait for all instances to finish
wait