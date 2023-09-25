#!/bin/bash
#List containing the models to run
for model_name in Cheby-T VAMPNets DPNets DPNets-relaxed NoiseKernel; do
    python run.py --model="$model_name"
done
# Wait for all instances to finish
wait