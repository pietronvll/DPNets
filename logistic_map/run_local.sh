for FDIM in {6..10}; do
    for model_name in Cheby-T VAMPNets DPNets DPNets-relaxed NoiseKernel; do
        python run.py --model="$model_name" --fdim=$PBS_ARRAY_INDEX
    done
done
# Wait for all instances to finish
wait