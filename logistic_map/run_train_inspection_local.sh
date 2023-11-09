 for RNG_SEED in {0..1}; do
    for model_name in VAMPNets DPNets DPNets-relaxed; do
        python training_loop_inspection.py --model="$model_name" --rngseed=$RNG_SEED
    done
done
# Wait for all instances to finish
wait