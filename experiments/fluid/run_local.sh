for RNG_SEED in {0..19}; do
    for model_name in DynamicalAE ConsistentAE; do
        python run.py --model="$model_name" --rngseed=$RNG_SEED
    done
done
# Wait for all instances to finish
wait