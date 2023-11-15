for model_name in ConsistentAE #Oracle-Features DMD KernelDMD-RBF KernelDMD-Poly3 KernelDMD-AbsExp VAMPNets DPNets DPNets-relaxed DynamicalAE
do
    python run.py --model="$model_name"
done
# Wait for all instances to finish
wait