for model_name in DynamicalAE #Oracle-Features DMD KernelDMD-RBF KernelDMD-Poly3 KernelDMD-AbsExp VAMPNets DPNets DPNets-relaxed ConsistentAE
do
    python run.py --model="$model_name"
done
# Wait for all instances to finish
wait