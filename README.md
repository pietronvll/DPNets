# Learning invariant representations of time-homogeneous stochastic dynamical systems - ICLR 2024

Warning! The repository is in the process of being updated to the latest version of `kooplearn`. Thanks for your patience!

### Timeline:
**Mar 20, 2024:**

- Working on the fluid flow. Need to fix `evaluate_model`.

**Mar 19, 2024:**

- Working on the fluid flow example. Refactoring to the new `kooplearn` API. Adding Kernel Baselines

### Roadmap:
- General
    - [ ] Create `requirements.txt`
    - [ ] Write instructions to run experiments.

- Logistic Map
    - [ ] Check that everything works with `kooplearn==1.1.0`

- Langevin Dynamics. See also [this Github Gist](https://gist.github.com/pietronvll/bc0887f9822311c32b46aa2d803299c1).
    - [ ] Port the code in `torch` 
    - [ ] Use `kooplearn.datasets` to generate the data
    - [ ] Re-run simulations

- Fluid Flow
    - [ ] Load the remeshed data into Zenodo
    - [ ] Make sure every baseline runs on `kooplearn==1.1.0` 

- Ordered MNIST
    - [ ] Almost done. Use what already in `kooplearn`.

- Chignolin
    - [ ] Re-run the Nystroem baseline with `kooplearn`.
    - [ ] Check De Shaw policy about data, and understand whether the checkpoints can be shared or not.
    - [ ] (Ideally) embed `schnetpack's` implementation into `kooplearn`.