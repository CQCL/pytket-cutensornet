# Using MPI together with cuTensorNet

In the examples of this folder we use `mpi4py` to run tasks in parallel on multiple GPUs. To run them you will need to install cuTensorNet together with an MPI library. See NVIDIA's [installation instructions](https://docs.nvidia.com/cuda/cuquantum/cutensornet/getting_started.html) on the matter. Notice that a CUDA-aware MPI library must be installed for these examples to run correctly.

If using a supercomputer, it is highly likely that some MPI library is already installed in the system. The steps to set up an environment with CUDA-aware MPI vary from system to system, as well as the commands necessary to run scripts that use MPI. If in doubt, contact your computing service provider.

# Requirements

* `mpi4py`.
* An MPI library with CUDA awareness.