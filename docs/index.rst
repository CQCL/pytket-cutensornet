pytket-cutensornet
==================

``pytket-cutensornet`` is an extension to ``pytket`` that allows ``pytket`` circuits and
expectation values to be simulated using `cuTensorNet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>`_.

`cuTensorNet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>`_ is a
high-performance library for tensor network computations, developed by NVIDIA.
It is part of the `cuQuantum <https://docs.nvidia.com/cuda/cuquantum/latest/index.html>`_ SDK --
a high-performance library aimed at quantum circuit simulations on the NVIDIA GPU chips.

We provide two core functionalities:

* *Full tensor network contraction*: use ``tk_to_tensor_network`` to translate a ``pytket`` circuit to a ``TensorNetwork`` and obtain expectation values and amplitudes via full tensor network contraction using ``cuQuantum``'s optimised contraction path.

* *Matrix Product State (MPS)*: use ``simulate`` to simulate a ``pytket`` circuit, returning an ``MPS`` representation of the output state, of which you can then ``get_amplitude`` or calculate inner products with other MPS via ``vdot``.

Currently, only single-GPU calculations are supported, but a multi-GPU execution will be
implemented in the due course using ``mpi4py`` library.

``pytket-cutensornet`` is available for Python 3.10, 3.11 and 3.12 on Linux.
In order to use it, you need access to a Linux machine (or WSL) with an NVIDIA GPU of
Compute Capability +7.0 (check it `here <https://developer.nvidia.com/cuda-gpus>`_).
You will need to install ``cuda-toolkit`` and ``cuquantum-python`` before ``pytket-cutensornet``;
for instance, in Ubuntu 24.04:

::
   sudo apt install cuda-toolkit
   pip install cuquantum-python
   pip install pytket-cutensornet

Alternatively, you may install cuQuantum Python following their `instructions <https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html>`_
using ``conda-forge``. This will include the necessary dependencies from CUDA toolkit. Then, you may install
``pytket-cutensornet`` using ``pip``.

.. toctree::
    api.rst
    changelog.rst

.. toctree::
   :caption: Example Notebooks

   examples/general_state_tutorial.ipynb
   examples/mps_tutorial.ipynb
   examples/ttn_tutorial.ipynb

.. toctree::
   :caption: Useful links

   Issue tracker <https://github.com/CQCL/pytket-cutensornet/issues>
   PyPi <https://pypi.org/project/pytket-cutensornet/>
