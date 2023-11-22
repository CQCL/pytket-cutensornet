# pytket-cutensornet

[Pytket](https://tket.quantinuum.com/api-docs/index.html) is a python module for interfacing
with tket, a quantum computing toolkit and optimising compiler developed by Quantinuum.


[cuTensorNet](https://docs.nvidia.com/cuda/cuquantum/cutensornet/index.html) is a
high-performance library for tensor network computations, developed by NVIDIA.
It is part of the [cuQuantum](https://docs.nvidia.com/cuda/cuquantum/index.html) SDK -
a high-performance library aimed at quantum circuit simulations on the NVIDIA GPU chips,
consisting of two major components:
 - `cuStateVec`: a high-performance library for state vector computations.
 - `cuTensorNet`: a high-performance library for tensor network computations.

Both components have both C and Python API.

`pytket-cutensornet` is an extension to `pytket` that allows `pytket` circuits and
expectation values to be simulated using `cuTensorNet` via an interface to
[cuQuantum Python](https://docs.nvidia.com/cuda/cuquantum/python/index.html).

Currently, only single-GPU calculations are supported, but a multi-GPU execution will be
implemented in the due course using `mpi4py` library.

## Getting started

`pytket-cutensornet` is available for Python 3.9, 3.10 and 3.11 on Linux.
In order to use it, you need access to a Linux machine with either `Volta`, `Ampere`
or `Hopper` GPU and first install `cuQuantum Python` following their installation
[instructions](https://docs.nvidia.com/cuda/cuquantum/python/README.html#installation).
This will include the necessary dependencies such as CUDA toolkit. Then, to install
`pytket-cutensornet`, run:

```pip install pytket-cutensornet```

## Bugs, support and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-cuquantum/issues).

## Development

To install an extension in editable mode, from its root folder run:

```shell
pip install -e .
```

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `develop` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Docstrings

We use the Google style docstrings, please see this 
[page](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for
reference.

#### Formatting

All code should be formatted using
[black](https://black.readthedocs.io/en/stable/), with default options. This is
checked on the CI. The CI is currently using version 22.12.0. You can install it
(as well as pylint as described below) by running from the root package folder:

```shell
pip install -r lint-requirements.txt
```

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. Because of the way
extension modules embed themselves into the `pytket` namespace this is a little
complicated, but it should be sufficient to run the script `mypy-check`
and passing as a single argument the root directory of the module to test. The directory
path should end with a `/`. For example, to run mypy on all Python files in this
repository, when in the root folder, run:

```shell
./mypy-check ./
```
The script requires `mypy` 0.800 or above.

#### Linting

We use [pylint](https://pypi.org/project/pylint/) on the CI to check compliance
with a set of style requirements (listed in `.pylintrc`). You should run
`pylint` over any changed files before submitting a PR, to catch any issues.

### Tests

To run the tests for a module:

1. `cd` into that module's `tests` directory;
2. ensure you have installed `pytest` and any other modules listed in
the `test-requirements.txt` file (all via `pip`);
3. run `pytest`.

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
