# pytket-cutensornet

[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#)
[![Stack Exchange](https://img.shields.io/badge/StackExchange-%23ffffff.svg?style=for-the-badge&logo=StackExchange)](https://quantumcomputing.stackexchange.com/tags/pytket)

[Pytket](https://docs.quantinuum.com/tket/api-docs/) is a python module for interfacing
with tket, a quantum computing toolkit and optimising compiler developed by Quantinuum.

[cuTensorNet](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html) is a
high-performance library for tensor network computations, developed by NVIDIA.
It is part of the [cuQuantum](https://docs.nvidia.com/cuda/cuquantum/latest/index.html) SDK -
a high-performance library aimed at quantum circuit simulations on the NVIDIA GPUs.

`pytket-cutensornet` is an extension to `pytket` that allows `pytket` circuits and
expectation values to be simulated using `cuTensorNet` via an interface to
[cuQuantum Python](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html).

Some useful links:
- [API Documentation](https://docs.quantinuum.com/tket/extensions/pytket-cutensornet/)

## Getting started

`pytket-cutensornet` is available for Python 3.10, 3.11 and 3.12 on Linux.
In order to use it, you need access to a Linux machine (or WSL) with an NVIDIA GPU of
Compute Capability +7.0 (check it [here](https://developer.nvidia.com/cuda-gpus)).
You will need to install `cuda-toolkit` and `cuquantum-python` before `pytket-cutensornet`;
for instance, in Ubuntu 24.04:

```shell
sudo apt install cuda-toolkit
pip install cuquantum-python
pip install pytket-cutensornet
```

Alternatively, you may install cuQuantum Python following their
[instructions](https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html) using `conda-forge`.
This will include the necessary dependencies from CUDA toolkit. Then, you may install
`pytket-cutensornet` using `pip`.


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
changes on the `main` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Docstrings

We use the Google style docstrings, please see this
[page](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for
reference.

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

```shell
pip install -r tests/test-requirements.txt
pytest tests/
```

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
