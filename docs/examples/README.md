# Contents

Available tutorials for users:
* `mps_tutorial.ipynb`: Use of MPS simulation and features.
* `ttn_tutorial.ipynb`: Use of TTN simulation and features.
* `mpi/`: Example on how to use MPS for embarrasingly parallel tasks with `mpi4py` see the `mpi` folder.

Developers:
* `check-examples`: The script to check that the Jupyter notebooks are generated correctly from the files in `python/`. To generate the `.ipynb` from these run the `p2j` command in this script.
* `python/`: The `.py` files that generate the `.ipynb` files. As a developer, you are expected to update these files instead of the `.ipynb` files. Remember to generate the latter using the `p2j` command before opening a pull request that changes these examples.