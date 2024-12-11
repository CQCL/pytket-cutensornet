# # Tree Tensor Network (TTN) Tutorial

import numpy as np
from time import time
import matplotlib.pyplot as plt
import networkx as nx
from pytket import Circuit
from pytket.circuit.display import render_circuit_jupyter

from pytket.extensions.cutensornet.structured_state import (
    CuTensorNetHandle,
    Config,
    SimulationAlgorithm,
    simulate,
)

# ## Introduction
# This notebook provides examples of the usage of the TTN functionalities of `pytket_cutensornet`. For more information, see the docs at https://docs.quantinuum.com/tket/extensions/pytket-cutensornet/.
# Some good references to learn about Tree Tensor Network state simulation:
# - For an introduction into TTN based simulation of quantum circuits: https://arxiv.org/abs/2206.01000
# - For an introduction on some of the optimisation concerns that are relevant to TTN: https://arxiv.org/abs/2209.03196
# The implementation in pytket-cutensornet differs from previously published literature. I am still experimenting with the algorithm. I intend to write up a document detailing the approach, once I reach a stable version.
# The main advantage of TTN over MPS is that it can be used to efficiently simulate circuits with richer qubit connectivity. This does **not** mean that TTN has an easy time simulating all-to-all connectivity, but it is far more flexible than MPS. TTN's strength is in simulating circuit where certain subsets of qubits interact densely with each other, and there is not that many gates acting on qubits in different subsets.

# ## How to use
# The interface for TTN matches that of MPS. As such, you should be able to run any code that uses `SimulationAlgorithm.MPSxGate` by replacing it with `SimulationAlgorithm.TTNxGate`. Calling `prepare_circuit_mps` is no longer necessary, since `TTNxGate` can apply gates between non-neighbouring qubits.
# **NOTE**: If you are new to pytket-cutensornet, it is highly recommended to start reading the `mps_tutorial.ipynb` notebook instead. More details about the use of the library are discussed there (for instance, why and when to call `CuTensorNetHandle()`).


def random_graph_circuit(n_qubits: int, edge_prob: float, layers: int) -> Circuit:
    """Random circuit with qubit connectivity determined by a random graph."""
    c = Circuit(n_qubits)

    for i in range(layers):
        # Layer of TK1 gates
        for q in range(n_qubits):
            c.TK1(np.random.rand(), np.random.rand(), np.random.rand(), q)

        # Layer of CX gates
        graph = nx.erdos_renyi_graph(n_qubits, edge_prob, directed=True)
        qubit_pairs = list(graph.edges)
        for pair in qubit_pairs:
            c.CX(pair[0], pair[1])

    return c


# For **exact** simulation, you can call `simulate` directly, providing the default `Config()`:

simple_circ = random_graph_circuit(n_qubits=10, edge_prob=0.1, layers=1)

with CuTensorNetHandle() as libhandle:
    my_ttn = simulate(libhandle, simple_circ, SimulationAlgorithm.TTNxGate, Config())

# ## Obtain an amplitude from a TTN
# Let's first see how to get the amplitude of the state `|10100>` from the output of the previous circuit.

state = int("10100", 2)
with CuTensorNetHandle() as libhandle:
    my_ttn.update_libhandle(libhandle)
    amplitude = my_ttn.get_amplitude(state)
print(amplitude)

# Since this is a very small circuit, we can use `pytket`'s state vector simulator capabilities to verify that the state is correct by checking the amplitude of each of the computational states.

state_vector = simple_circ.get_statevector()
n_qubits = len(simple_circ.qubits)

correct_amplitude = [False] * (2**n_qubits)
with CuTensorNetHandle() as libhandle:
    my_ttn.update_libhandle(libhandle)
    for i in range(2**n_qubits):
        correct_amplitude[i] = np.isclose(state_vector[i], my_ttn.get_amplitude(i))

print("Are all amplitudes correct?")
print(all(correct_amplitude))

# ## Sampling from a TTN
# Sampling and measurement from a TTN state is not currently supported. This will be added in an upcoming release.

# ## Approximate simulation
# We provide two policies for approximate simulation:
# * Bound the maximum value of the virtual bond dimension `chi`. If a bond dimension would increase past that point, we *truncate* (i.e. discard) the degrees of freedom that contribute the least to the state description. We can keep track of a lower bound of the error that this truncation causes.
# * Provide a value for acceptable two-qubit gate fidelity `truncation_fidelity`. After each two-qubit gate we truncate the dimension of virtual bonds as much as we can while guaranteeing the target gate fidelity. The more fidelity you require, the longer it will take to simulate. **Note**: this is *not* the final fidelity of the output state, but the fidelity per gate.
# Values for `chi` and `truncation_fidelity` can be set via `Config`. To showcase approximate simulation, let's define a circuit where exact TTN contraction would not be enough.

circuit = random_graph_circuit(n_qubits=30, edge_prob=0.1, layers=1)

# We can simulate it using bounded `chi` as follows:

start = time()
with CuTensorNetHandle() as libhandle:
    config = Config(chi=64, float_precision=np.float32)
    bound_chi_ttn = simulate(libhandle, circuit, SimulationAlgorithm.TTNxGate, config)
end = time()
print("Time taken by approximate contraction with bounded chi:")
print(f"{round(end-start,2)} seconds")
print("\nLower bound of the fidelity:")
print(round(bound_chi_ttn.fidelity, 4))

# Alternatively, we can fix `truncation_fidelity` and let the bond dimension increase as necessary to satisfy it.

start = time()
with CuTensorNetHandle() as libhandle:
    config = Config(truncation_fidelity=0.99, float_precision=np.float32)
    fixed_fidelity_ttn = simulate(
        libhandle, circuit, SimulationAlgorithm.TTNxGate, config
    )
end = time()
print("Time taken by approximate contraction with fixed truncation fidelity:")
print(f"{round(end-start,2)} seconds")
print("\nLower bound of the fidelity:")
print(round(fixed_fidelity_ttn.fidelity, 4))

# ## Contraction algorithms

# We currently offer only one TTN-based simulation algorithm.
# * **TTNxGate**: Apply gates one by one to the TTN, canonicalising the TTN and truncating when necessary.

# ## Using the logger

# You can request a verbose log to be produced during simulation, by assigning the `loglevel` argument when creating a `Config` instance. Currently, two log levels are supported (other than default, which is silent):
# - `logging.INFO` will print information about progress percent, memory currently occupied by the TTN and current fidelity. Additionally, some high level information of the current stage of the simulation is provided.
# - `logging.DEBUG` provides all of the messages from the loglevel above plus detailed information of the current operation being carried out and the values of important variables.
# **Note**: Due to technical issues with the `logging` module and Jupyter notebooks we need to reload the `logging` module. When working with python scripts and command line, just doing `import logging` is enough.
