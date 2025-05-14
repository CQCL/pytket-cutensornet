# # `GeneralState` Tutorial

import numpy as np
from sympy import Symbol
from scipy.stats import unitary_group  # type: ignore
from pytket.circuit import Circuit, OpType, Unitary2qBox, Qubit, Bit
from pytket.passes import DecomposeBoxes
from pytket.utils import QubitPauliOperator
from pytket._tket.pauli import Pauli, QubitPauliString
from pytket.circuit.display import render_circuit_jupyter

from pytket.extensions.cutensornet.general_state import (
    GeneralState,
    GeneralBraOpKet,
)
from pytket.extensions.cutensornet.backends import CuTensorNetShotsBackend

# ## Introduction
# This notebook is a guide on how to use the features provided in the `general_state` submodule of pytket-cutensornet. This submodule is a thin wrapper of CuTensorNet's `NetworkState`, allowing users to convert pytket circuits into tensor networks and use CuTensorNet's contraction path optimisation algorithm.
# All simulations realised with this submodule are *exact*. Once the pytket circuit has been converted to a tensor network, the computation has two steps:
#   1. *Contraction path optimisation*. Attempts to find an order of contracting pairs of tensors in which the the total number of FLOPs is minimised. No operation on the tensor network occurs at this point. Runs on CPU.
#   2. *Tensor network contraction*. Uses the ordering of contractions found in the previous step evaluate the tensor network. Runs on GPU.
#
# **Reference**: The original contraction path optimisation algorithm that NVIDIA implemented on CuTensorNet: https://arxiv.org/abs/2002.01935

# ## `GeneralState`
# The class `GeneralState` is used to convert a circuit into a tensor network and query information from the final state. Let's walk through a simple example.

my_circ = Circuit(5)
my_circ.CX(3, 4)
my_circ.H(2)
my_circ.CZ(0, 1)
my_circ.ZZPhase(0.1, 4, 3)
my_circ.TK2(0.3, 0.5, 0.7, 2, 1)
my_circ.Ry(0.2, 0)
my_circ.measure_all()

render_circuit_jupyter(my_circ)

# The first step is to convert our pytket circuit into a tensor network. This is straightforward:
tn_state = GeneralState(my_circ)

# The variable `tn_state` now holds a tensor network representation of `my_circ`.
# **Note**: Circuits must not have mid-circuit measurements or classical logic. The measurements at the end of the circuit are stripped and only considered when calling `tn_state.sample(n_shots)`.
# We can now query information from the state. For instance, let's calculate the probability of in the qubits 0 and 3 agreeing in their outcome.

# First, let's generate `|x>` computational basis states where `q[0]` and `q[3]` agree on their values. We can do this with some bitwise operators and list comprehension.
# **Note**: Remember that pytket uses "increasing lexicographic order" (ILO) for qubits, so `q[0]` is the most significant bit.
selected_states = [
    x
    for x in range(2**my_circ.n_qubits)
    if (  # Iterate over all possible states
        x & int("10000", 2) == 0
        and x & int("00010", 2) == 0  # both qubits are 0 or...
        or x & int("10000", 2) != 0
        and x & int("00010", 2) != 0  # both qubits are 1
    )
]

# We can now query the amplitude of all of these states and calculate the probability by summing their squared absolute values.
amplitudes = []
for x in selected_states:
    amplitudes.append(tn_state.get_amplitude(x))
probability = sum(abs(a) ** 2 for a in amplitudes)
print(f"Probability: {probability}")

# Of course, calculating probabilities by considering the amplitudes of all relevant states is not efficient in general, since we may need to calculate a number of amplitudes that scales exponentially with the number of qubits. An alternative is to use expectation values. In particular, all of the states in `selected_states` are +1 eigenvectors of the `ZIIZI` observable and, hence, we can calculate the probability `p` by solving the equation `<ZIIZI> = (+1)p + (-1)(1-p)` using the fact that `ZIIZI` only has +1 and -1 eigenvalues.
string_ZIIZI = QubitPauliString(
    my_circ.qubits, [Pauli.Z, Pauli.I, Pauli.I, Pauli.Z, Pauli.I]
)
observable = QubitPauliOperator({string_ZIIZI: 1.0})
expectation_val = tn_state.expectation_value(observable).real
exp_probability = (expectation_val + 1) / 2
assert np.isclose(probability, exp_probability, atol=0.0001)
print(f"Probability: {exp_probability}")

# Alternatively, we can estimate the probability by sampling.
n_shots = 100000
outcomes = tn_state.sample(n_shots)
hit_count = 0
for bit_tuple, count in outcomes.get_counts().items():
    if bit_tuple[0] == bit_tuple[3]:
        hit_count += count
samp_probability = hit_count / n_shots
assert np.isclose(probability, samp_probability, atol=0.01)
print(f"Probability: {samp_probability}")

# When we finish doing computations with the `tn_state` we must destroy it to free GPU memory.
tn_state.destroy()

# To avoid forgetting this final step, we recommend users call `GeneralState` (and `GeneralBraOpKet`) as context managers:
with GeneralState(my_circ) as my_state:
    expectation_val = my_state.expectation_value(observable)
print(expectation_val)

# Using this syntax, `my_state` is automatically destroyed when the code exists the `with ...` block.

# ## Parameterised circuits
# Circuits that only differ on the parameters of their gates have the same tensor network topology and, hence, we may use the same contraction path for all of them.
a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
param_circ1 = Circuit(5)
param_circ1.Ry(a, 3).Ry(0.27, 4).CX(4, 3).Ry(b, 2).Ry(0.21, 3)
param_circ1.Ry(0.12, 0).Ry(a, 1)
param_circ1.add_gate(OpType.CnX, [0, 1, 4]).add_gate(OpType.CnX, [4, 1, 3])
param_circ1.X(0).X(1).add_gate(OpType.CnY, [0, 1, 2]).add_gate(OpType.CnY, [0, 4, 3]).X(
    0
).X(1)
param_circ1.Ry(-b, 0).Ry(-c, 1)
render_circuit_jupyter(param_circ1)

# We can pass a parameterised circuit to `GeneralState`. The value of the parameters is provided when calling methods of `GeneralState`. The contraction path is automatically reused on different calls to the same method.
n_circs = 5
with GeneralState(param_circ1) as param_state:
    for i in range(n_circs):
        symbol_map = {s: np.random.random() for s in [a, b, c]}
        exp_val = param_state.expectation_value(observable, symbol_map=symbol_map)
        print(f"Expectation value for circuit {i}: {exp_val.real}")


# ## `GeneralBraOpKet`
# The `GeneralBraOpKet` can be used to calculate any number that can be represented as the result of some `<bra|op|ket>` where `|bra>` and `|ket>` are the final states of pytket circuits, and `op` is a `QubitPauliOperator`. The circuits for `|bra>` and `|ket>` need not be the same.
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
param_circ2 = Circuit(5)
param_circ2.H(0)
param_circ2.S(1)
param_circ2.Rz(x * z, 2)
param_circ2.Ry(y + x, 3)
param_circ2.TK1(x, y, z, 4)
param_circ2.TK2(z - y, z - x, (x + y) * z, 1, 3)
symbol_map = {a: 2.1, b: 1.3, c: 0.7, x: 3.0, y: 1.6, z: -8.3}

# We can calculate inner products by providing no `op`:
with GeneralBraOpKet(bra=param_circ2, ket=param_circ1) as braket:
    inner_prod = braket.contract(symbol_map=symbol_map)
with GeneralBraOpKet(bra=param_circ1, ket=param_circ2) as braket:
    inner_prod_conj = braket.contract(symbol_map=symbol_map)
assert np.isclose(np.conj(inner_prod), inner_prod_conj)
print(f"<circ_b|circ_a> = {inner_prod}")
print(f"<circ_a|circ_b> = {inner_prod_conj}")

# And we are not constrained to Hermitian operators:
string_XZIXX = QubitPauliString(
    param_circ2.qubits, [Pauli.X, Pauli.Z, Pauli.I, Pauli.X, Pauli.X]
)
string_IZZYX = QubitPauliString(
    param_circ2.qubits, [Pauli.I, Pauli.Z, Pauli.Z, Pauli.Y, Pauli.X]
)
string_ZIZXY = QubitPauliString(
    param_circ2.qubits, [Pauli.Z, Pauli.I, Pauli.Z, Pauli.X, Pauli.Y]
)
operator = QubitPauliOperator(
    {string_XZIXX: -1.38j, string_IZZYX: 2.36, string_ZIZXY: 0.42j + 0.3}
)
with GeneralBraOpKet(bra=param_circ2, ket=param_circ1) as braket:
    value = braket.contract(operator, symbol_map=symbol_map)
print(value)

# ## Backends
# We provide a pytket `Backend` to obtain shots using `GeneralState`.


# Let's consider a more challenging circuit
def random_circuit(n_qubits: int, n_layers: int) -> Circuit:
    """Random quantum volume circuit."""
    c = Circuit(n_qubits, n_qubits)

    for _ in range(n_layers):
        qubits = np.random.permutation([i for i in range(n_qubits)])
        qubit_pairs = [[qubits[i], qubits[i + 1]] for i in range(0, n_qubits - 1, 2)]

        for pair in qubit_pairs:
            # Generate random 4x4 unitary matrix.
            SU4 = unitary_group.rvs(4)  # random unitary in SU4
            SU4 = SU4 / (np.linalg.det(SU4) ** 0.25)
            SU4 = np.matrix(SU4)
            c.add_unitary2qbox(Unitary2qBox(SU4), *pair)

    DecomposeBoxes().apply(c)
    return c


# Let's measure only three of the qubits.
# **Note**: The complexity of this simulation increases exponentially with the number of qubits measured. Other factors leading to intractability are circuit depth and qubit connectivity.
n_shots = 1000
quantum_vol_circ = random_circuit(n_qubits=40, n_layers=5)
quantum_vol_circ.Measure(Qubit(0), Bit(0))
quantum_vol_circ.Measure(Qubit(1), Bit(1))
quantum_vol_circ.Measure(Qubit(2), Bit(2))

# The `CuTensorNetShotsBackend` is used in the same way as any other pytket `Backend`.
backend = CuTensorNetShotsBackend()
compiled_circ = backend.get_compiled_circuit(quantum_vol_circ)
results = backend.run_circuit(compiled_circ, n_shots=n_shots)
print(results.get_counts())
