import random

import conftest
import numpy as np
import pytest
from sympy import Symbol

from pytket.circuit import Bit, Circuit, Qubit, ToffoliBox
from pytket.extensions.cutensornet.general_state import GeneralBraOpKet, GeneralState
from pytket.passes import CnXPairwiseDecomposition, DecomposeBoxes
from pytket.pauli import Pauli, QubitPauliString
from pytket.transform import Transform
from pytket.utils.operators import QubitPauliOperator


@pytest.mark.parametrize(
    "circname",
    [
        "q5_empty",
        "q8_empty",
        "q2_x0",
        "q2_x1",
        "q2_v0",
        "q2_x0cx01",
        "q2_x1cx10x1",
        "q2_x0cx01cx10",
        "q2_v0cx01cx10",
        "q2_hadamard_test",
        "q2_lcu1",
        "q2_lcu2",
        "q2_lcu3",
        "q3_v0cx02",
        "q3_cx01cz12x1rx0",
        "q3_toffoli_box_with_implicit_swaps",
        "q4_lcu1",
        "q4_multicontrols",
        "q4_with_creates",
        "q5_h0s1rz2ry3tk4tk13",
        "q5_line_circ_30_layers",
        "q6_qvol",
        "q8_x0h2v5z6",
    ],
)
def test_basic_circs_state(circname: str) -> None:
    circuit = getattr(conftest, circname)()
    sv_pytket = circuit.get_statevector()

    op = QubitPauliOperator(
        {
            QubitPauliString(dict.fromkeys(circuit.qubits, Pauli.I)): 1.0,
        }
    )

    with GeneralState(circuit) as state:
        sv = state.get_statevector()
        assert np.allclose(sv, sv_pytket, atol=1e-10)

        # Calculate the inner product as the expectation value
        # of the identity operator: <psi|psi> = <psi|I|psi>
        ovl = state.expectation_value(op)
        assert ovl == pytest.approx(1.0)

        # Check that all amplitudes agree
        for i in range(len(sv)):
            assert np.isclose(sv[i], state.get_amplitude(i))

    # Calculate the inner product again, using GeneralBraOpKet
    with GeneralBraOpKet(circuit, circuit) as braket:
        ovl = braket.contract()
        assert ovl == pytest.approx(1.0)


def test_sv_toffoli_box_with_implicit_swaps() -> None:
    # Using specific permutation here
    perm = {
        (False, False): (True, True),
        (False, True): (False, False),
        (True, False): (True, False),
        (True, True): (False, True),
    }

    # Create a circuit with more qubits and multiple applications of the permutation
    # above
    ket_circ = Circuit(3)

    # Create the circuit
    ket_circ.add_toffolibox(ToffoliBox(perm), [Qubit(0), Qubit(1)])  # type: ignore
    ket_circ.add_toffolibox(ToffoliBox(perm), [Qubit(1), Qubit(2)])  # type: ignore

    DecomposeBoxes().apply(ket_circ)
    CnXPairwiseDecomposition().apply(ket_circ)
    Transform.OptimiseCliffords().apply(ket_circ)

    # Convert and contract
    with GeneralState(ket_circ) as state:
        ket_net_vector = state.get_statevector()

    # Compare to pytket statevector
    ket_pytket_vector = ket_circ.get_statevector()

    assert np.allclose(ket_net_vector, ket_pytket_vector)


@pytest.mark.parametrize("n_qubits", [4, 5, 6])
def test_sv_generalised_toffoli_box(n_qubits: int) -> None:
    def to_bool_tuple(n_qubits: int, x: int) -> tuple:
        bool_list = []
        for i in reversed(range(n_qubits)):
            bool_list.append((x >> i) % 2 == 1)  # noqa: PERF401
        return tuple(bool_list)

    random.seed(1)

    # Generate a random permutation
    cycle = list(range(2**n_qubits))
    random.shuffle(cycle)

    perm = dict()  # noqa: C408
    for orig, dest in enumerate(cycle):
        perm[to_bool_tuple(n_qubits, orig)] = to_bool_tuple(n_qubits, dest)

    # Create a circuit implementing the permutation above
    ket_circ = ToffoliBox(perm).get_circuit()  # type: ignore

    DecomposeBoxes().apply(ket_circ)
    CnXPairwiseDecomposition().apply(ket_circ)
    Transform.OptimiseCliffords().apply(ket_circ)

    with GeneralState(ket_circ) as state:
        ket_net_vector = state.get_statevector()

    ket_pytket_vector = ket_circ.get_statevector()
    assert np.allclose(ket_net_vector, ket_pytket_vector)

    # Calculate the inner product as the expectation value
    # of the identity operator: <psi|psi> = <psi|I|psi>
    op = QubitPauliOperator(
        {
            QubitPauliString(dict.fromkeys(ket_circ.qubits, Pauli.I)): 1.0,
        }
    )

    with GeneralState(ket_circ) as state:
        ovl = state.expectation_value(op)
    assert ovl == pytest.approx(1.0)


@pytest.mark.parametrize(
    "circname",
    [
        "q5_empty",
        "q8_empty",
        "q2_x0",
        "q2_x1",
        "q2_v0",
        "q2_x0cx01",
        "q2_x1cx10x1",
        "q2_x0cx01cx10",
        "q2_v0cx01cx10",
        "q2_hadamard_test",
        "q2_lcu1",
        "q2_lcu2",
        "q2_lcu3",
        "q3_v0cx02",
        "q3_cx01cz12x1rx0",
        "q3_toffoli_box_with_implicit_swaps",
        "q4_lcu1",
        "q4_multicontrols",
        "q4_with_creates",
        "q5_h0s1rz2ry3tk4tk13",
        "q5_line_circ_30_layers",
        "q6_qvol",
        "q8_x0h2v5z6",
    ],
)
@pytest.mark.parametrize(
    "observable",
    [
        QubitPauliOperator(
            {
                QubitPauliString({Qubit(0): Pauli.I, Qubit(1): Pauli.X}): 1.0,
            }
        ),
        QubitPauliOperator(
            {
                QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.Y}): 3.5 + 0.3j,
            }
        ),
        QubitPauliOperator(
            {
                QubitPauliString({Qubit(0): Pauli.Z}): 0.25,
                QubitPauliString({Qubit(1): Pauli.Y}): 0.33j,
                QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X}): 0.42 + 0.1j,
            }
        ),
    ],
)
def test_expectation_value(circname: str, observable: QubitPauliOperator) -> None:
    circuit = getattr(conftest, circname)()
    # Note: not all qubits are acted on by the observable. The remaining qubits are
    # interpreted to have I (identity) operators on them both by pytket and cutensornet.
    exp_val_tket = observable.state_expectation(circuit.get_statevector())

    # Calculate using GeneralState
    with GeneralState(circuit) as state:
        exp_val = state.expectation_value(observable)

    assert np.isclose(exp_val, exp_val_tket)

    # Calculate using GeneralBraOpKet
    with GeneralBraOpKet(circuit, circuit) as braket:
        exp_val = braket.contract(observable)

    assert np.isclose(exp_val, exp_val_tket)


@pytest.mark.parametrize(
    "circname",
    [
        "q5_empty",
        "q8_empty",
        "q2_x0",
        "q2_x1",
        "q2_v0",
        "q2_x0cx01",
        "q2_x1cx10x1",
        "q2_x0cx01cx10",
        "q2_v0cx01cx10",
        "q2_hadamard_test",
        "q2_lcu1",
        "q2_lcu2",
        "q2_lcu3",
        "q3_v0cx02",
        "q3_cx01cz12x1rx0",
        "q3_toffoli_box_with_implicit_swaps",
        "q4_lcu1",
        "q4_multicontrols",
        "q4_with_creates",
        "q5_h0s1rz2ry3tk4tk13",
        "q5_line_circ_30_layers",
        "q6_qvol",
        "q8_x0h2v5z6",
    ],
)
@pytest.mark.parametrize("measure_all", [True, False])  # Measure all or a subset
def test_sampler(circname: str, measure_all: bool) -> None:
    circuit = getattr(conftest, circname)()
    n_shots = 100000

    # Get the statevector so that we can calculate theoretical probabilities
    sv_pytket = circuit.get_statevector()

    # Add measurements to qubits
    if measure_all:  # noqa: SIM108
        num_measured = circuit.n_qubits
    else:
        num_measured = circuit.n_qubits // 2

    for i, q in enumerate(circuit.qubits):
        if i < num_measured:  # Skip the least significant qubits
            circuit.add_bit(Bit(i))
            circuit.Measure(q, Bit(i))

    # Sample using our library
    with GeneralState(circuit) as state:
        results = state.sample(n_shots)

    # Verify distribution matches theoretical probabilities
    for bit_tuple, count in results.get_counts().items():
        # Convert bitstring (Tuple[int,...]) to integer base 10
        outcome = sum(bit << i for i, bit in enumerate(reversed(bit_tuple)))

        # Calculate the theoretical probabilities
        if measure_all:
            prob = abs(sv_pytket[outcome]) ** 2
        else:
            # Obtain all compatible basis states (bitstring encoded as int)
            non_measured = circuit.n_qubits - num_measured
            compatible = [
                (outcome << non_measured) + offset for offset in range(2**non_measured)
            ]
            # The probability is the sum of that of all compatible basis states
            prob = sum(abs(sv_pytket[v]) ** 2 for v in compatible)

        assert np.isclose(count / n_shots, prob, atol=0.01)


@pytest.mark.parametrize(
    "circname",
    [
        "q4_lcu1_parameterised",
        "q5_h0s1rz2ry3tk4tk13_parameterised",
    ],
)
@pytest.mark.parametrize(
    "symbol_map",
    [
        {Symbol("a"): 0.3, Symbol("b"): 0.42, Symbol("c"): -0.13},
        {Symbol("a"): 5.3, Symbol("b"): 1.42, Symbol("c"): -0.07, Symbol("d"): 0.53},
    ],
)
def test_parameterised(circname: str, symbol_map: dict[Symbol, float]) -> None:
    circuit = getattr(conftest, circname)()
    state = GeneralState(circuit)
    sv = state.get_statevector(symbol_map)

    circuit.symbol_substitution(symbol_map)
    sv_pytket = circuit.get_statevector()
    assert np.allclose(sv, sv_pytket, atol=1e-10)

    op = QubitPauliOperator(
        {
            QubitPauliString(dict.fromkeys(circuit.qubits, Pauli.I)): 1.0,
        }
    )

    # Calculate the inner product as the expectation value
    # of the identity operator: <psi|psi> = <psi|I|psi>
    with GeneralState(circuit) as state:
        ovl = state.expectation_value(op)
        assert ovl == pytest.approx(1.0)

        # Check that all amplitudes agree
        for i in range(len(sv)):
            assert np.isclose(sv[i], state.get_amplitude(i))

    # Calculate the inner product again, using GeneralBraOpKet
    with GeneralBraOpKet(circuit, circuit) as braket:
        ovl = braket.contract()
    assert ovl == pytest.approx(1.0)
