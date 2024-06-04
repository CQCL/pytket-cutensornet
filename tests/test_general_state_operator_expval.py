import cmath
import random
import numpy as np
import pytest
from pytket.circuit import ToffoliBox, Qubit
from pytket.passes import DecomposeBoxes, CnXPairwiseDecomposition
from pytket.transform import Transform
from pytket.pauli import QubitPauliString, Pauli
from pytket.utils.operators import QubitPauliOperator
from pytket.circuit import Circuit
from pytket.extensions.cutensornet.general_state import (
    GeneralState,
    GeneralOperator,
    GeneralExpectationValue,
)
from pytket.extensions.cutensornet.structured_state import CuTensorNetHandle


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01"),  # type: ignore
        pytest.lazy_fixture("q2_x1cx10x1"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_v0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q2_lcu1"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q2_lcu3"),  # type: ignore
        pytest.lazy_fixture("q3_v0cx02"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q4_lcu1"),  # type: ignore
        pytest.lazy_fixture("q4_multicontrols"),  # type: ignore
    ],
)
def test_convert_statevec_ovl(circuit: Circuit) -> None:
    with CuTensorNetHandle() as libhandle:
        state = GeneralState(circuit, libhandle)
        sv = state.configure().prepare().compute()
        state.destroy()
    sv_pytket = np.array([circuit.get_statevector()])
    assert np.allclose(sv.round(10), sv_pytket.round(10))

    op = QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.I, Qubit(1): Pauli.I}): 1.0,
        }
    )
    with CuTensorNetHandle() as libhandle:
        state = GeneralState(circuit, libhandle)
        oper = GeneralOperator(op, 2, libhandle)
        ev = GeneralExpectationValue(state, oper, libhandle)
        ovl, state_norm = ev.configure().prepare().compute()
        ev.destroy()
        oper.destroy()
        state.destroy()
    assert ovl == pytest.approx(1.0)
    assert state_norm == pytest.approx(1.0)


def test_toffoli_box_with_implicit_swaps() -> None:
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
    with CuTensorNetHandle() as libhandle:
        state = GeneralState(ket_circ, libhandle)
        ket_net_vector = state.configure().prepare().compute()
        state.destroy()

    # Apply phase
    ket_net_vector = ket_net_vector * cmath.exp(1j * cmath.pi * ket_circ.phase)

    # Compare to pytket statevector
    ket_pytket_vector = ket_circ.get_statevector()

    assert np.allclose(ket_net_vector, ket_pytket_vector)


@pytest.mark.parametrize("n_qubits", [4, 5, 6])
def test_generalised_toffoli_box(n_qubits: int) -> None:
    def to_bool_tuple(n_qubits: int, x: int) -> tuple:
        bool_list = []
        for i in reversed(range(n_qubits)):
            bool_list.append((x >> i) % 2 == 1)
        return tuple(bool_list)

    random.seed(1)

    # Generate a random permutation
    cycle = list(range(2**n_qubits))
    random.shuffle(cycle)

    perm = dict()
    for orig, dest in enumerate(cycle):
        perm[to_bool_tuple(n_qubits, orig)] = to_bool_tuple(n_qubits, dest)

    # Create a circuit implementing the permutation above
    ket_circ = ToffoliBox(perm).get_circuit()  # type: ignore

    DecomposeBoxes().apply(ket_circ)
    CnXPairwiseDecomposition().apply(ket_circ)
    Transform.OptimiseCliffords().apply(ket_circ)

    # The ideal outcome on ket 0 input
    output = perm[(False,) * n_qubits]
    # A trivial circuit generating this state
    bra_circ = Circuit()
    for q in ket_circ.qubits:
        bra_circ.add_qubit(q)
    for i, bit in enumerate(output):
        if bit:
            bra_circ.X(i)

    with CuTensorNetHandle() as libhandle:
        state = GeneralState(ket_circ, libhandle)
        ket_net_vector = state.configure().prepare().compute()
        state.destroy()
    ket_net_vector = ket_net_vector * cmath.exp(1j * cmath.pi * ket_circ.phase)
    ket_pytket_vector = ket_circ.get_statevector()
    assert np.allclose(ket_net_vector, ket_pytket_vector)

    with CuTensorNetHandle() as libhandle:
        state = GeneralState(bra_circ, libhandle)
        bra_net_vector = state.configure().prepare().compute()
        state.destroy()
    bra_net_vector = bra_net_vector * cmath.exp(1j * cmath.pi * bra_circ.phase)
    bra_pytket_vector = bra_circ.get_statevector()
    assert np.allclose(bra_net_vector, bra_pytket_vector)

    op = QubitPauliOperator(
        {
            QubitPauliString({Qubit(i): Pauli.I for i in range(n_qubits)}): 1.0,
        }
    )

    with CuTensorNetHandle() as libhandle:
        state = GeneralState(ket_circ, libhandle)
        oper = GeneralOperator(op, n_qubits, libhandle)
        ev = GeneralExpectationValue(state, oper, libhandle)
        ovl, state_norm = ev.configure().prepare().compute()
        ev.destroy()
        oper.destroy()
        state.destroy()
    assert ovl == pytest.approx(1.0)
    assert state_norm == pytest.approx(1.0)