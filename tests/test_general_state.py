import random
import numpy as np
import pytest
from pytket.circuit import ToffoliBox, Qubit
from pytket.passes import DecomposeBoxes, CnXPairwiseDecomposition
from pytket.transform import Transform
from pytket.pauli import QubitPauliString, Pauli
from pytket.utils.operators import QubitPauliOperator
from pytket.circuit import Circuit
from pytket.extensions.cutensornet.general_state import GeneralState
from pytket.extensions.cutensornet.structured_state import CuTensorNetHandle


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q8_empty"),  # type: ignore
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
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q4_lcu1"),  # type: ignore
        pytest.lazy_fixture("q4_multicontrols"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
        pytest.lazy_fixture("q8_x0h2v5z6"),  # type: ignore
    ],
)
def test_get_statevec(circuit: Circuit) -> None:
    with CuTensorNetHandle() as libhandle:
        state = GeneralState(circuit, libhandle)
        sv = state.get_statevector()

        sv_pytket = circuit.get_statevector()
        assert np.allclose(sv, sv_pytket, atol=1e-10)

        op = QubitPauliOperator(
            {
                QubitPauliString({q: Pauli.I for q in circuit.qubits}): 1.0,
            }
        )

        # Calculate the inner product as the expectation value
        # of the identity operator: <psi|psi> = <psi|I|psi>
        state = GeneralState(circuit, libhandle)
        ovl = state.expectation_value(op)
        assert ovl == pytest.approx(1.0)

    state.destroy()


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
    with CuTensorNetHandle() as libhandle:
        state = GeneralState(ket_circ, libhandle)
        ket_net_vector = state.get_statevector()
    state.destroy()

    # Compare to pytket statevector
    ket_pytket_vector = ket_circ.get_statevector()

    assert np.allclose(ket_net_vector, ket_pytket_vector)


@pytest.mark.parametrize("n_qubits", [4, 5, 6])
def test_sv_generalised_toffoli_box(n_qubits: int) -> None:
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

    with CuTensorNetHandle() as libhandle:
        state = GeneralState(ket_circ, libhandle)
        ket_net_vector = state.get_statevector()

        ket_pytket_vector = ket_circ.get_statevector()
        assert np.allclose(ket_net_vector, ket_pytket_vector)

        # Calculate the inner product as the expectation value
        # of the identity operator: <psi|psi> = <psi|I|psi>
        op = QubitPauliOperator(
            {
                QubitPauliString({q: Pauli.I for q in ket_circ.qubits}): 1.0,
            }
        )

        state = GeneralState(ket_circ, libhandle)
        ovl = state.expectation_value(op)
        assert ovl == pytest.approx(1.0)

    state.destroy()


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q8_empty"),  # type: ignore
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
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q4_lcu1"),  # type: ignore
        pytest.lazy_fixture("q4_multicontrols"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
        pytest.lazy_fixture("q8_x0h2v5z6"),  # type: ignore
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
def test_expectation_value(circuit: Circuit, observable: QubitPauliOperator) -> None:
    # Note: not all qubits are acted on by the observable. The remaining qubits are
    # interpreted to have I (identity) operators on them both by pytket and cutensornet.
    exp_val_tket = observable.state_expectation(circuit.get_statevector())

    # Calculate using GeneralState
    with CuTensorNetHandle() as libhandle:
        state = GeneralState(circuit, libhandle)
        exp_val = state.expectation_value(observable)

    assert np.isclose(exp_val, exp_val_tket)
    state.destroy()


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q8_empty"),  # type: ignore
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
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q4_lcu1"),  # type: ignore
        pytest.lazy_fixture("q4_multicontrols"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
        pytest.lazy_fixture("q8_x0h2v5z6"),  # type: ignore
    ],
)
def test_sampler(circuit: Circuit) -> None:

    n_shots = 100000

    # Get the statevector so that we can calculate theoretical probabilities
    sv_pytket = circuit.get_statevector()

    with CuTensorNetHandle() as libhandle:
        state = GeneralState(circuit, libhandle)
        shots = state.sample(n_shots)

    for out, count in shots.counts().items():
        outcome = out.to_intlist()[0]  # Unpack from singleton OutcomeArray
        prob = abs(sv_pytket[outcome])**2  # Theoretical probability
        assert np.isclose(count / n_shots, prob, atol=0.01)

    state.destroy()
