import numpy as np
import cuquantum as cq  # type: ignore
import pytest
from pytket.circuit import Qubit, Circuit  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.utils import QubitPauliOperator
from pytket.extensions.cutensornet.backends import CuTensorNetStateBackend
from pytket.extensions.cutensornet.general_state.tensor_network_convert import (  # type: ignore
    TensorNetwork,
    measure_qubits_state,
    get_operator_expectation_value,
)
from pytket.extensions.cutensornet.general_state.utils import (
    circuit_statevector_postselect,
)


@pytest.mark.parametrize(
    "circuit_2q",
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
    ],
)
@pytest.mark.parametrize("postselect_dict", [{Qubit("q", 0): 0}, {Qubit("q", 0): 1}])
def test_postselect_qubits_state_2q(circuit_2q: Circuit, postselect_dict: dict) -> None:
    sv = circuit_statevector_postselect(circuit_2q, postselect_dict)
    tn = TensorNetwork(circuit_2q)
    ten_net = measure_qubits_state(tn, postselect_dict)
    result_cu = cq.contract(*ten_net.cuquantum_interleaved).flatten().round(10)
    assert np.allclose(result_cu, sv)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q3_v0cx02"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q4_lcu1"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "postselect_dict",
    [
        {Qubit("q", 0): 0, Qubit("q", 1): 0},
        {Qubit("q", 0): 1, Qubit("q", 1): 1},
        {Qubit("q", 0): 0, Qubit("q", 1): 1},
        {Qubit("q", 0): 1, Qubit("q", 1): 1},
    ],
)
def test_postselect_qubits_state(circuit: Circuit, postselect_dict: dict) -> None:
    sv = circuit_statevector_postselect(circuit, postselect_dict.copy())
    tn = TensorNetwork(circuit)
    ten_net = measure_qubits_state(tn, postselect_dict)
    result_cu = cq.contract(*ten_net.cuquantum_interleaved).flatten()
    assert np.allclose(result_cu, sv)


@pytest.mark.parametrize(
    "circuit_2q",
    [
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01"),  # type: ignore
        pytest.lazy_fixture("q2_x1cx10x1"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q2_lcu1"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q2_lcu3"),  # type: ignore
    ],
)
@pytest.mark.parametrize("postselect_dict", [{Qubit("q", 1): 0}, {Qubit("q", 1): 1}])
def test_expectation_value_postselect_2q(
    circuit_2q: Circuit, postselect_dict: dict
) -> None:
    op = QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.Z}): 1.0,
        }
    )
    op_matrix = op.to_sparse_matrix(1).todense()
    sv = np.array([circuit_statevector_postselect(circuit_2q, postselect_dict)]).T
    sv_exp = (sv.conj().T @ op_matrix @ sv)[0, 0]
    b = CuTensorNetStateBackend()
    c = b.get_compiled_circuit(circuit_2q)
    ten_exp = get_operator_expectation_value(c.copy(), op, postselect_dict)
    assert np.isclose(ten_exp, sv_exp)


@pytest.mark.parametrize(
    "circuit_lcu_4q",
    [
        pytest.lazy_fixture("q4_lcu1"),  # type: ignore
    ],
)
def test_expectation_value_postselect_4q_lcu(circuit_lcu_4q: Circuit) -> None:
    postselect_dict = {Qubit("q", 2): 0, Qubit("q", 3): 0}
    op = QubitPauliOperator(
        {
            QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.X}): 0.25,
        }
    )
    op_matrix = op.to_sparse_matrix(2).todense()
    sv = np.array(
        [circuit_statevector_postselect(circuit_lcu_4q, postselect_dict.copy())]
    ).T
    b = CuTensorNetStateBackend()
    c = b.get_compiled_circuit(circuit_lcu_4q)
    sv = sv * np.exp(1j * np.pi * c.phase)
    sv_exp = (sv.conj().T @ op_matrix @ sv)[0, 0]
    ten_exp = get_operator_expectation_value(c.copy(), op, postselect_dict)
    assert np.isclose(ten_exp, sv_exp)
