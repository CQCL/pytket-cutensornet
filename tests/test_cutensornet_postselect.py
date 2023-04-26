from typing import List, Union
import numpy as np
from numpy.typing import NDArray
import pytest
from pytket.backends.backendresult import BackendResult
from pytket.circuit import Qubit, Circuit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator

from pytket.circuit import Circuit, Qubit
from pytket.backends.backendresult import BackendResult
import numpy.typing as npt
import numpy as np
import cuquantum as cq 

from pytket.extensions.cuquantum.tensor_network_convert import (  # type: ignore
    tk_to_tensor_network,
    TensorNetwork,
    measure_qubits_state
)

def _reorder_qlist(post_select_dict:dict, qlist:list[Qubit]):
    """Reorder qlist so that post_select_qubit is first in the list.

    Args:
        post_select_dict (dict): Dictionary of post selection qubit and value
        qlist (list): List of qubits

    Returns:
        tuple: Tuple containing: q_list_reordered (list): List of qubits reordered so that post_select_qubit is first in the list. q (Qubit): The post select qubit
    """

    post_select_q = list(post_select_dict.keys())[0]

    for i,q in enumerate(qlist):
        if q == post_select_q:
            pop_i = i
            break
        if i == len(qlist)-1:
            raise ValueError("post_select_q not in qlist")

    q = qlist.pop(pop_i)

    q_list_reordered = [q]
    
    q_list_reordered.extend(qlist)

    return q_list_reordered, q

def statevector_postselect(qlist: list[Qubit], sv: npt.NDArray, post_select_dict: dict[Qubit, int]):
    """Post selects a statevector. recursively calls itself if there are multiple post select qubits.
    Uses backend result to get statevector and permutes so the the post select qubit for each iteration is first in the list.

    Args:
        qlist (list): List of qubits
        sv (npt.NDArray): Statevector
        post_select_dict (dict): Dictionary of post selection qubit and value

    Returns:
        npt.NDArray: Post selected statevector
    """

    n = len(qlist)
    n_p = len(post_select_dict)

    b_res = BackendResult(state = sv, q_bits=qlist)

    q_list_reordered, q = _reorder_qlist(post_select_dict, qlist)

    sv = b_res.get_state(qbits=q_list_reordered)

    if post_select_dict[q] == 0:
        new_sv = sv[:2**(n-1):]
    elif post_select_dict[q] == 1:
        new_sv = sv[2**(n-1):]
    else:
        raise ValueError("post_select_dict[q] must be 0 or 1")

    if n_p == 1:
        return new_sv

    post_select_dict.pop(q)
    q_list_reordered.pop(0)

    return statevector_postselect(q_list_reordered, new_sv, post_select_dict)

def circuit_statevector_postselect(circ: Circuit, post_select_dict: dict[Qubit, int]):
    """Post selects a circuit statevector. recursively calls itself if there are multiple post select qubits.
    Should only be used for testing small circuits as it uses the circuit.get_unitary() method.
    
    Args:
        circ (Circuit): Circuit
        post_select_dict (dict): Dictionary of post selection qubit and value
        
    Returns:
        npt.NDArray: Post selected statevector
    """
    
    return statevector_postselect(circ.qubits, circ.get_statevector(), post_select_dict)

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
    ],
)

def test_measure_qubits_state_2q(circuit_2q: Circuit) -> None:

    measurement_dict = {Qubit('q',0) : 0}
    sv = circuit_statevector_postselect(circuit_2q, measurement_dict)
    tn = TensorNetwork(circuit_2q)
    ten_net = measure_qubits_state(tn, measurement_dict)
    result_cu = cq.contract(*ten_net.cuquantum_interleaved).flatten().round(10)
    assert np.allclose(result_cu, sv)

    measurement_dict = {Qubit('q',0) : 1}
    sv = circuit_statevector_postselect(circuit_2q, measurement_dict)
    tn = TensorNetwork(circuit_2q)
    ten_net = measure_qubits_state(tn, measurement_dict)
    result_cu = cq.contract(*ten_net.cuquantum_interleaved).flatten().round(10)
    assert np.allclose(result_cu, sv)

@pytest.mark.parametrize(
    "circuit_3q",
    [
        pytest.lazy_fixture("q3_v0cx02"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
    ],
)

def test_measure_qubits_state_3q(circuit_3q: Circuit) -> None:

    measurement_dict = {Qubit('q',0) : 0, Qubit('q',0) : 0}
    sv = circuit_statevector_postselect(circuit_3q, measurement_dict)
    tn = TensorNetwork(circuit_3q)
    ten_net = measure_qubits_state(tn, measurement_dict)
    result_cu = cq.contract(*ten_net.cuquantum_interleaved).flatten().round(10)
    assert np.allclose(result_cu, sv)

    measurement_dict = {Qubit('q',1) : 0, Qubit('q',1) : 0}
    sv = circuit_statevector_postselect(circuit_3q, measurement_dict)
    tn = TensorNetwork(circuit_3q)
    ten_net = measure_qubits_state(tn, measurement_dict)
    result_cu = cq.contract(*ten_net.cuquantum_interleaved).flatten().round(10)
    assert np.allclose(result_cu, sv)