from pytket.backends.backendresult import BackendResult
from pytket.circuit import Qubit, Circuit
from numpy.typing import NDArray

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

def statevector_postselect(qlist: list[Qubit], sv: NDArray, post_select_dict: dict[Qubit, int]):
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
    
    return statevector_postselect(circ.qubits, circ.get_statevector(), post_select_dict) #TODO this does not account for global phase if just taking circuit