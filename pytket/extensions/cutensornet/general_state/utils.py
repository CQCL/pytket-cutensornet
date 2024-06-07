# Copyright 2019-2024 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy.typing import NDArray
from pytket.backends.backendresult import BackendResult
from pytket.circuit import Qubit, Circuit


def _reorder_qlist(
    post_select_dict: dict, qlist: list[Qubit]
) -> tuple[list[Qubit], Qubit]:
    """Reorder qlist so that post_select_qubit is first in the list.

    Args:
        post_select_dict (dict): Dictionary of post selection qubit and value
        qlist (list): List of qubits

    Returns:
        Tuple containing a list of qubits reordered so that `post_select_qubit` is first
        in the list, and the post select qubit.
    """

    post_select_q = list(post_select_dict.keys())[0]

    pop_i = qlist.index(post_select_q)

    q = qlist.pop(pop_i)

    q_list_reordered = [q]
    q_list_reordered.extend(qlist)

    return q_list_reordered, q


def statevector_postselect(
    qlist: list[Qubit], sv: NDArray, post_select_dict: dict[Qubit, int]
) -> NDArray:
    """Post selects a statevector.

    Recursively calls itself if there are multiple post select qubits.
    Uses backend result to get statevecto and permutes so the the post select qubit for
    each iteration is first in the list.

    Args:
        qlist: List of qubits.
        sv: Statevector.
        post_select_dict: Dictionary of post selection qubit and value.

    Returns:
        Post selected statevector.
    """

    n = len(qlist)
    n_p = len(post_select_dict)

    b_res = BackendResult(state=sv, q_bits=qlist)

    q_list_reordered, q = _reorder_qlist(post_select_dict, qlist)

    sv = b_res.get_state(qbits=q_list_reordered)

    if post_select_dict[q] == 0:
        new_sv = sv[: 2 ** (n - 1) :]
    elif post_select_dict[q] == 1:
        new_sv = sv[2 ** (n - 1) :]
    else:
        raise ValueError("post_select_dict[q] must be 0 or 1")

    if n_p == 1:
        return new_sv

    post_select_dict.pop(q)
    q_list_reordered.pop(0)

    return statevector_postselect(q_list_reordered, new_sv, post_select_dict)


def circuit_statevector_postselect(
    circ: Circuit, post_select_dict: dict[Qubit, int]
) -> NDArray:
    """Post selects a circuit statevector. recursively calls
    itself if there are multiple post select qubits. Should only be
    used for testing small circuits as it uses the circuit.get_unitary() method.

    Args:
        circ: Circuit.
        post_select_dict: Dictionary of post selection qubit and value.

    Returns:
        Post selected statevector.
    """

    return statevector_postselect(
        circ.qubits, circ.get_statevector(), post_select_dict
    )  # TODO this does not account for global phase if just taking circuit
