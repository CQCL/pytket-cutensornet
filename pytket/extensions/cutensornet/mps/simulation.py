from typing import Any
from enum import Enum
from random import choice  # type: ignore

from pytket.circuit import Circuit, Command, Qubit, Op, OpType  # type: ignore
from pytket.transform import Transform  # type: ignore
from pytket.architecture import Architecture  # type: ignore
from pytket.passes import DefaultMappingPass  # type: ignore
from pytket.predicates import CompilationUnit  # type: ignore

from .mps import MPS
from .mps_gate import MPSxGate
from .mps_mpo import MPSxMPO


class ContractionAlg(Enum):
    """An enum to refer to the MPS contraction algorithm. Each enum value corresponds
    to the class with the same name; see its docs for information of the algorithm.
    """

    MPSxGate = 0
    MPSxMPO = 1


def simulate(circuit: Circuit, algorithm: ContractionAlg, **kwargs: Any) -> MPS:
    """Simulate the given circuit and return the ``MPS`` representing the final state.

    Note:
        The input ``circuit`` must be composed of one-qubit and two-qubit gates only.
        Any gateset supported by ``pytket`` can be used.

        This method will add SWAP gates to the circuit as appropriate to guarantee
        that all two-qubit gates act between nearest-neighbours in a line. If you
        wish to retrieve the circuit after this pass, use ``prepare_circuit()``.

    Args:
        circuit: The pytket circuit to be simulated.
        algorithm: Choose between the values of the ``ContractionAlg`` enum.
        **kwargs: Any argument accepted by the initialisers of the chosen
            ``algorithm`` class can be passed as a keyword argument. See the
            documentation of the corresponding class for details.

    Returns:
        An instance of ``MPS`` containing (an approximation of) the final state
        of the circuit.
    """

    prep_circ, qubit_map = prepare_circuit(circuit)

    chi = kwargs.get("chi", None)
    truncation_fidelity = kwargs.get("truncation_fidelity", None)
    float_precision = kwargs.get("float_precision", None)
    device_id = kwargs.get("device_id", None)

    if algorithm == ContractionAlg.MPSxGate:
        mps = MPSxGate(  # type: ignore
            prep_circ.qubits, chi, truncation_fidelity, float_precision, device_id
        )
    elif algorithm == ContractionAlg.MPSxMPO:
        k = kwargs.get("k", None)
        mps = MPSxMPO(  # type: ignore
            prep_circ.qubits, chi, truncation_fidelity, k, float_precision, device_id
        )

    # Sort the gates so there isn't much overhead from canonicalising back and forth.
    sorted_gates = get_sorted_gates(prep_circ)

    # Apply the gates
    with mps.init_cutensornet():
        for g in sorted_gates:
            mps.apply_gate(g)

    # Finally, restore the original name of the qubits
    mps.qubit_position = {qubit_map[q]: i for q, i in mps.qubit_position.items()}

    return mps


def get_amplitude(mps: MPS, state: int) -> complex:
    """Return the amplitude of the chosen computational state.

    Args:
        mps: The MPS to get the amplitude from.
        state: The integer whose bitstring describes the computational state.
            The qubits in the bitstring are ordered in increasing lexicographic order.

    Returns:
        The amplitude of the computational state in the MPS.
    """
    if mps._libhandle is None:
        raise RuntimeError("Must be called inside a with mps.init_cutensornet() block.")

    mps_qubits = list(mps.qubit_position.keys())
    bra_mps = MPSxGate(mps_qubits)
    bra_mps._libhandle = mps._libhandle

    ilo_qubits = sorted(mps_qubits)
    for i, q in enumerate(ilo_qubits):
        if state & 2 ** (len(mps_qubits) - i - 1):
            pos = bra_mps.qubit_position[q]
            bra_mps._apply_1q_gate(pos, Op.create(OpType.X))
    return bra_mps.vdot(mps)


def prepare_circuit(circuit: Circuit) -> tuple[Circuit, dict[Qubit, Qubit]]:
    """Return an equivalent circuit with the appropriate structure to be simulated by
    an ``MPS`` algorithm.

    Args:
        circuit: The circuit to be simulated.

    Returns:
        A tuple with an equivalent circuit with the appropriate structure. And a
        map of qubit names at the end of the circuit to their corresponding
        original names.
    """
    prep_circ = circuit.copy()

    # Implement it in a line architecture
    cu = CompilationUnit(prep_circ)
    architecture = Architecture([(i, i + 1) for i in range(prep_circ.n_qubits - 1)])
    DefaultMappingPass(architecture).apply(cu)
    prep_circ = cu.circuit
    Transform.DecomposeBRIDGE().apply(prep_circ)

    qubit_map = {arch_q: orig_q for orig_q, arch_q in cu.final_map.items()}

    return (prep_circ, qubit_map)


def get_sorted_gates(circuit: Circuit) -> list[Command]:
    """Sort the list of gates so that we obtain an equivalent circuit where we apply
    2-qubit gates that are close to each other first. This reduces the overhead of
    canonicalisation of the MPS, since we try to apply as many gates as we can on one
    end of the MPS before we go to the other end.

    Args:
        circuit: The original circuit.

    Returns:
        The same gates, ordered in a beneficial way.
    """
    all_gates = circuit.get_commands()
    sorted_gates = []
    # Keep track of the qubit at the center of the canonical form; start arbitrarily
    current_qubit = circuit.qubits[0]
    # Entries from `all_gates` that are not yet in `sorted_gates`
    remaining = set(range(len(all_gates)))

    # Create the list of indices of gates acting on each qubit
    gate_indices: dict[Qubit, list[int]] = {q: [] for q in circuit.qubits}
    for i, g in enumerate(all_gates):
        for q in g.qubits:
            gate_indices[q].append(i)
    # Apply all 1-qubit gates at the beginning of the circuit
    for q, indices in gate_indices.items():
        while indices and len(all_gates[indices[0]].qubits) == 1:
            i = indices.pop(0)
            sorted_gates.append(all_gates[i])
            remaining.remove(i)
    # Decide which 2-qubit gate to apply next
    while remaining:
        q_index = circuit.qubits.index(current_qubit)
        # Find distance from q_index to first qubit with an applicable 2-qubit gate
        left_distance = None
        prev_q = current_qubit
        for i, q in enumerate(reversed(circuit.qubits[:q_index])):
            if (
                gate_indices[prev_q]
                and gate_indices[q]
                and gate_indices[prev_q][0] == gate_indices[q][0]
            ):
                left_distance = i
                break
            prev_q = q
        right_distance = None
        prev_q = current_qubit
        for i, q in enumerate(circuit.qubits[q_index + 1 :]):
            if (
                gate_indices[prev_q]
                and gate_indices[q]
                and gate_indices[prev_q][0] == gate_indices[q][0]
            ):
                right_distance = i
                break
            prev_q = q
        # Choose the shortest distance
        if left_distance is None:
            current_qubit = circuit.qubits[q_index + right_distance]
        elif right_distance is None:
            current_qubit = circuit.qubits[q_index - left_distance]
        elif left_distance < right_distance:
            current_qubit = circuit.qubits[q_index - left_distance]
        elif left_distance > right_distance:
            current_qubit = circuit.qubits[q_index + right_distance]
        else:
            current_qubit = circuit.qubits[
                q_index + choice([-left_distance, right_distance])
            ]
        # Apply the gate
        i = gate_indices[current_qubit][0]
        next_gate = all_gates[i]
        sorted_gates.append(next_gate)
        remaining.remove(i)
        # Apply all 1-qubit gates after this gate
        for q in next_gate.qubits:
            gate_indices[q].pop(0)  # Remove the 2-qubit gate `next_gate`
            indices = gate_indices[q]
            while indices and len(all_gates[indices[0]].qubits) == 1:
                i = indices.pop(0)
                sorted_gates.append(all_gates[i])
                remaining.remove(i)

    assert len(all_gates) == len(sorted_gates)
    return sorted_gates
