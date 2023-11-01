# Copyright 2019-2023 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
#     http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum

from random import choice  # type: ignore
from collections import defaultdict  # type: ignore
import numpy as np  # type: ignore

from networkx import Graph, community  # type: ignore

from pytket.circuit import Circuit, Command, Qubit
from pytket.transform import Transform
from pytket.architecture import Architecture
from pytket.passes import DefaultMappingPass
from pytket.predicates import CompilationUnit

from pytket.extensions.cutensornet.general import set_logger
from .general import CuTensorNetHandle, Config, TNState
from .mps_gate import MPSxGate
from .mps_mpo import MPSxMPO
from .ttn_gate import TTNxGate


class ContractionAlg(Enum):
    """An enum to refer to the TNState contraction algorithm.

    Each enum value corresponds to the class with the same name; see its docs for
    information about the algorithm.
    """

    TTNxGate = 0
    MPSxGate = 1
    MPSxMPO = 2


def simulate(
    libhandle: CuTensorNetHandle,
    circuit: Circuit,
    algorithm: ContractionAlg,
    config: Config,
) -> TNState:
    """Simulates the circuit and returns the ``TNState`` representing the final state.

    Note:
        A ``libhandle`` should be created via a ``with CuTensorNet() as libhandle:``
        statement. The device where the ``TNState`` is stored will match the one
        specified by the library handle.

        The input ``circuit`` must be composed of one-qubit and two-qubit gates only.
        Any gateset supported by ``pytket`` can be used.

    Args:
        libhandle: The cuTensorNet library handle that will be used to carry out
            tensor operations.
        circuit: The pytket circuit to be simulated.
        algorithm: Choose between the values of the ``ContractionAlg`` enum.
        config: The configuration object for simulation.

    Returns:
        An instance of ``TNState`` containing (an approximation of) the final state
        of the circuit. The instance be of the class matching ``algorithm``.
    """
    logger = set_logger("Simulation", level=config.loglevel)

    logger.info(
        "Ordering the gates in the circuit to reduce canonicalisation overhead."
    )
    if algorithm == ContractionAlg.MPSxGate:
        tnstate = MPSxGate(  # type: ignore
            libhandle,
            circuit.qubits,
            config,
        )
        sorted_gates = _get_sorted_gates(circuit)

    elif algorithm == ContractionAlg.MPSxMPO:
        tnstate = MPSxMPO(  # type: ignore
            libhandle,
            circuit.qubits,
            config,
        )
        sorted_gates = _get_sorted_gates(circuit)

    elif algorithm == ContractionAlg.TTNxGate:
        tnstate = TTNxGate(  # type: ignore
            libhandle,
            _get_qubit_partition(circuit, config.leaf_size),
            config,
        )
        sorted_gates = circuit.get_commands()  # TODO: change!

    logger.info("Running simulation...")
    # Apply the gates
    for i, g in enumerate(sorted_gates):
        tnstate.apply_gate(g)
        logger.info(f"Progress... {(100*i) // len(sorted_gates)}%")

    # Apply the batched operations that are left (if any)
    tnstate._flush()

    # Apply the circuit's phase to the state
    tnstate.apply_scalar(np.exp(1j * np.pi * circuit.phase))

    logger.info("Simulation completed.")
    logger.info(f"Final TNState size={tnstate.get_byte_size() / 2**20} MiB")
    logger.info(f"Final TNState fidelity={tnstate.fidelity}")
    return tnstate


def prepare_circuit_mps(circuit: Circuit) -> tuple[Circuit, dict[Qubit, Qubit]]:
    """Transpiles the circuit for it to be ``MPS``-friendly.

    Returns an equivalent circuit with the appropriate structure to be simulated by
    an ``MPS`` algorithm.

    Note:
        The qubits in the output circuit will be renamed. Implicit SWAPs may be added
        to the circuit, meaning that the logical qubit held at the ``node[i]`` qubit
        at the beginning of the circuit may differ from the one it holds at the end.

    Args:
        circuit: The circuit to be simulated.

    Returns:
        A tuple with an equivalent circuit with the appropriate structure and a
        map of qubit names at the end of the circuit to their corresponding
        original names.
    """

    # Implement it in a line architecture
    cu = CompilationUnit(circuit)
    architecture = Architecture([(i, i + 1) for i in range(circuit.n_qubits - 1)])
    DefaultMappingPass(architecture).apply(cu)
    prep_circ = cu.circuit
    Transform.DecomposeBRIDGE().apply(prep_circ)

    qubit_map: dict[Qubit, Qubit] = {}
    for orig_q, arch_q in cu.final_map.items():
        assert isinstance(orig_q, Qubit)
        assert isinstance(arch_q, Qubit)
        qubit_map[arch_q] = orig_q

    return (prep_circ, qubit_map)


def _get_qubit_partition(
    circuit: Circuit, max_q_per_leaf: int
) -> dict[int, list[Qubit]]:
    """Returns a qubit partition for a TTN.

    Proceeds by recursive bisection of the qubit connectivity graph, so that
    qubits that interact with each other less are connected by a common ancestor
    closer to the root.

    Args:
        circuit: The circuit to be simulated.
        max_q_per_leaf: The maximum allowed number of qubits per node leaf

    Returns:
        A dictionary describing the partition in the format expected by TTN.

    Raises:
        RuntimeError: If gate acts on more than 2 qubits.
    """

    # Scan the circuit and generate the edges of the connectivity graph
    edge_weights: dict[tuple[Qubit, Qubit], int] = dict()
    for cmd in circuit.get_commands():
        if cmd.op.is_gate():
            if cmd.op.n_qubits == 2:
                edge = (min(cmd.qubits), max(cmd.qubits))

                if edge in edge_weights:
                    edge_weights[edge] += 1
                else:
                    edge_weights[edge] = 1

            elif cmd.op.n_qubits > 2:
                raise RuntimeError(
                    "Gates must act on only 1 or 2 qubits! "
                    + f"This is not satisfied by {cmd}."
                )

    # Create the connectivity graph in NetworkX
    connectivity_graph = Graph()
    connectivity_graph.add_nodes_from(circuit.qubits)
    for (u, v), weight in edge_weights.items():
        connectivity_graph.add_edge(u, v, weight=weight)

    # Apply balanced bisections until each qubit group is small enough
    partition = {0: set(circuit.qubits)}
    stop_bisec = False  # Do at least one bisection (TTN reqs >1 leaf nodes)

    while not stop_bisec:
        old_partition = partition.copy()
        for key, group in old_partition.items():
            # Apply the balanced bisection on this group
            (groupA, groupB) = community.kernighan_lin_bisection(
                connectivity_graph.subgraph(group),
                max_iter=2 * len(group),  # Iteractions scaling with number of nodes
                weight="weight",
            )
            # Groups A and B are on the same subtree (key separated by +1)
            partition[2 * key] = groupA
            partition[2 * key + 1] = groupB

        # Stop if all groups have less than ``max_q_per_leaf`` qubits in them
        stop_bisec = all(len(group) <= max_q_per_leaf for group in partition.values())

    qubit_partition = {key: list(leaf_qubits) for key, leaf_qubits in partition.items()}
    return qubit_partition


def _get_sorted_gates(circuit: Circuit) -> list[Command]:
    """Sorts the list of gates, placing 2-qubit gates close to each other first.

    Returns an equivalent list of commands fixing the order of parallel gates so that
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
    gate_indices: dict[Qubit, list[int]] = defaultdict(list)
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
        if left_distance is None and right_distance is None:
            raise RuntimeError(
                "Some two-qubit gate in the circuit is not acting between",
                "nearest neighbour qubits. Consider using prepare_circuit_mps().",
            )
        elif left_distance is None:
            assert right_distance is not None
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
