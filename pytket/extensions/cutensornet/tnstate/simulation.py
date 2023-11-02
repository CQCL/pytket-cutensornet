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
from typing import Optional
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
        sorted_gates = _get_sorted_gates(circuit, algorithm)

    elif algorithm == ContractionAlg.MPSxMPO:
        tnstate = MPSxMPO(  # type: ignore
            libhandle,
            circuit.qubits,
            config,
        )
        sorted_gates = _get_sorted_gates(circuit, algorithm)

    elif algorithm == ContractionAlg.TTNxGate:
        qubit_partition = _get_qubit_partition(circuit, config.leaf_size)
        tnstate = TTNxGate(  # type: ignore
            libhandle,
            qubit_partition,
            config,
        )
        sorted_gates = _get_sorted_gates(circuit, algorithm, qubit_partition)

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


def _get_sorted_gates(
    circuit: Circuit,
    algorithm: ContractionAlg,
    qubit_partition: Optional[dict[int, list[Qubit]]] = None,
) -> list[Command]:
    """Sorts the list of gates so that there's less canonicalisation during simulation.

    Returns an equivalent list of commands fixing the order of parallel gates so that
    2-qubit gates that are close together are applied one after the other. This reduces
    the overhead of canonicalisation during simulation.

    Args:
        circuit: The original circuit.
        algorithm: The simulation algorithm that will be used on this circuit.
        qubit_partition: For TTN simulation algorithms only. A partition of the
            qubits in the circuit into disjoint groups, describing the hierarchical
            structure of the TTN.

    Returns:
        The same gates, ordered in a beneficial way for the given algorithm.
    """
    all_gates = circuit.get_commands()
    sorted_gates = []
    # Entries from `all_gates` that are not yet in `sorted_gates`
    remaining = set(range(len(all_gates)))

    # Do some precomputation depending on the algorithm
    if algorithm in [ContractionAlg.TTNxGate]:
        if qubit_partition is None:
            raise RuntimeError("You must provide a qubit partition!")

        leaf_of_qubit: dict[Qubit, int] = dict()
        for leaf, qubits in qubit_partition.items():
            for q in qubits:
                leaf_of_qubit[q] = leaf

    elif algorithm in [ContractionAlg.MPSxGate, ContractionAlg.MPSxMPO]:
        idx_of_qubit = {q: i for i, q in enumerate(circuit.qubits)}

    else:
        raise RuntimeError(f"Sorting gates for {algorithm} not supported.")

    # Create the list of indices of gates acting on each qubit
    gate_indices: dict[Qubit, list[int]] = defaultdict(list)
    for i, g in enumerate(all_gates):
        for q in g.qubits:
            gate_indices[q].append(i)
    # Schedule all 1-qubit gates at the beginning of the circuit
    for q, indices in gate_indices.items():
        while indices and len(all_gates[indices[0]].qubits) == 1:
            i = indices.pop(0)
            sorted_gates.append(all_gates[i])
            remaining.remove(i)

    # Decide which 2-qubit gate to apply next
    last_qubits = [circuit.qubits[0], circuit.qubits[0]]  # Arbitrary choice at start
    while remaining:
        # Gather all gates that have nothing in front of them at one of its qubits
        reachable_gates = [gates[0] for gates in gate_indices.values() if gates]
        # Among them, find those that are available in both qubits
        available_gates: list[int] = []
        for gate_idx in reachable_gates:
            gate_qubits = all_gates[gate_idx].qubits
            assert len(gate_qubits) == 2  # Sanity check: all of them are 2-qubit gates
            # If the first gate in both qubits coincides, then this gate is available
            if gate_indices[gate_qubits[0]][0] == gate_indices[gate_qubits[1]][0]:
                assert gate_indices[gate_qubits[0]][0] == gate_idx
                available_gates.append(gate_idx)
        # Sanity check: there is at least one available 2-qubit gate
        assert available_gates

        # Find distance from last_qubits to current applicable 2-qubit gates
        gate_distance: dict[int, int] = dict()
        for gate_idx in available_gates:
            gate_qubits = all_gates[gate_idx].qubits

            # Criterion for distance depends on the simulation algorithm
            if algorithm in [ContractionAlg.TTNxGate]:
                gate_distance[gate_idx] = max(  # Max common ancestor distance
                    leaf_of_qubit[last_qubits[0]] ^ leaf_of_qubit[gate_qubits[0]],
                    leaf_of_qubit[last_qubits[0]] ^ leaf_of_qubit[gate_qubits[1]],
                    leaf_of_qubit[last_qubits[1]] ^ leaf_of_qubit[gate_qubits[0]],
                    leaf_of_qubit[last_qubits[1]] ^ leaf_of_qubit[gate_qubits[1]],
                )
            elif algorithm in [ContractionAlg.MPSxGate, ContractionAlg.MPSxMPO]:
                gate_distance[gate_idx] = max(  # Max linear distance between qubits
                    abs(idx_of_qubit[last_qubits[0]] - idx_of_qubit[gate_qubits[0]]),
                    abs(idx_of_qubit[last_qubits[0]] - idx_of_qubit[gate_qubits[1]]),
                    abs(idx_of_qubit[last_qubits[1]] - idx_of_qubit[gate_qubits[0]]),
                    abs(idx_of_qubit[last_qubits[1]] - idx_of_qubit[gate_qubits[1]]),
                )
            else:
                raise RuntimeError(f"Sorting gates for {algorithm} not supported.")

        # Choose the gate with shortest distance
        chosen_gate_idx = min(gate_distance, key=gate_distance.get)  # type: ignore
        chosen_gate = all_gates[chosen_gate_idx]

        # Schedule the gate
        last_qubits = chosen_gate.qubits
        sorted_gates.append(chosen_gate)
        remaining.remove(chosen_gate_idx)
        # Schedule all 1-qubit gates after this gate
        for q in last_qubits:
            gate_indices[q].pop(0)  # Remove the 2-qubit `chosen_gate`
            indices = gate_indices[q]
            while indices and len(all_gates[indices[0]].qubits) == 1:
                i = indices.pop(0)
                sorted_gates.append(all_gates[i])
                remaining.remove(i)

    assert len(all_gates) == len(sorted_gates)
    return sorted_gates
