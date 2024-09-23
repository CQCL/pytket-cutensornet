# Copyright 2019-2024 Quantinuum
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
from typing import Optional, Any
import warnings
from enum import Enum

from pathlib import Path
from collections import defaultdict  # type: ignore
import numpy as np  # type: ignore

import networkx as nx  # type: ignore

try:
    import kahypar  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import kahypar", ImportWarning)

from pytket.circuit import Circuit, Command, OpType, Qubit
from pytket.transform import Transform
from pytket.architecture import Architecture
from pytket.passes import DefaultMappingPass
from pytket.predicates import CompilationUnit

from pytket.extensions.cutensornet.general import CuTensorNetHandle, set_logger
from .general import Config, StructuredState
from .mps_gate import MPSxGate
from .mps_mpo import MPSxMPO
from .ttn_gate import TTNxGate


class SimulationAlgorithm(Enum):
    """An enum to refer to the StructuredState contraction algorithm.

    Each enum value corresponds to the class with the same name; see its docs for
    information about the algorithm.
    """

    TTNxGate = 0
    MPSxGate = 1
    MPSxMPO = 2


def simulate(
    libhandle: CuTensorNetHandle,
    circuit: Circuit,
    algorithm: SimulationAlgorithm,
    config: Config,
    compilation_params: Optional[dict[str, Any]] = None,
) -> StructuredState:
    """Simulates the circuit and returns the ``StructuredState`` of the final state.

    Note:
        A ``libhandle`` is created via a ``with CuTensorNetHandle() as libhandle:``
        statement. The device where the ``StructuredState`` is stored will match the one
        specified by the library handle.

        The input ``circuit`` must be composed of one-qubit and two-qubit gates only.
        Any gateset supported by ``pytket`` can be used.

    Args:
        libhandle: The cuTensorNet library handle that will be used to carry out
            tensor operations.
        circuit: The pytket circuit to be simulated.
        algorithm: Choose between the values of the ``SimulationAlgorithm`` enum.
        config: The configuration object for simulation.
        compilation_params: Experimental feature. Defaults to no compilation.
            Parameters currently not documented.

    Returns:
        An instance of ``StructuredState`` for (an approximation of) the final state
        of the circuit. The instance be of the class matching ``algorithm``.
    """
    logger = set_logger("Simulation", level=config.loglevel)

    if compilation_params is None:
        compilation_params = dict()

    # Initialise the StructuredState
    if algorithm == SimulationAlgorithm.MPSxGate:
        state = MPSxGate(  # type: ignore
            libhandle,
            circuit.qubits,
            bits=circuit.bits,
            config=config,
        )

    elif algorithm == SimulationAlgorithm.MPSxMPO:
        state = MPSxMPO(  # type: ignore
            libhandle,
            circuit.qubits,
            bits=circuit.bits,
            config=config,
        )

    elif algorithm == SimulationAlgorithm.TTNxGate:
        use_kahypar_option: bool = compilation_params.get("use_kahypar", False)

        qubit_partition = _get_qubit_partition(
            circuit, config.leaf_size, use_kahypar_option
        )
        state = TTNxGate(  # type: ignore
            libhandle,
            qubit_partition,
            bits=circuit.bits,
            config=config,
        )

    # If requested by the user, sort the gates to reduce canonicalisation overhead.
    sort_gates_option: bool = compilation_params.get("sort_gates", False)
    if sort_gates_option:
        logger.info(
            "Ordering the gates in the circuit to reduce canonicalisation overhead."
        )

        if algorithm == SimulationAlgorithm.TTNxGate:
            commands = _get_sorted_gates(circuit, algorithm, qubit_partition)
        else:
            commands = _get_sorted_gates(circuit, algorithm)
    else:
        commands = circuit.get_commands()

    # Run the simulation
    logger.info("Running simulation...")
    # Apply the gates
    for i, g in enumerate(commands):
        state.apply_gate(g)
        logger.info(f"Progress... {(100*i) // len(commands)}%")

    # Apply the batched operations that are left (if any)
    state._flush()

    # Apply the circuit's phase to the state
    state.apply_scalar(np.exp(1j * np.pi * circuit.phase))

    # Relabel qubits according to the implicit swaps (if any)
    state.apply_qubit_relabelling(circuit.implicit_qubit_permutation())

    logger.info("Simulation completed.")
    logger.info(f"Final StructuredState size={state.get_byte_size() / 2**20} MiB")
    logger.info(f"Final StructuredState fidelity={state.fidelity}")
    return state


def prepare_circuit_mps(circuit: Circuit) -> tuple[Circuit, dict[Qubit, Qubit]]:
    """Adds SWAP gates to the circuit so that all gates act on adjacent qubits.

    The qubits in the output circuit will be renamed. Implicit SWAPs may be added
    to the circuit, meaning that the logical qubit held at the ``node[i]`` qubit
    at the beginning of the circuit may differ from the one it holds at the end.
    Consider applying ``apply_qubit_relabelling`` on the MPS after simulation.

    Note:
        This preprocessing is *not* required by the MPS algorithms we provide.
        Shallow circuits tend to run faster if this preprocessing is *not* used.
        In occassions, it has been shown to improve runtime for deep circuits.

    Args:
        circuit: The circuit to be simulated.

    Returns:
        A tuple with an equivalent circuit with the appropriate structure and a
        map of qubit names at the end of the circuit to their corresponding
        original names.
    """
    if circuit.n_qubits < 2:
        # Nothing needs to be done
        return (circuit, {q: q for q in circuit.qubits})

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
    circuit: Circuit, max_q_per_leaf: int, use_kahypar: bool
) -> dict[int, list[Qubit]]:
    """Returns a qubit partition for a TTN.

    Proceeds by recursive bisection of the qubit connectivity graph, so that
    qubits that interact with each other less are connected by a common ancestor
    closer to the root.

    Args:
        circuit: The circuit to be simulated.
        max_q_per_leaf: The maximum allowed number of qubits per node leaf
        use_kahypar: Use KaHyPar for graph partitioning if this is True.
            Otherwise, use NetworkX (worse, but easy to setup).

    Returns:
        A dictionary describing the partition in the format expected by TTN.

    Raises:
        RuntimeError: If a gate acts on more than 2 qubits.
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
    connectivity_graph = nx.Graph()
    connectivity_graph.add_nodes_from(circuit.qubits)
    for (u, v), weight in edge_weights.items():
        connectivity_graph.add_edge(u, v, weight=weight)

    # Apply balanced bisections until each qubit group is small enough
    partition = {0: circuit.qubits}

    # Stop if all groups have less than ``max_q_per_leaf`` qubits in them
    while not all(len(group) <= max_q_per_leaf for group in partition.values()):
        old_partition = partition.copy()
        for key, group in old_partition.items():
            # Apply the balanced bisection on this group
            if use_kahypar:  # Using KaHyPar
                (groupA, groupB) = _apply_kahypar_bisection(
                    connectivity_graph.subgraph(group),
                )
            else:  # Using NetworkX
                (groupA, groupB) = nx.community.kernighan_lin_bisection(
                    connectivity_graph.subgraph(group),
                )
            # Groups A and B are on the same subtree (key separated by +1)
            partition[2 * key] = groupA
            partition[2 * key + 1] = groupB

    qubit_partition = {key: list(leaf_qubits) for key, leaf_qubits in partition.items()}
    return qubit_partition


def _apply_kahypar_bisection(
    graph: nx.Graph,
) -> tuple[list[Qubit], list[Qubit]]:
    """Use KaHyPar to obtain a bisection of the graph.

    Returns:
        Two lists, each containing the vertices in either group of the bisection.
    """
    vertices = list(graph.nodes)
    edges = list(graph.edges)
    weight_dict = nx.get_edge_attributes(graph, "weight")
    qubit_dict = {q: i for i, q in enumerate(vertices)}

    num_vertices = len(vertices)
    num_edges = len(edges)
    k = 2  # Number of groups in the partition
    epsilon = 0.03  # Imbalance tolerance

    # Special case where the graph has no edges; KaHyPar cannot deal with it
    if num_edges == 0:
        # Just split the list of vertices in half
        return (vertices[: num_vertices // 2], vertices[num_vertices // 2 :])

    # KaHyPar expects the list of edges to be provided as a continuous set of vertices
    # ``edge_stream`` where ``edge_indices`` indicates where each new edge begins
    # (the latter is necessary because KaHyPar can accept hyperedges)
    edge_stream = [qubit_dict[vertex] for edge in edges for vertex in edge]
    edge_indices = [0] + [2 * (i + 1) for i in range(num_edges)]
    edge_weights = [weight_dict[edge] for edge in edges]
    vertex_weights = [1 for _ in range(num_vertices)]

    hypergraph = kahypar.Hypergraph(
        num_vertices,
        num_edges,
        edge_indices,
        edge_stream,
        k,
        edge_weights,
        vertex_weights,
    )

    # Set up the configuration for KaHyPar
    context = kahypar.Context()
    context.setK(k)
    context.setEpsilon(epsilon)
    context.suppressOutput(True)

    # Load the default configuration file provided by the KaHyPar devs
    ini_file = str(Path(__file__).parent / "cut_rKaHyPar_sea20.ini")
    context.loadINIconfiguration(ini_file)

    # Run the partitioner
    kahypar.partition(hypergraph, context)
    partition_dict = {i: hypergraph.blockID(i) for i in range(hypergraph.numNodes())}

    # Obtain the two groups of qubits from ``partition_dict``
    groupA = [vertices[i] for i, block in partition_dict.items() if block == 0]
    groupB = [vertices[i] for i, block in partition_dict.items() if block == 1]

    return (groupA, groupB)


def _get_sorted_gates(
    circuit: Circuit,
    algorithm: SimulationAlgorithm,
    qubit_partition: Optional[dict[int, list[Qubit]]] = None,
) -> list[Command]:
    """Sorts the list of gates so that there's less canonicalisation during simulation.

    Returns an equivalent list of commands fixing the order of parallel gates so that
    2-qubit gates that are close together are applied one after the other. This reduces
    the overhead of canonicalisation during simulation.

    Notes:
        If the circuit has any command (other than measurement) acting on bits, this
        function gives up trying to sort the gates, and simply returns the standard
        `circuit.get_commands()`. It would be possible to update this function so that
        it can manage these commands as well, but it is not clear that there is a strong
        use case for this.

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

    # Abort if there is classical logic or classical control in the circuit (see note)
    if any(len(g.bits) != 0 and g.op.type is not OpType.Measure for g in all_gates):
        return all_gates

    sorted_gates = []
    # Entries from `all_gates` that are not yet in `sorted_gates`
    remaining = set(range(len(all_gates)))

    # Do some precomputation depending on the algorithm
    if algorithm in [SimulationAlgorithm.TTNxGate]:
        if qubit_partition is None:
            raise RuntimeError("You must provide a qubit partition!")

        leaf_of_qubit: dict[Qubit, int] = dict()
        for leaf, qubits in qubit_partition.items():
            for q in qubits:
                leaf_of_qubit[q] = leaf

    elif algorithm in [SimulationAlgorithm.MPSxGate, SimulationAlgorithm.MPSxMPO]:
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
            if algorithm in [SimulationAlgorithm.TTNxGate]:
                gate_distance[gate_idx] = max(  # Max common ancestor distance
                    leaf_of_qubit[last_qubits[0]] ^ leaf_of_qubit[gate_qubits[0]],
                    leaf_of_qubit[last_qubits[0]] ^ leaf_of_qubit[gate_qubits[1]],
                    leaf_of_qubit[last_qubits[1]] ^ leaf_of_qubit[gate_qubits[0]],
                    leaf_of_qubit[last_qubits[1]] ^ leaf_of_qubit[gate_qubits[1]],
                )
            elif algorithm in [
                SimulationAlgorithm.MPSxGate,
                SimulationAlgorithm.MPSxMPO,
            ]:
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
