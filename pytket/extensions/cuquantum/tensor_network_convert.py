# Copyright 2019-2023 Cambridge Quantum Computing
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

"""Tools to convert tket circuit to tensor network to be contracted with cuTensorNet."""

from collections import defaultdict
import logging
from logging import Logger
from typing import List, Tuple, Union, Any, DefaultDict
import networkx as nx  # type: ignore
from networkx.classes.reportviews import OutMultiEdgeView, OutMultiEdgeDataView  # type: ignore
import numpy as np
from numpy.typing import NDArray
from pytket.utils import Graph
from pytket.pauli import QubitPauliString  # type: ignore
from pytket.circuit import Circuit
from pytket.utils import permute_rows_cols_in_unitary


# TODO: decide whether to use logger.
def set_logger(
    logger_name: str,
    level: int = logging.INFO,
    fmt: str = "%(name)s - %(levelname)s - %(message)s",
) -> Logger:
    """Initialises and configures a logger object.

    Args:
        logger_name: Name for the logger object.
        level: Logger output level.
        fmt: Logger output format.

    Returns:
        New configured logger object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class TensorNetwork:
    """Responsible for converting pytket circuit to a tensor network and handling it."""

    def __init__(
        self, circuit: Circuit, adj: bool = False, loglevel: int = logging.INFO
    ) -> None:
        """Constructs a tensor network from a pytket circuit.

        Resulting tensor network in einsum notation suitable to use with cuTensorNet.

        Args:
            circuit: A pytket circuit to be converted to a tensor network.
            adj: Whether to create an adjoint representation of the original circuit.
            loglevel: Internal logger output level.
        """
        self._logger = set_logger("TensorNetwork", loglevel)
        self._circuit = circuit
        self._network = Graph(circuit).as_nx()
        self._node_tensors = self._assign_node_tensors(adj=adj)
        if adj:
            self._node_tensor_indices, self.sticky_indices = self._get_tn_indices(
                self._network, adj=adj
            )
        else:
            self._node_tensor_indices, self.sticky_indices = self._get_tn_indices(
                self._network
            )
        self._cuquantum_interleaved = self._make_interleaved()

    @property
    def cuquantum_interleaved(self) -> list:
        """Returns an interleaved format of the circuit tensor network."""
        return self._cuquantum_interleaved

    def _get_gate_tensors(self, adj: bool = False) -> DefaultDict[Any, List[Any]]:
        """Computes and stores tensors for each gate type from the circuit.

        The unitaries are reshaped into tensors of bond dimension two prior to being
         stored.

        Args:
            adj: Whether an adjoint representation of the original circuit is to be
             created.

        Returns:
            A map between the gate type and corresponding tensor representation(s).

        Note:
           The returned map values are lists and may contain more than one
            representation - for >1-qubit gates, different topologies (e.g. upward and
            downward) are taken into account.
        """
        name_set = {com.op.get_name() for com in self._circuit.get_commands()}
        gate_tensors = defaultdict(list)
        for i in name_set:
            for com in self._circuit.get_commands():
                if i == com.op.get_name():
                    if adj:
                        gate_tensors[i].append(
                            com.op.get_unitary()
                            .T.conjugate()
                            .reshape([2] * (2 * com.op.n_qubits))
                        )
                        self._logger.debug(
                            f"Adding unitary: \n {com.op.get_unitary().T.conjugate()}"
                        )
                    else:
                        gate_tensors[i].append(
                            com.op.get_unitary().reshape([2] * (2 * com.op.n_qubits))
                        )
                        self._logger.debug(f"Adding unitary: \n {com.op.get_unitary()}")
                    # Add a unitary for a gate pointing "upwards" (e.g. CX[1, 0])
                    if com.op.n_qubits > 1:
                        com_qix = [self._circuit.qubits.index(qb) for qb in com.args]
                        self._logger.debug(f"command qubit indices: {com_qix}")
                        com_qix_compressed = [i for i, _ in enumerate(com_qix)]
                        self._logger.debug(
                            f"command qubit indices compressed: {com_qix_compressed}"
                        )
                        com_qix_permut = list(reversed(com_qix_compressed))
                        self._logger.debug(
                            f"command qubit indices compressed permuted:"
                            f" {com_qix_permut}"
                        )
                        # TODO: check type inconsistency and remove type ignore
                        #  statements
                        if adj:
                            gate_tensors[i].append(
                                permute_rows_cols_in_unitary(
                                    com.op.get_unitary(), com_qix_permut  # type: ignore
                                )
                                .T.conjugate()
                                .reshape([2] * (2 * com.op.n_qubits))
                            )
                            self._logger.debug(
                                f"Adding unitary: \n {permute_rows_cols_in_unitary(com.op.get_unitary(), com_qix_permut).T.conjugate()}"  # type: ignore
                            )
                        else:
                            gate_tensors[i].append(
                                permute_rows_cols_in_unitary(
                                    com.op.get_unitary(), com_qix_permut  # type: ignore
                                ).reshape([2] * (2 * com.op.n_qubits))
                            )
                            self._logger.debug(  # type: ignore
                                f"Adding unitary: \n {permute_rows_cols_in_unitary(com.op.get_unitary(),com_qix_permut)}"  # type: ignore
                            )
                    break
        self._logger.debug(f"Gate tensors: \n{gate_tensors}\n")
        return gate_tensors

    def _assign_node_tensors(self, adj: bool = False) -> List[Any]:
        """Creates a list of tensors representing circuit gates (tensor network nodes).

        Args:
            adj: Whether an adjoint representation of the original circuit is to be
             created.

        Returns:
            List of tensors representing circuit gates (tensor network nodes) in the
             reversed order of circuit graph nodes.
        """
        self._gate_tensors = self._get_gate_tensors(adj=adj)
        node_tensors = []
        self._input_nodes = []
        self._output_nodes = []
        for i, node in reversed(list(enumerate(self._network.nodes(data=True)))):
            if node[1]["desc"] not in ("Input", "Output"):
                n_out_edges = len(list(self._network.out_edges(node[0])))
                if n_out_edges > 1:
                    src_ports = [
                        edge[-1]["src_port"]
                        for edge in self._network.out_edges(node[0], data=True)
                    ]
                    unit_idx = [
                        edge[-1]["unit_id"]
                        for edge in self._network.out_edges(node[0], data=True)
                    ]
                    # Detect if this is a reversed gate (pointing upward)
                    self._logger.debug(f"src_ports: {src_ports}, unit_idx: {unit_idx}")
                    self._logger.debug(
                        f"src_ports relation: {src_ports[0] < src_ports[1]}"
                    )
                    self._logger.debug(
                        f"unit_idx relation: {unit_idx[0] < unit_idx[1]}"
                    )
                    self._logger.debug(
                        f"criteria: "
                        f"{(src_ports[0] < src_ports[1]) != (unit_idx[0] < unit_idx[1])}"  # pylint: disable=line-too-long
                    )
                    if (src_ports[0] < src_ports[1]) != (unit_idx[0] < unit_idx[1]):
                        node_tensors.append(self._gate_tensors[node[1]["desc"]][1])
                        self._logger.debug(f"Adding an upward gate tensor")
                    else:
                        node_tensors.append(self._gate_tensors[node[1]["desc"]][0])
                        self._logger.debug(f"Adding a downward gate tensor")
                else:
                    node_tensors.append(self._gate_tensors[node[1]["desc"]][0])
                    self._logger.debug(f"Adding a 1-qubit gate tensor")
            else:
                if node[1]["desc"] == "Output":
                    self._output_nodes.append(i)
                if node[1]["desc"] == "Input":
                    self._input_nodes.append(i)
                    node_tensors.append(np.array([1, 0], dtype="complex128"))
        if adj:
            node_tensors.reverse()
        self._logger.debug(f"Node tensors: \n{node_tensors}\n")

        return node_tensors

    def _get_tn_indices(
        self, net: nx.MultiDiGraph, adj: bool = False
    ) -> Tuple[List[Any], dict[Any]]:
        """Computes indices of the edges of the tensor network nodes (tensors).

        Indices are computed such that they range from high (for circuit leftmost gates)
        absolute values to |1|. Sign of the indices is negative if an adjoint
        representation of the circuit is to be constructed. The outward or "sticky"
        indices (for circuit rightmost gates) are sorted (and possibly swapped with
        inner indices) such that they match qubit indices (+1) in the graph. Remaining
        indices follow the graph edges reverse order (except for the swapped ones).

        Indices of the tensors dimensions to be contracted along must match, so they are
        ordered consistently for each tensor.

        Lists of indices for each tensor (node) are stored in the same order in which
        the tensors themselves are stored.

        Args:
            net: Graph, representing the current circuit.
            adj: Whether an adjoint representation of the original circuit is to be
             created.

        Returns:
            A list of lists of tensor network nodes edges (tensors dimensions) indices
             and a list of outward ("sticky") indices along which there will be no
             contraction.
        """
        sign = -1 if adj else 1
        self._logger.debug(f"Network nodes: \n{net.nodes(data=True)}")
        self._logger.debug(f"Network edges: \n{net.edges(data=True)}")
        # There can be several identical edges for which we need different indices
        edge_indices = defaultdict(list)
        n_edges = nx.number_of_edges(net)
        # Append tuples of inverse edge indices (starting from 1) and qubit indices
        # to each edge entry
        for i, (e, ed) in enumerate(zip(net.edges(), net.edges(data=True))):
            edge_indices[e].append((sign * (n_edges - i), int(ed[-1]["unit_id"] / 2)))
        self._logger.debug(f"Network edge indices: \n {edge_indices}")
        nodes_out = self._output_nodes
        # Check if need to swap indices for outward indices
        for node in nodes_out:
            prenode = next(net.predecessors(node))
            eid = edge_indices[(prenode, node)][0][0]
            qid = edge_indices[(prenode, node)][0][1]
            if (
                eid - sign * 1 != sign * qid
            ):  # Edge indexing starts from 1 or -1, qubit from 0
                lswap = False
                # expensive:
                for edge, idx_lst in edge_indices.items():
                    for i, (ei, qi) in enumerate(idx_lst):
                        if ei - sign * 1 == sign * qid:
                            self._logger.debug(
                                f"Swapping indices of edges {edge} and "
                                f"({prenode, node})!"
                            )
                            edge_indices[(prenode, node)] = [
                                (edge_indices[edge][i][0], qid)
                            ]
                            edge_indices[edge][i] = (eid, qi)
                            lswap = True
                            break
                    if lswap:
                        break
        self._logger.debug(
            f"Network edge indices after swaps (if any): \n {edge_indices}"
        )
        # Store the "sticky" indices
        sticky_indices = {}
        for edge in net.edges():
            for node in nodes_out:
                if node in edge:
                    for ei, qi in edge_indices[edge]:
                        print(f"edge: {ei}, qubit: {qi}")
                        sticky_indices[qi] = ei
        if len(sticky_indices) != len(self._output_nodes):
            raise RuntimeError(
                f"Number of sticky indices ({len(sticky_indices)})"
                f" is not equal to number of qubits"
                f" ({len(self._output_nodes)})"
            )
        # sticky_indices.sort(key=abs)
        self._logger.debug(f"sticky (outer) edge indices: \n {sticky_indices}")
        # Assign correctly ordered indices to tensors (nodes) and store their lists in
        # the same order as we store tensors themselves.
        tn_indices = []
        for node in reversed(list(net.nodes)):
            if node in nodes_out:
                continue
            self._logger.debug(f"Node: {node}")
            num_edges = len(list(net.in_edges(node))) + len(list(net.out_edges(node)))
            in_edges_data = net.in_edges(node, data=True)
            out_edges_data = net.out_edges(node, data=True)
            in_edges = net.in_edges(node)
            out_edges = net.out_edges(node)
            self._logger.debug(f"in_edges: {in_edges}")
            self._logger.debug(f"out_edges: {out_edges}")
            ordered_edges = [0] * num_edges
            if num_edges > 2:
                ordered_out_edges = self._order_edges_for_multiqubit_gate(
                    edge_indices, out_edges, out_edges_data, 0, self._logger
                )
                ordered_in_edges = self._order_edges_for_multiqubit_gate(
                    edge_indices,
                    in_edges,
                    in_edges_data,
                    int(num_edges / 2),
                    self._logger,
                )
                for loc_idx, edge_idx in ordered_in_edges.items():
                    ordered_edges[loc_idx] = edge_idx
                for loc_idx, edge_idx in ordered_out_edges.items():
                    ordered_edges[loc_idx] = edge_idx

            else:
                ordered_edges[0] = edge_indices[list(out_edges)[0]][0][0]
                if in_edges:
                    ordered_edges[1] = edge_indices[list(in_edges)[0]][0][0]
            if adj and len(ordered_edges) > 1:
                m = int(len(ordered_edges) / 2)
                ordered_edges[:m], ordered_edges[m:] = (
                    ordered_edges[m:],
                    ordered_edges[:m],
                )
            self._logger.debug(f"New node edges: \n {ordered_edges}")
            tn_indices.append(ordered_edges)
        if adj:
            tn_indices.reverse()
        self._logger.debug(f"Final TN edges: \n {tn_indices}")
        return tn_indices, sticky_indices

    @staticmethod
    def _order_edges_for_multiqubit_gate(
        edge_indices: DefaultDict[Any, List[Tuple[Any, int]]],
        edges: OutMultiEdgeView,
        edges_data: OutMultiEdgeDataView,
        offset: int,
        logger: Logger,
    ) -> dict:
        """Returns a map from local tensor indices to global edges indices.

        Aimed at multi-qubit gates.

        This map assures correct ordering of edge indices for each multi-qubit gate
        tensor representation within the tensor network (which is important for the
        correct contraction). It should be called separately for the "incoming" and
        "outgoing" edges of a node (dimensions of a tensor, representing a gate).

        Args:
            edge_indices: a map from pytket graph edges (tuples of two integers,
             representing adjacent nodes) to a list of tuples, containing an assigned
             edge index and a corresponding qubit index.
            edges: pytket graph edges (list of tuples of two integers).
            edges_data: pytket graph edges with metadata (list of tuples of two integers
             and a dict).
            offset: an integer offset, being 0 if the "incoming" edges are to be mapped,
             or half the number of edges of the node (dimensions of a tensor) if the
             "outgoing" edges are to be mapped.
            logger: a logger object.
        """
        gate_edges_ordered = {}
        qi_to_local_ei = {}
        qis = []
        for edge_data in edges_data:
            logger.debug(f"Edge data: {edge_data}")
            logger.debug(f"Qubit id: {int(edge_data[-1]['unit_id'] / 2)}")
            qis.append(int(edge_data[-1]["unit_id"] / 2))
        qis.sort()
        for i, qi in enumerate(qis):
            qi_to_local_ei[qi] = offset + i
        logger.debug(f"Qubit to local edge index map: {qi_to_local_ei}")
        for edge_data, edge in zip(edges_data, edges):
            qi = int(edge_data[-1]["unit_id"] / 2)
            if len(edge_indices[edge]) == 1:
                gate_edges_ordered[qi_to_local_ei[qi]] = edge_indices[edge][0][0]
            else:
                for e, q in edge_indices[edge]:
                    if q == qi:
                        gate_edges_ordered[qi_to_local_ei[qi]] = e
                        break
        return gate_edges_ordered

    def _make_interleaved(self) -> list:
        """Returns an interleaved form of a tensor network.

        The format is suitable as an input for the cuQuantum-Python `contract` function.

        Combines the list of tensor representations of circuit gates and corresponding
        edges indices, that must have been constructed in the same order.

        Returns:
            A list of interleaved tensors (ndarrays) and lists of corresponding edges
             indices.
        """
        tn_interleaved = []
        for tensor, indices in zip(self._node_tensors, self._node_tensor_indices):
            tn_interleaved.append(tensor)
            tn_interleaved.append(indices)
        self._logger.debug(f"cuQuantum input list: \n{input}")
        return tn_interleaved

    def dagger(self) -> "TensorNetwork":
        """Constructs an adjoint of a tensor network object.

        Returns:
            A new TensorNetwork object, containing an adjoint representation of the
             input object.
        """
        tn_dagger = TensorNetwork(self._circuit.copy(), adj=True)
        self._logger.debug(
            f"dagger cuquantum input list: \n{tn_dagger._cuquantum_interleaved}"
        )
        return tn_dagger

    def vdot(self, tn_other: "TensorNetwork") -> list:
        """Returns a tensor network representing an overlap of two circuits.

        An adjoint representation of `tn_other` is obtained first (with the indices
        having negative sign). Then the two tensor networks are concatenated, separated
        by a single layer of unit matrices. The "sticky" indices of the two tensor
        networks connect with their counterparts via those unit matrices.

        Args:
            tn_other: a TensorNetwork object representing a circuit, an overlap with
             which is to be calculated.

        Returns:
            A tensor network in an interleaved form, representing an overlap of two
             circuits.
        """
        if set(self.sticky_indices.keys()) != set(tn_other.sticky_indices.keys()):
            raise RuntimeError("The two tensor networks are incompatible!")
        tn_other_adj = tn_other.dagger()
        i_mat = np.array([[1, 0], [0, 1]], dtype="complex128")
        sticky_index_pairs = []
        for iq in self.sticky_indices:
            sticky_index_pairs.append(
                (self.sticky_indices[iq], tn_other_adj.sticky_indices[iq])
            )
        connector = [
            f(x, y)  # type: ignore
            for x, y in sticky_index_pairs
            for f in (lambda x, y: i_mat, lambda x, y: [y, x])
        ]
        tn_concatenated = tn_other_adj.cuquantum_interleaved
        tn_concatenated.extend(connector)
        tn_concatenated.extend(self.cuquantum_interleaved)
        self._logger.debug(f"Overlap input list: \n{tn_concatenated}")
        return tn_concatenated


class PauliOperatorTensorNetwork:
    """Handles a tensor network representing a Pauli operator string."""

    PAULI = {
        "X": np.array([[0, 1], [1, 0]], dtype="complex128"),
        "Y": np.array([[0, -1j], [1j, 0]], dtype="complex128"),
        "Z": np.array([[1, 0], [0, -1]], dtype="complex128"),
        "I": np.array([[1, 0], [0, 1]], dtype="complex128"),
    }

    def __init__(
        self,
        paulis: QubitPauliString,
        bra: TensorNetwork,
        ket: TensorNetwork,
        loglevel: int = logging.INFO,
    ) -> None:
        """Constructs a tensor network representing a Pauli operator string.

        Contains a single layer of unitaries, corresponding to the provided Pauli string
        operators and identity matrices.

        Takes a circuit tensor network as input and uses its "sticky" indices to assign
        indices to the unitaries in the network - the "incoming" indices have negative
        sign and "outgoing" - positive sign.

        Args:
            paulis: Pauli operators string.
            bra: Tensor network object representing a bra circuit.
            ket: Tensor network object representing a ket circuit.
            loglevel: Logger verbosity level.
        """
        self._logger = set_logger("PauliOperatorTensorNetwork", loglevel)
        self._pauli_tensors = [self.PAULI[pauli.name] for pauli in paulis.map.values()]
        self._logger.debug(f"Pauli tensors: {self._pauli_tensors}")
        qubit_ids = [qubit.to_list()[1][0] + 1 for qubit in paulis.map.keys()]
        qubit_to_pauli = {
            qubit: pauli_tensor
            for (qubit, pauli_tensor) in zip(qubit_ids, self._pauli_tensors)
        }
        self._logger.debug(f"qubit to Pauli mapping: {qubit_to_pauli}")
        if set(bra.sticky_indices.keys()) != set(ket.sticky_indices.keys()):
            raise RuntimeError("The bra and ket tensor networks are incompatible!")
        sticky_index_pairs = []
        for iq in ket.sticky_indices:
            sticky_index_pairs.append((ket.sticky_indices[iq], bra.sticky_indices[iq]))
        self._cuquantum_interleaved = [
            f(x, y)  # type: ignore
            for x, y in sticky_index_pairs
            for f in (
                lambda x, y: qubit_to_pauli[x] if (x in qubit_ids) else self.PAULI["I"],
                lambda x, y: [y, x],
            )
        ]
        self._logger.debug(f"Pauli TN: {self.cuquantum_interleaved}")

    @property
    def cuquantum_interleaved(self) -> list:
        """Returns an interleaved format of the circuit tensor network."""
        return self._cuquantum_interleaved


class ExpectationValueTensorNetwork:
    """Handles a tensor network representing an expectation value."""

    def __init__(
        self,
        bra: TensorNetwork,
        paulis: QubitPauliString,
        ket: TensorNetwork,
        loglevel: int = logging.INFO,
    ) -> None:
        """Constructs a tensor network representing expectation value.

        Simply concatenates input tensor networks for bra and ket circuits and a string
        of Pauli operators in-between.

        Args:
            bra: Tensor network object representing a bra circuit.
            ket: Tensor network object representing a ket circuit.
            paulis: Pauli operator string.
            loglevel: Logger verbosity level.
        """
        self._bra = bra
        self._ket = ket
        self._operator = PauliOperatorTensorNetwork(paulis, bra, ket, loglevel)
        self._cuquantum_interleaved = self._make_interleaved()

    @property
    def cuquantum_interleaved(self) -> list:
        """Returns an interleaved format of the circuit tensor network."""
        return self._cuquantum_interleaved

    def _make_interleaved(self) -> list:
        """Concatenates the tensor networks elements of the expectation value.

        Returns:
            A tensor network representing expectation value in the interleaved format
             (list).
        """
        tn_concatenated = self._bra.cuquantum_interleaved
        tn_concatenated.extend(self._operator.cuquantum_interleaved)
        tn_concatenated.extend(self._ket.cuquantum_interleaved)
        return tn_concatenated


def tk_to_tensor_network(tkc: Circuit) -> List[Union[NDArray, List]]:
    """Converts pytket circuit into a tensor network.

    Args:
        tkc: Circuit.

    Returns:
        A tensor network representing the input circuit in the interleaved format
         (list).
    """
    return TensorNetwork(tkc).cuquantum_interleaved
