# Copyright 2019-2024 Quantinuum  # type: ignore
#  # type: ignore
# Licensed under the Apache License, Version 2.0 (the "License");  # type: ignore
# you may not use this file except in compliance with the License.  # type: ignore
# You may obtain a copy of the License at  # type: ignore
##  # type: ignore
#     http://www.apache.org/licenses/LICENSE-2.0  # type: ignore
##  # type: ignore
# Unless required by applicable law or agreed to in writing, software  # type: ignore
# distributed under the License is distributed on an "AS IS" BASIS,  # type: ignore
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # type: ignore
# See the License for the specific language governing permissions and  # type: ignore
# limitations under the License.  # type: ignore
  # type: ignore
"""Tools to convert tket circuit to tensor network to be contracted with cuTensorNet."""  # type: ignore
  # type: ignore
from collections import defaultdict  # type: ignore
import logging  # type: ignore
from logging import Logger  # type: ignore
from typing import List, Tuple, Union, Any, DefaultDict  # type: ignore
import networkx as nx  # type: ignore  # type: ignore
from networkx.classes.reportviews import OutMultiEdgeView, OutMultiEdgeDataView  # type: ignore  # type: ignore
import numpy as np  # type: ignore
from numpy.typing import NDArray  # type: ignore
from pytket.utils import Graph  # type: ignore  # type: ignore
from pytket.pauli import QubitPauliString  # type: ignore
from pytket.circuit import Circuit, Qubit  # type: ignore
from pytket.utils import permute_rows_cols_in_unitary  # type: ignore
from pytket.extensions.cutensornet.general import set_logger  # type: ignore
  # type: ignore
  # type: ignore
class TensorNetwork:  # type: ignore
    """Responsible for converting pytket circuit to a tensor network and handling it."""  # type: ignore
  # type: ignore
    def __init__(  # type: ignore
        self, circuit: Circuit, adj: bool = False, loglevel: int = logging.INFO  # type: ignore
    ) -> None:  # type: ignore
        """Constructs a tensor network from a pytket circuit.  # type: ignore
  # type: ignore
        Resulting tensor network in einsum notation suitable to use with cuTensorNet.  # type: ignore
  # type: ignore
        Args:  # type: ignore
            circuit: A pytket circuit to be converted to a tensor network.  # type: ignore
            adj: Whether to create an adjoint representation of the original circuit.  # type: ignore
            loglevel: Internal logger output level.  # type: ignore
  # type: ignore
        Raises:  # type: ignore
            RuntimeError: If ``Box`` objects are present in the circuit.  # type: ignore
        """  # type: ignore
        self._logger = set_logger("TensorNetwork", loglevel)  # type: ignore
        self._circuit = circuit  # type: ignore
        # self._circuit.replace_implicit_wire_swaps()  # type: ignore
        self._qubit_names_ilo = [str(q) for q in self._circuit.qubits]  # type: ignore
        self._logger.debug(f"ILO-ordered qubit names: {self._qubit_names_ilo}")  # type: ignore
        self._graph = Graph(self._circuit)  # type: ignore  # type: ignore
        qname_to_q = {  # type: ignore
            qname: q for qname, q in zip(self._qubit_names_ilo, self._circuit.qubits)  # type: ignore
        }  # type: ignore
        self._output_index_to_qubit = {  # type: ignore
            oi: qname_to_q[qname] for oi, qname in self._graph.output_names.items()  # type: ignore  # type: ignore
        }  # type: ignore
        self._logger.debug(  # type: ignore
            f"NX output index to (possibly re-labeled) qubit objects map: "  # type: ignore
            f"{self._output_index_to_qubit}"  # type: ignore
        )  # type: ignore
        self._network = self._graph.as_nx()  # type: ignore  # type: ignore
        self._node_tensors = self._assign_node_tensors(adj=adj)  # type: ignore
        self._node_tensor_indices, self.sticky_indices = self._get_tn_indices(  # type: ignore
            self._network, adj=adj  # type: ignore
        )  # type: ignore
        self._cuquantum_interleaved = self._make_interleaved()  # type: ignore
  # type: ignore
    @property  # type: ignore
    def cuquantum_interleaved(self) -> list:  # type: ignore
        """Returns an interleaved format of the circuit tensor network."""  # type: ignore
        return self._cuquantum_interleaved  # type: ignore
  # type: ignore
    def _get_gate_tensors(self, adj: bool = False) -> DefaultDict[Any, List[Any]]:  # type: ignore
        """Computes and stores tensors for each gate type from the circuit.  # type: ignore
  # type: ignore
        The unitaries are reshaped into tensors of bond dimension two prior to being  # type: ignore
         stored.  # type: ignore
  # type: ignore
        Args:  # type: ignore
            adj: Whether an adjoint representation of the original circuit is to be  # type: ignore
             created.  # type: ignore
  # type: ignore
        Returns:  # type: ignore
            A map between the gate type and corresponding tensor representation(s).  # type: ignore
  # type: ignore
        Raises:  # type: ignore
            RuntimeError: If ``Box`` objects are present in the circuit.  # type: ignore
  # type: ignore
        Note:  # type: ignore
           The returned map values are lists and may contain more than one  # type: ignore
           representation - for >1-qubit gates, different topologies (e.g. upward and  # type: ignore
           downward) are taken into account.  # type: ignore
        """  # type: ignore
        name_set = {com.op.get_name() for com in self._circuit.get_commands()}  # type: ignore
        for name in name_set:  # type: ignore
            if "Box" in name:  # type: ignore
                raise RuntimeError(  # type: ignore
                    "Currently TensorNetwork does not accept pytket Box"  # type: ignore
                    " objects. Please first run"  # type: ignore
                    " ``DecomposeBoxes().apply(circuit)``"  # type: ignore
                )  # type: ignore
        gate_tensors = defaultdict(list)  # type: ignore
        for i in name_set:  # type: ignore
            for com in self._circuit.get_commands():  # type: ignore
                if i == com.op.get_name():  # type: ignore
                    if adj:  # type: ignore
                        gate_tensors[i].append(  # type: ignore
                            com.op.get_unitary()  # type: ignore
                            .T.conjugate()  # type: ignore
                            .reshape([2] * (2 * com.op.n_qubits))  # type: ignore
                        )  # type: ignore
                        self._logger.debug(  # type: ignore
                            f"Adding unitary: \n {com.op.get_unitary().T.conjugate()}"  # type: ignore
                        )  # type: ignore
                    else:  # type: ignore
                        gate_tensors[i].append(  # type: ignore
                            com.op.get_unitary().reshape([2] * (2 * com.op.n_qubits))  # type: ignore
                        )  # type: ignore
                        self._logger.debug(f"Adding unitary: \n {com.op.get_unitary()}")  # type: ignore
                    # Add a unitary for a gate pointing "upwards" (e.g. CX[1, 0])  # type: ignore
                    if com.op.n_qubits > 1:  # type: ignore
                        com_qix = [self._circuit.qubits.index(qb) for qb in com.qubits]  # type: ignore
                        self._logger.debug(f"command qubit indices: {com_qix}")  # type: ignore
                        com_qix_compressed = [i for i, _ in enumerate(com_qix)]  # type: ignore
                        self._logger.debug(  # type: ignore
                            f"command qubit indices compressed: {com_qix_compressed}"  # type: ignore
                        )  # type: ignore
                        com_qix_permut = list(reversed(com_qix_compressed))  # type: ignore
                        self._logger.debug(  # type: ignore
                            f"command qubit indices compressed permuted:"  # type: ignore
                            f" {com_qix_permut}"  # type: ignore
                        )  # type: ignore
                        # TODO: check type inconsistency and remove type ignore  # type: ignore
                        #  statements  # type: ignore
                        if adj:  # type: ignore
                            gate_tensors[i].append(  # type: ignore
                                permute_rows_cols_in_unitary(  # type: ignore
                                    com.op.get_unitary(), com_qix_permut  # type: ignore  # type: ignore
                                )  # type: ignore
                                .T.conjugate()  # type: ignore
                                .reshape([2] * (2 * com.op.n_qubits))  # type: ignore
                            )  # type: ignore
                            self._logger.debug(  # type: ignore
                                f"Adding unitary: \n {permute_rows_cols_in_unitary(com.op.get_unitary(), com_qix_permut).T.conjugate()}"  # type: ignore  # type: ignore
                            )  # type: ignore
                        else:  # type: ignore
                            gate_tensors[i].append(  # type: ignore
                                permute_rows_cols_in_unitary(  # type: ignore
                                    com.op.get_unitary(), com_qix_permut  # type: ignore  # type: ignore
                                ).reshape([2] * (2 * com.op.n_qubits))  # type: ignore
                            )  # type: ignore
                            self._logger.debug(  # type: ignore  # type: ignore
                                f"Adding unitary: \n {permute_rows_cols_in_unitary(com.op.get_unitary(),com_qix_permut)}"  # type: ignore  # type: ignore
                            )  # type: ignore
                    break  # type: ignore
        self._logger.debug(f"Gate tensors: \n{gate_tensors}\n")  # type: ignore
        return gate_tensors  # type: ignore
  # type: ignore
    def _assign_node_tensors(self, adj: bool = False) -> List[Any]:  # type: ignore
        """Creates a list of tensors representing circuit gates (tensor network nodes).  # type: ignore
  # type: ignore
        Args:  # type: ignore
            adj: Whether an adjoint representation of the original circuit is to be  # type: ignore
             created.  # type: ignore
  # type: ignore
        Returns:  # type: ignore
            List of tensors representing circuit gates (tensor network nodes) in the  # type: ignore
            reversed order of circuit graph nodes.  # type: ignore
        """  # type: ignore
        self._gate_tensors = self._get_gate_tensors(adj=adj)  # type: ignore
        node_tensors = []  # type: ignore
        self._input_nodes = []  # type: ignore
        self._output_nodes = []  # type: ignore
        for i, node in reversed(list(enumerate(self._network.nodes(data=True)))):  # type: ignore
            if node[1]["desc"] not in ("Input", "Output", "Create"):  # type: ignore
                n_out_edges = len(list(self._network.out_edges(node[0])))  # type: ignore
                if n_out_edges > 1:  # type: ignore
                    src_ports = [  # type: ignore
                        edge[-1]["src_port"]  # type: ignore
                        for edge in self._network.out_edges(node[0], data=True)  # type: ignore
                    ]  # type: ignore
                    unit_idx = [  # type: ignore
                        edge[-1]["unit_id"]  # type: ignore
                        for edge in self._network.out_edges(node[0], data=True)  # type: ignore
                    ]  # type: ignore
                    # Detect if this is a reversed gate (pointing upward)  # type: ignore
                    self._logger.debug(f"src_ports: {src_ports}, unit_idx: {unit_idx}")  # type: ignore
                    self._logger.debug(  # type: ignore
                        f"src_ports relation: {src_ports[0] < src_ports[1]}"  # type: ignore
                    )  # type: ignore
                    self._logger.debug(  # type: ignore
                        f"unit_idx relation: {unit_idx[0] < unit_idx[1]}"  # type: ignore
                    )  # type: ignore
                    self._logger.debug(  # type: ignore
                        f"criteria: "  # type: ignore
                        f"{(src_ports[0] < src_ports[1]) != (unit_idx[0] < unit_idx[1])}"  # pylint: disable=line-too-long  # type: ignore
                    )  # type: ignore
                    if (src_ports[0] < src_ports[1]) != (unit_idx[0] < unit_idx[1]):  # type: ignore
                        node_tensors.append(self._gate_tensors[node[1]["desc"]][1])  # type: ignore
                        self._logger.debug(f"Adding an upward gate tensor")  # type: ignore
                    else:  # type: ignore
                        node_tensors.append(self._gate_tensors[node[1]["desc"]][0])  # type: ignore
                        self._logger.debug(f"Adding a downward gate tensor")  # type: ignore
                else:  # type: ignore
                    node_tensors.append(self._gate_tensors[node[1]["desc"]][0])  # type: ignore
                    self._logger.debug(f"Adding a 1-qubit gate tensor")  # type: ignore
            else:  # type: ignore
                if node[1]["desc"] == "Output":  # type: ignore
                    self._output_nodes.append(i)  # type: ignore
                if node[1]["desc"] == "Input" or node[1]["desc"] == "Create":  # type: ignore
                    self._input_nodes.append(i)  # type: ignore
                    node_tensors.append(np.array([1, 0], dtype="complex128"))  # type: ignore
        if adj:  # type: ignore
            node_tensors.reverse()  # type: ignore
        self._logger.debug(f"Node tensors: \n{node_tensors}\n")  # type: ignore
  # type: ignore
        return node_tensors  # type: ignore
  # type: ignore
    def _get_tn_indices(  # type: ignore
        self, net: nx.MultiDiGraph, adj: bool = False  # type: ignore
    ) -> Tuple[List[Any], dict[Qubit, int]]:  # type: ignore
        """Computes indices of the edges of the tensor network nodes (tensors).  # type: ignore
  # type: ignore
        Indices are computed such that they range from high (for circuit leftmost gates)  # type: ignore
        absolute values to |1|. Sign of the indices is negative if an adjoint  # type: ignore
        representation of the circuit is to be constructed. The outward or "sticky"  # type: ignore
        indices (for circuit rightmost gates) are sorted (and possibly swapped with  # type: ignore
        inner indices) such that they match qubit indices (+1) in the graph. Remaining  # type: ignore
        indices follow the graph edges reverse order (except for the swapped ones).  # type: ignore
  # type: ignore
        Indices of the tensors dimensions to be contracted along must match, so they are  # type: ignore
        ordered consistently for each tensor.  # type: ignore
  # type: ignore
        Lists of indices for each tensor (node) are stored in the same order in which  # type: ignore
        the tensors themselves are stored.  # type: ignore
  # type: ignore
        Args:  # type: ignore
            net: Graph, representing the current circuit.  # type: ignore
            adj: Whether an adjoint representation of the original circuit is to be  # type: ignore
             created.  # type: ignore
  # type: ignore
        Returns:  # type: ignore
            A list of lists of tensor network nodes edges (tensors dimensions) indices  # type: ignore
            and a list of outward ("sticky") indices along which there will be no  # type: ignore
            contraction.  # type: ignore
        """  # type: ignore
        sign = -1 if adj else 1  # type: ignore
        self._logger.debug(f"Network nodes: \n{net.nodes(data=True)}")  # type: ignore
        self._logger.debug(f"Network edges: \n{net.edges(data=True)}")  # type: ignore
        # There can be several identical edges for which we need different indices  # type: ignore
        edge_indices = defaultdict(list)  # type: ignore
        n_edges = nx.number_of_edges(net)  # type: ignore
        # Append tuples of inverse edge indices (starting from 1) and their unit_id's  # type: ignore
        # to each edge entry  # type: ignore
        for i, (e, ed) in enumerate(zip(net.edges(), net.edges(data=True))):  # type: ignore
            edge_indices[e].append((sign * (n_edges - i), int(ed[-1]["unit_id"])))  # type: ignore
        self._logger.debug(f"Network edge indices: \n {edge_indices}")  # type: ignore
        nodes_out = self._output_nodes  # type: ignore
        # Re-order outward edges indices according to ILO  # type: ignore
        edges_out = [  # type: ignore
            edge for edge in net.edges() if edge[1] in self._graph.output_names  # type: ignore  # type: ignore
        ]  # type: ignore
        eids = [  # type: ignore
            record[0][0] for key, record in edge_indices.items() if key in edges_out  # type: ignore
        ]  # type: ignore
        eids_sorted = sorted(eids, key=abs)  # type: ignore
        qnames_graph_ordered = [qname for qname in self._graph.output_names.values()]  # type: ignore  # type: ignore
        oids_graph_ordered = [oid for oid in self._graph.output_names.keys()]  # type: ignore  # type: ignore
        eids_qubit_ordered = [  # type: ignore
            eids_sorted[qnames_graph_ordered.index(q)] for q in self._qubit_names_ilo  # type: ignore
        ]  # Order eid's in the same way as qnames_graph_ordered as compared to ILO  # type: ignore
        oids_qubit_ordered = [  # type: ignore
            oids_graph_ordered[qnames_graph_ordered.index(q)]  # type: ignore
            for q in self._qubit_names_ilo  # type: ignore
        ]  # Order output edges indexes such that each still corresponds to the same  # type: ignore
        # qubit from the graph output_names, but with the qubits re-ordered in ILO order  # type: ignore
        oid_to_eid = {  # type: ignore
            oid: eid for oid, eid in zip(oids_qubit_ordered, eids_qubit_ordered)  # type: ignore
        }  # type: ignore
        for edge in edges_out:  # type: ignore
            uid = edge_indices[edge][0][1]  # type: ignore
            edge_indices[edge] = [(oid_to_eid[edge[1]], uid)]  # type: ignore
        self._logger.debug(  # type: ignore
            f"Network edge indices after swaps (if any): \n {edge_indices}"  # type: ignore
        )  # type: ignore
        # Store the "sticky" indices  # type: ignore
        sticky_indices = {}  # type: ignore
        for edge in edges_out:  # type: ignore
            for eid, _ in edge_indices[edge]:  # type: ignore
                sticky_indices[self._output_index_to_qubit[edge[1]]] = eid  # type: ignore
        if len(sticky_indices) != len(self._output_nodes):  # type: ignore
            raise RuntimeError(  # type: ignore
                f"Number of sticky indices ({len(sticky_indices)})"  # type: ignore
                f" is not equal to number of qubits"  # type: ignore
                f" ({len(self._output_nodes)})"  # type: ignore
            )  # type: ignore
        self._logger.debug(f"sticky (outer) edge indices: \n {sticky_indices}")  # type: ignore
        # Assign correctly ordered indices to tensors (nodes) and store their lists in  # type: ignore
        # the same order as we store tensors themselves.  # type: ignore
        tn_indices = []  # type: ignore
        for node in reversed(list(net.nodes)):  # type: ignore
            if node in nodes_out:  # type: ignore
                continue  # type: ignore
            self._logger.debug(f"Node: {node}")  # type: ignore
            num_edges = len(list(net.in_edges(node))) + len(list(net.out_edges(node)))  # type: ignore
            in_edges_data = net.in_edges(node, data=True)  # type: ignore
            out_edges_data = net.out_edges(node, data=True)  # type: ignore
            in_edges = net.in_edges(node)  # type: ignore
            out_edges = net.out_edges(node)  # type: ignore
            self._logger.debug(f"in_edges: {in_edges}")  # type: ignore
            self._logger.debug(f"out_edges: {out_edges}")  # type: ignore
            ordered_edges = [0] * num_edges  # type: ignore
            if num_edges > 2:  # type: ignore
                ordered_out_edges = self._order_edges_for_multiqubit_gate(  # type: ignore
                    edge_indices, out_edges, out_edges_data, 0, self._logger  # type: ignore
                )  # type: ignore
                ordered_in_edges = self._order_edges_for_multiqubit_gate(  # type: ignore
                    edge_indices,  # type: ignore
                    in_edges,  # type: ignore
                    in_edges_data,  # type: ignore
                    int(num_edges / 2),  # type: ignore
                    self._logger,  # type: ignore
                )  # type: ignore
                for loc_idx, edge_idx in ordered_in_edges.items():  # type: ignore
                    ordered_edges[loc_idx] = edge_idx  # type: ignore
                for loc_idx, edge_idx in ordered_out_edges.items():  # type: ignore
                    ordered_edges[loc_idx] = edge_idx  # type: ignore
  # type: ignore
            else:  # type: ignore
                ordered_edges[0] = edge_indices[list(out_edges)[0]][0][0]  # type: ignore
                if in_edges:  # type: ignore
                    ordered_edges[1] = edge_indices[list(in_edges)[0]][0][0]  # type: ignore
            if adj and len(ordered_edges) > 1:  # type: ignore
                m = int(len(ordered_edges) / 2)  # type: ignore
                ordered_edges[:m], ordered_edges[m:] = (  # type: ignore
                    ordered_edges[m:],  # type: ignore
                    ordered_edges[:m],  # type: ignore
                )  # type: ignore
            self._logger.debug(f"New node edges: \n {ordered_edges}")  # type: ignore
            tn_indices.append(ordered_edges)  # type: ignore
        if adj:  # type: ignore
            tn_indices.reverse()  # type: ignore
        self._logger.debug(f"Final TN edges: \n {tn_indices}")  # type: ignore
        return tn_indices, sticky_indices  # type: ignore
  # type: ignore
    @staticmethod  # type: ignore
    def _order_edges_for_multiqubit_gate(  # type: ignore
        edge_indices: DefaultDict[Any, List[Tuple[Any, int]]],  # type: ignore
        edges: OutMultiEdgeView,  # type: ignore
        edges_data: OutMultiEdgeDataView,  # type: ignore
        offset: int,  # type: ignore
        logger: Logger,  # type: ignore
    ) -> dict:  # type: ignore
        """Returns a map from local tensor indices to global edges indices.  # type: ignore
  # type: ignore
        Aimed at multi-qubit gates.  # type: ignore
  # type: ignore
        This map assures correct ordering of edge indices for each multi-qubit gate  # type: ignore
        tensor representation within the tensor network (which is important for the  # type: ignore
        correct contraction). It should be called separately for the "incoming" and  # type: ignore
        "outgoing" edges of a node (dimensions of a tensor, representing a gate).  # type: ignore
  # type: ignore
        Args:  # type: ignore
            edge_indices: a map from pytket graph edges (tuples of two integers,  # type: ignore
             representing adjacent nodes) to a list of tuples, containing an assigned  # type: ignore
             edge index and a corresponding unit_id (graph-specific qubit label).  # type: ignore
            edges: pytket graph edges (list of tuples of two integers).  # type: ignore
            edges_data: pytket graph edges with metadata (list of tuples of two integers  # type: ignore
             and a dict).  # type: ignore
            offset: an integer offset, being 0 if the "incoming" edges are to be mapped,  # type: ignore
             or half the number of edges of the node (dimensions of a tensor) if the  # type: ignore
             "outgoing" edges are to be mapped.  # type: ignore
            logger: a logger object.  # type: ignore
        """  # type: ignore
        gate_edges_ordered = {}  # type: ignore
        uid_to_local_ei = {}  # type: ignore
        uids = []  # type: ignore
        for edge_data in edges_data:  # type: ignore
            logger.debug(f"Edge data: {edge_data}")  # type: ignore
            uids.append(int(edge_data[-1]["unit_id"]))  # type: ignore
            logger.debug(f"UID: {uids[-1]}")  # type: ignore
        uids.sort()  # type: ignore
        for i, uid in enumerate(uids):  # type: ignore
            uid_to_local_ei[uid] = offset + i  # type: ignore
        logger.debug(f"UID to local edge index map: {uid_to_local_ei}")  # type: ignore
        for edge_data, edge in zip(edges_data, edges):  # type: ignore
            uid = int(edge_data[-1]["unit_id"])  # type: ignore
            if len(edge_indices[edge]) == 1:  # type: ignore
                gate_edges_ordered[uid_to_local_ei[uid]] = edge_indices[edge][0][0]  # type: ignore
            else:  # type: ignore
                for e, u in edge_indices[edge]:  # type: ignore
                    if u == uid:  # type: ignore
                        gate_edges_ordered[uid_to_local_ei[uid]] = e  # type: ignore
                        break  # type: ignore
        return gate_edges_ordered  # type: ignore
  # type: ignore
    def _make_interleaved(self) -> list:  # type: ignore
        """Returns an interleaved form of a tensor network.  # type: ignore
  # type: ignore
        The format is suitable as an input for the cuQuantum-Python `contract` function.  # type: ignore
  # type: ignore
        Combines the list of tensor representations of circuit gates and corresponding  # type: ignore
        edges indices, that must have been constructed in the same order.  # type: ignore
  # type: ignore
        Returns:  # type: ignore
            A list of interleaved tensors (ndarrays) and lists of corresponding edges  # type: ignore
            indices.  # type: ignore
        """  # type: ignore
        tn_interleaved = []  # type: ignore
        for tensor, indices in zip(self._node_tensors, self._node_tensor_indices):  # type: ignore
            tn_interleaved.append(tensor)  # type: ignore
            tn_interleaved.append(indices)  # type: ignore
        self._logger.debug(f"cuQuantum input list: \n{input}")  # type: ignore
        return tn_interleaved  # type: ignore
  # type: ignore
    def dagger(self) -> "TensorNetwork":  # type: ignore
        """Constructs an adjoint of a tensor network object.  # type: ignore
  # type: ignore
        Returns:  # type: ignore
            A new TensorNetwork object, containing an adjoint representation of the  # type: ignore
            input object.  # type: ignore
        """  # type: ignore
        tn_dagger = TensorNetwork(self._circuit.copy(), adj=True)  # type: ignore
        self._logger.debug(  # type: ignore
            f"dagger cutensornet input list: \n{tn_dagger._cuquantum_interleaved}"  # type: ignore
        )  # type: ignore
        return tn_dagger  # type: ignore
  # type: ignore
    def vdot(self, tn_other: "TensorNetwork") -> list:  # type: ignore
        """Returns a tensor network representing an overlap of two circuits.  # type: ignore
  # type: ignore
        An adjoint representation of `tn_other` is obtained first (with the indices  # type: ignore
        having negative sign). Then the two tensor networks are concatenated, separated  # type: ignore
        by a single layer of unit matrices. The "sticky" indices of the two tensor  # type: ignore
        networks connect with their counterparts via those unit matrices.  # type: ignore
  # type: ignore
        Args:  # type: ignore
            tn_other: a TensorNetwork object representing a circuit, an overlap with  # type: ignore
             which is to be calculated.  # type: ignore
  # type: ignore
        Returns:  # type: ignore
            A tensor network in an interleaved form, representing an overlap of two  # type: ignore
            circuits.  # type: ignore
        """  # type: ignore
        if set(self.sticky_indices.keys()) != set(tn_other.sticky_indices.keys()):  # type: ignore
            raise RuntimeError("The two tensor networks are incompatible!")  # type: ignore
        tn_other_adj = tn_other.dagger()  # type: ignore
        i_mat = np.array([[1, 0], [0, 1]], dtype="complex128")  # type: ignore
        sticky_index_pairs = []  # type: ignore
        for q in self.sticky_indices:  # type: ignore
            sticky_index_pairs.append(  # type: ignore
                (self.sticky_indices[q], tn_other_adj.sticky_indices[q])  # type: ignore
            )  # type: ignore
        connector = [  # type: ignore
            f(x, y)  # type: ignore  # type: ignore
            for x, y in sticky_index_pairs  # type: ignore
            for f in (lambda x, y: i_mat, lambda x, y: [y, x])  # type: ignore
        ]  # type: ignore
        tn_concatenated = tn_other_adj.cuquantum_interleaved  # type: ignore
        tn_concatenated.extend(connector)  # type: ignore
        tn_concatenated.extend(self.cuquantum_interleaved)  # type: ignore
        self._logger.debug(f"Overlap input list: \n{tn_concatenated}")  # type: ignore
        return tn_concatenated  # type: ignore
  # type: ignore
  # type: ignore
def measure_qubit_state(  # type: ignore
    ket: TensorNetwork, qubit_id: Qubit, bit_value: int, loglevel: int = logging.INFO  # type: ignore
) -> TensorNetwork:  # type: ignore
    """Measures a qubit in a tensor network.  # type: ignore
  # type: ignore
    Does so by appending a measurement gate to the tensor network.  # type: ignore
    The measurment gate is applied via appending a tensor cap of  # type: ignore
    the form:  0: [1, 0] or 1: [0, 1] to the interleaved einsum input.  # type: ignore
    Therefor removing one of the open indices of the tensor network.  # type: ignore
  # type: ignore
    Args:  # type: ignore
        ket: a TensorNetwork object representing a quantum state.  # type: ignore
        qubit_id: a qubit id.  # type: ignore
        bit_value: a bit value to be assigned to the measured qubit.  # type: ignore
        loglevel: logging level.  # type: ignore
  # type: ignore
    Returns:  # type: ignore
        A TensorNetwork object representing a quantum state after the  # type: ignore
        measurement with a modified interleaved notation containing the extra  # type: ignore
        measurement tensor.  # type: ignore
    """  # type: ignore
  # type: ignore
    cap = {  # type: ignore
        0: np.array([1, 0], dtype="complex128"),  # type: ignore
        1: np.array([0, 1], dtype="complex128"),  # type: ignore
    }  # type: ignore
  # type: ignore
    sticky_ind = ket.sticky_indices[qubit_id]  # type: ignore
    ket._cuquantum_interleaved.extend([cap[bit_value], [sticky_ind]])  # type: ignore
    ket.sticky_indices.pop(qubit_id)  # type: ignore
    return ket  # type: ignore
  # type: ignore
  # type: ignore
# TODO: Make this compatible with mid circuit measurements and reset  # type: ignore
def measure_qubits_state(  # type: ignore
    ket: TensorNetwork, measurement_dict: dict[Qubit, int], loglevel: int = logging.INFO  # type: ignore
) -> TensorNetwork:  # type: ignore
    """Measures a list of qubits in a tensor network.  # type: ignore
  # type: ignore
    Does so by appending a measurement gate to the tensor network.  # type: ignore
    The measurment gate is applied via appending a tensor cap  # type: ignore
    of the form:  0: [1, 0] or 1: [0, 1] to the interleaved einsum input.  # type: ignore
    Therefor removing the open indices of the tensor network corresponding  # type: ignore
    to the measured qubits.  # type: ignore
  # type: ignore
    Args:  # type: ignore
        ket: a TensorNetwork object representing a quantum state.  # type: ignore
        measurement_dict: a dictionary of qubit ids and their corresponding bit values  # type: ignore
         to be assigned to the measured qubits.  # type: ignore
        loglevel: logging level.  # type: ignore
  # type: ignore
    Returns:  # type: ignore
        A TensorNetwork object representing a quantum state after  # type: ignore
        the measurement with a modified interleaved notation containing  # type: ignore
        the extra measurement tensors.  # type: ignore
    """  # type: ignore
    for qubit_id, bit_value in measurement_dict.items():  # type: ignore
        ket = measure_qubit_state(ket, qubit_id, bit_value, loglevel)  # type: ignore
    return ket  # type: ignore
  # type: ignore
  # type: ignore
class PauliOperatorTensorNetwork:  # type: ignore
    """Handles a tensor network representing a Pauli operator string."""  # type: ignore
  # type: ignore
    PAULI = {  # type: ignore
        "X": np.array([[0, 1], [1, 0]], dtype="complex128"),  # type: ignore
        "Y": np.array([[0, -1j], [1j, 0]], dtype="complex128"),  # type: ignore
        "Z": np.array([[1, 0], [0, -1]], dtype="complex128"),  # type: ignore
        "I": np.array([[1, 0], [0, 1]], dtype="complex128"),  # type: ignore
    }  # type: ignore
  # type: ignore
    def __init__(  # type: ignore
        self,  # type: ignore
        paulis: QubitPauliString,  # type: ignore
        bra: TensorNetwork,  # type: ignore
        ket: TensorNetwork,  # type: ignore
        loglevel: int = logging.INFO,  # type: ignore
    ) -> None:  # type: ignore
        """Constructs a tensor network representing a Pauli operator string.  # type: ignore
  # type: ignore
        Contains a single layer of unitaries, corresponding to the provided Pauli string  # type: ignore
        operators and identity matrices.  # type: ignore
  # type: ignore
        Takes a circuit tensor network as input and uses its "sticky" indices to assign  # type: ignore
        indices to the unitaries in the network - the "incoming" indices have negative  # type: ignore
        sign and "outgoing" - positive sign.  # type: ignore
  # type: ignore
        Args:  # type: ignore
            paulis: Pauli operators string.  # type: ignore
            bra: Tensor network object representing a bra circuit.  # type: ignore
            ket: Tensor network object representing a ket circuit.  # type: ignore
            loglevel: Logger verbosity level.  # type: ignore
        """  # type: ignore
        self._logger = set_logger("PauliOperatorTensorNetwork", loglevel)  # type: ignore
        self._pauli_tensors = [self.PAULI[pauli.name] for pauli in paulis.map.values()]  # type: ignore
        self._logger.debug(f"Pauli tensors: {self._pauli_tensors}")  # type: ignore
        qubits = [q for q in paulis.map.keys()]  # type: ignore
        # qubit_names = [  # type: ignore
        #    "".join([q.reg_name, "".join([f"[{str(i)}]" for i in q.index])])  # type: ignore
        #    for q in paulis.map.keys()  # type: ignore
        # ]  # type: ignore
        # qubit_ids = [qubit.to_list()[1][0] + 1 for qubit in paulis.map.keys()]  # type: ignore
        qubit_to_pauli = {  # type: ignore
            qubit: pauli_tensor  # type: ignore
            for (qubit, pauli_tensor) in zip(qubits, self._pauli_tensors)  # type: ignore
        }  # type: ignore
        self._logger.debug(f"qubit to Pauli mapping: {qubit_to_pauli}")  # type: ignore
        if set(bra.sticky_indices.keys()) != set(ket.sticky_indices.keys()):  # type: ignore
            raise RuntimeError("The bra and ket tensor networks are incompatible!")  # type: ignore
        sticky_index_pairs = []  # type: ignore
        sticky_qubits = []  # type: ignore
        for q in ket.sticky_indices:  # type: ignore
            sticky_index_pairs.append((ket.sticky_indices[q], bra.sticky_indices[q]))  # type: ignore
            sticky_qubits.append(q)  # type: ignore
        self._cuquantum_interleaved = [  # type: ignore
            f(x, y, q)  # type: ignore  # type: ignore
            for (x, y), q in zip(sticky_index_pairs, sticky_qubits)  # type: ignore
            for f in (  # type: ignore
                lambda x, y, q: qubit_to_pauli[q] if (q in qubits) else self.PAULI["I"],  # type: ignore
                lambda x, y, q: [y, x],  # type: ignore
            )  # type: ignore
        ]  # type: ignore
        self._logger.debug(f"Pauli TN: {self.cuquantum_interleaved}")  # type: ignore
  # type: ignore
    @property  # type: ignore
    def cuquantum_interleaved(self) -> list:  # type: ignore
        """Returns an interleaved format of the circuit tensor network."""  # type: ignore
        return self._cuquantum_interleaved  # type: ignore
  # type: ignore
  # type: ignore
class ExpectationValueTensorNetwork:  # type: ignore
    """Handles a tensor network representing an expectation value."""  # type: ignore
  # type: ignore
    def __init__(  # type: ignore
        self,  # type: ignore
        bra: TensorNetwork,  # type: ignore
        paulis: QubitPauliString,  # type: ignore
        ket: TensorNetwork,  # type: ignore
        loglevel: int = logging.INFO,  # type: ignore
    ) -> None:  # type: ignore
        """Constructs a tensor network representing expectation value.  # type: ignore
  # type: ignore
        Simply concatenates input tensor networks for bra and ket circuits and a string  # type: ignore
        of Pauli operators in-between.  # type: ignore
  # type: ignore
        Args:  # type: ignore
            bra: Tensor network object representing a bra circuit.  # type: ignore
            ket: Tensor network object representing a ket circuit.  # type: ignore
            paulis: Pauli operator string.  # type: ignore
            loglevel: Logger verbosity level.  # type: ignore
        """  # type: ignore
        self._bra = bra  # type: ignore
        self._ket = ket  # type: ignore
        self._operator = PauliOperatorTensorNetwork(paulis, bra, ket, loglevel)  # type: ignore
        self._cuquantum_interleaved = self._make_interleaved()  # type: ignore
  # type: ignore
    @property  # type: ignore
    def cuquantum_interleaved(self) -> list:  # type: ignore
        """Returns an interleaved format of the circuit tensor network."""  # type: ignore
        return self._cuquantum_interleaved  # type: ignore
  # type: ignore
    def _make_interleaved(self) -> list:  # type: ignore
        """Concatenates the tensor networks elements of the expectation value.  # type: ignore
  # type: ignore
        Returns:  # type: ignore
            A tensor network representing expectation value in the interleaved format  # type: ignore
            (list).  # type: ignore
        """  # type: ignore
        tn_concatenated = self._bra.cuquantum_interleaved.copy()  # type: ignore
        tn_concatenated.extend(self._operator.cuquantum_interleaved)  # type: ignore
        tn_concatenated.extend(self._ket.cuquantum_interleaved)  # type: ignore
        return tn_concatenated  # type: ignore
  # type: ignore
  # type: ignore
def tk_to_tensor_network(tkc: Circuit) -> List[Union[NDArray, List]]:  # type: ignore
    """Converts pytket circuit into a tensor network.  # type: ignore
  # type: ignore
    Args:  # type: ignore
        tkc: Circuit.  # type: ignore
  # type: ignore
    Returns:  # type: ignore
        A tensor network representing the input circuit in the interleaved format  # type: ignore
        (list).  # type: ignore
    """  # type: ignore
    return TensorNetwork(tkc).cuquantum_interleaved  # type: ignore
