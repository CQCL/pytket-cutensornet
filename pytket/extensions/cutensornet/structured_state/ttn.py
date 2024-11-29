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
from __future__ import annotations  # type: ignore
import warnings
from typing import Optional, Union
from enum import IntEnum

from random import Random  # type: ignore
import math  # type: ignore
import numpy as np  # type: ignore
from numpy.typing import NDArray  # type: ignore

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
try:
    import cuquantum as cq  # type: ignore
    from cuquantum.cutensornet import tensor  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)

from pytket.circuit import Qubit, Bit
from pytket.pauli import QubitPauliString

from pytket.extensions.cutensornet.general import CuTensorNetHandle, set_logger

from .general import Config, StructuredState, Tensor


class DirTTN(IntEnum):
    """An enum to refer to relative directions within the TTN."""

    PARENT = -1
    LEFT = 0
    RIGHT = 1


# An alias for the TTN path from root to a TreeNode
RootPath = tuple[DirTTN, ...]


class TreeNode:
    """Represents a single tensor in the TTN.

    The shape of the tensor agrees with the convention set in ``DirTTN``, meaning
    that ``tensor.shape[DirTTN.PARENT]`` corresponds to the dimension of the bond
    connecting this tree node with its parent. Notice that, since DirTTN.PARENT is
    -1, this is always the last entry.

    In the case the TreeNode is a leaf, it will contain only one virtual bond
    (the parent) and as many physical bonds as qubits in the group it represents.
    These qubits will correspond to bonds from ``0`` to ``len(tensor.shape)-2``.
    """

    def __init__(self, tensor: Tensor, is_leaf: bool = False):
        self.tensor = tensor
        self.is_leaf = is_leaf
        self.canonical_form: Optional[DirTTN] = None

    def copy(self) -> TreeNode:
        new_node = TreeNode(
            self.tensor.copy(),
            is_leaf=self.is_leaf,
        )
        new_node.canonical_form = self.canonical_form
        return new_node


class TTN(StructuredState):
    """Represents a state as a Tree Tensor Network.

    Attributes:
        nodes (dict[RootPath, TreeNode]): A dictionary providing the tree node
            of the given root path in the TTN.
        qubit_position (dict[pytket.circuit.Qubit, tuple[RootPath, int]]): A dictionary
            mapping circuit qubits to their address in the TTN.
        fidelity (float): A lower bound of the fidelity, obtained by multiplying
            the fidelities after each contraction. The fidelity of a contraction
            corresponds to ``|<psi|phi>|^2`` where ``|psi>`` and ``|phi>`` are the
            states before and after truncation (assuming both are normalised).
    """

    def __init__(
        self,
        libhandle: CuTensorNetHandle,
        qubit_partition: dict[int, list[Qubit]],
        config: Config,
        bits: Optional[list[Bit]] = None,
    ):
        """Initialise a TTN on the computational state ``|0>``.

        Note:
            A ``libhandle`` should be created via a ``with CuTensorNet() as libhandle:``
            statement. The device where the TTN is stored will match the one specified
            by the library handle.

            The current implementation requires the keys of ``qubit_partition`` to be
            integers from ``0`` to ``2^l - 1`` for some ``l``.

        Args:
            libhandle: The cuTensorNet library handle that will be used to carry out
                tensor operations on the TTN.
            qubit_partition: A partition of the qubits in the circuit into disjoint
                groups, describing the hierarchical structure of the TTN. Each key
                identifies a leaf of the TTN, with its corresponding value indicating
                the list of qubits represented by the leaf. The leaves are numbered
                from left to right on a planar representation of the tree. Hence, the
                smaller half of the keys correspond to leaves in the left subtree and
                the rest are in the right subtree; providing recursive bipartitions.
            config: The object describing the configuration for simulation.

        Raises:
            ValueError: If the keys of ``qubit_partition`` do not range from ``0`` to
                ``2^l - 1`` for some ``l``.
            ValueError: If a ``Qubit`` is repeated in ``qubit_partition``.
        """
        self._lib = libhandle
        self._cfg = config
        self._logger = set_logger("TTN", level=config.loglevel)
        self._rng = Random()
        self._rng.seed(self._cfg.seed)

        if bits is None:
            self._bits_dict = dict()
        else:
            self._bits_dict = {b: False for b in bits}

        self.fidelity = 1.0
        self.nodes: dict[RootPath, TreeNode] = dict()
        self.qubit_position: dict[Qubit, tuple[RootPath, int]] = dict()

        n_groups = len(qubit_partition)
        if n_groups == 0:  # There's no initialisation to be done
            pass
        else:
            n_levels = math.floor(math.log2(n_groups))
            if n_groups != 2**n_levels:
                raise ValueError(
                    "The number of entries in qubit_partition must be a power of two."
                )

            # Create the TreeNodes of the different groups of qubits
            for k, qubits in qubit_partition.items():
                if k < 0 or k >= n_groups:
                    raise ValueError(
                        f"Keys of qubit_partition must range from 0 to {n_groups-1}."
                    )

                # Calculate the root path of this group
                path = []
                for l in reversed(range(n_levels)):
                    if k < 2**l:
                        path.append(DirTTN.LEFT)
                    else:
                        path.append(DirTTN.RIGHT)
                        k -= 2**l

                # Add each qubit to the qubit_position dictionary
                for i, q in enumerate(qubits):
                    if q in self.qubit_position:
                        raise ValueError(
                            f"Qubit {q} appears more than once in qubit_partition."
                        )
                    self.qubit_position[q] = (tuple(path), i)

                # This tensor has a physical bond per qubit and one virtual bond at the
                # end for the parent (dim=1)
                shape = tuple([2] * len(qubits) + [1])
                # Initialise the tensor of this group of qubits to |0>
                tensor = cp.zeros(shape=shape, dtype=self._cfg._complex_t)
                ket_zero_entry = tuple(0 for _ in shape)  # Index 0 on all bonds
                tensor[ket_zero_entry] = 1  # Amplitude of |0> set to 1

                # Create the TreeNode
                node = TreeNode(tensor, is_leaf=True)
                self.nodes[tuple(path)] = node

            # Create the internal TreeNodes
            paths: list[list[DirTTN]] = [[]]
            for _ in range(n_levels):
                # Create the TreeNode at this path
                for p in paths:
                    tensor = cp.ones(shape=(1, 1, 1), dtype=self._cfg._complex_t)
                    self.nodes[tuple(p)] = TreeNode(tensor)
                # Generate the paths for the next level
                paths = [
                    p + [direction]
                    for p in paths
                    for direction in [DirTTN.LEFT, DirTTN.RIGHT]
                ]
            self._logger.debug(f"qubit_position={self.qubit_position}")
            self._logger.debug(f"All root paths: {list(self.nodes.keys())}")

    def is_valid(self) -> bool:
        """Verify that the TTN object is valid.

        Specifically, verify that the TTN does not exceed the dimension limit ``chi``
        specified in the ``Config`` object, that physical bonds have dimension 2,
        that all tensors except the leaves are rank three and that tensors have shapes
        consistent with the bond dimensions.

        Returns:
            False if a violation was detected or True otherwise.
        """
        chi_ok = all(
            self.get_dimension(path, DirTTN.PARENT) <= self._cfg.chi
            for path in self.nodes.keys()
        )
        phys_ok = all(
            self.nodes[path].tensor.shape[bond] == 2
            for path, bond in self.qubit_position.values()
        )
        rank_ok = all(
            node.is_leaf or len(node.tensor.shape) == 3 for node in self.nodes.values()
        )
        shape_ok = all(
            self.get_dimension(path, DirTTN.PARENT)
            == self.get_dimension(path[:-1], path[-1])
            for path in self.nodes.keys()
            if len(path) != 0
        )
        shape_ok = shape_ok and self.get_dimension((), DirTTN.PARENT) == 1

        # Debugger logging
        self._logger.debug(
            "Checking validity of TTN... "
            f"chi_ok={chi_ok}, "
            f"phys_ok={phys_ok}, "
            f"rank_ok={rank_ok}, "
            f"shape_ok={shape_ok}"
        )
        return chi_ok and phys_ok and rank_ok and shape_ok

    def apply_unitary(self, unitary: NDArray, qubits: list[Qubit]) -> StructuredState:
        """Applies the unitary to the specified qubits of the StructuredState.

        Note:
            It is assumed that the matrix provided by the user is unitary. If this is
            not the case, the program will still run, but its behaviour is undefined.

        Args:
            unitary: The matrix to be applied as a NumPy or CuPy ndarray. It should
                either be a 2x2 matrix if acting on one qubit or a 4x4 matrix if acting
                on two.
            qubits: The qubits the unitary acts on. Only one qubit and two qubit
                unitaries are supported.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
            ValueError: If the number of qubits provided is not one or two.
            ValueError: If the size of the matrix does not match with the number of
                qubits provided.
        """
        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        if not isinstance(unitary, cp.ndarray):
            # Load the gate's unitary to the GPU memory
            unitary = unitary.astype(dtype=self._cfg._complex_t, copy=False)
            unitary = cp.asarray(unitary, dtype=self._cfg._complex_t)

        self._logger.debug(f"Applying unitary {unitary} on {qubits}.")

        if len(qubits) == 1:
            if unitary.shape != (2, 2):
                raise ValueError(
                    "The unitary introduced acts on one qubit but it is not 2x2."
                )
            self._apply_1q_unitary(unitary, qubits[0])

        elif len(qubits) == 2:
            if unitary.shape != (4, 4):
                raise ValueError(
                    "The unitary introduced acts on two qubits but it is not 4x4."
                )
            self._apply_2q_unitary(unitary, qubits[0], qubits[1])

        else:
            raise ValueError("Gates must act on only 1 or 2 qubits!")

        return self

    def apply_scalar(self, scalar: complex) -> TTN:
        """Multiplies the state by a complex number.

        Args:
            scalar: The complex number to be multiplied.

        Returns:
            ``self``, to allow for method chaining.
        """
        self.nodes[()].tensor *= scalar
        return self

    def apply_qubit_relabelling(self, qubit_map: dict[Qubit, Qubit]) -> TTN:
        """Relabels each qubit ``q`` as ``qubit_map[q]``.

        This does not apply any SWAP gate, nor it changes the internal structure of the
        state. It simply changes the label of the physical bonds of the tensor network.

        Args:
            qubit_map: Dictionary mapping each qubit to its new label.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            ValueError: If any of the keys in ``qubit_map`` are not qubits in the state.
        """
        new_qubit_position = dict()
        for q_orig, q_new in qubit_map.items():
            # Check the qubit is in the state
            if q_orig not in self.qubit_position:
                raise ValueError(f"Qubit {q_orig} is not in the state.")
            # Apply the relabelling for this qubit
            new_qubit_position[q_new] = self.qubit_position[q_orig]

        self.qubit_position = new_qubit_position
        self._logger.debug(f"Relabelled qubits... {qubit_map}")
        return self

    def canonicalise(
        self, center: Union[RootPath, Qubit], unsafe: bool = False
    ) -> Tensor:
        """Canonicalise the TTN so that all tensors are isometries from ``center``.

        Args:
            center: Identifies the bond that is to become the center of the canonical
                form. If it is a ``RootPath`` it refers to the parent bond of
                ``self.nodes[center]``. If it is a ``Qubit`` it refers to its physical
                bond.
            unsafe: If True, the final state will be different than the starting one.
                Specifically, the information in the returned bond tensor at ``center``
                is removed from the TTN. It is expected that the caller will reintroduce
                the bond tensor after some processing (e.g. after SVD truncation).

        Returns:
            The bond tensor created at ``center`` when canonicalisation is complete.
            Applying SVD to this tensor yields the global SVD of the TTN.

        Raises:
            ValueError: If the ``center`` is ``tuple()``.
        """
        self._logger.debug(f"Canonicalising to {str(center)}")
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        if isinstance(center, Qubit):
            target_path = self.qubit_position[center][0]
            assert not unsafe  # Unsafe disallowed when ``center`` is a qubit
        elif center == ():
            raise ValueError("There is no bond at path ().")
        else:
            target_path = center

        # Separate nodes to be canonicalised towards children from those towards parent
        towards_child = []
        towards_parent = []
        for path in self.nodes.keys():
            # Nodes towards children are closer to the root and coincide in the path
            if len(path) < len(target_path) and all(
                path[l] == target_path[l] for l in range(len(path))
            ):
                towards_child.append(path)
            # If the center is a physical bond (qubit), its node is skipped
            elif path == target_path and isinstance(center, Qubit):
                continue
            # All other nodes are canonicalised towards their parent
            else:
                towards_parent.append(path)
        # Sanity checks
        assert len(towards_child) != 0
        assert len(towards_parent) != 0

        # Glossary of bond IDs
        # chr(x) -> bond of the x-th qubit in the node (if it is a leaf)
        # l -> left child bond of the TTN node
        # r -> right child bond of the TTN node
        # p -> parent bond of the TTN node
        # s -> bond between Q and R after decomposition

        # Canonicalise nodes towards parent, start from the furthest away from root
        for path in sorted(towards_parent, key=len, reverse=True):
            self._logger.debug(f"Canonicalising node at {path} towards parent.")

            # If already in desired canonical form, do nothing
            if self.nodes[path].canonical_form == DirTTN.PARENT:
                self._logger.debug("Skipping, already in canonical form.")
                continue

            # Otherwise, apply QR decomposition
            if self.nodes[path].is_leaf:
                n_qbonds = len(self.nodes[path].tensor.shape) - 1  # Num of qubit bonds
                q_bonds = "".join(chr(x) for x in range(n_qbonds))
                node_bonds = q_bonds + "p"
                Q_bonds = q_bonds + "s"
            else:
                node_bonds = "lrp"
                Q_bonds = "lrs"
            R_bonds = "sp"

            Q, R = tensor.decompose(
                node_bonds + "->" + Q_bonds + "," + R_bonds,
                self.nodes[path].tensor,
                method=tensor.QRMethod(),
                options=options,
            )

            # Update the tensor
            self.nodes[path].tensor = Q
            self.nodes[path].canonical_form = DirTTN.PARENT

            # Contract R with the parent node
            if path[-1] == DirTTN.LEFT:
                R_bonds = "sl"
                result_bonds = "srp"
            else:
                R_bonds = "sr"
                result_bonds = "lsp"
            node_bonds = "lrp"

            parent_node = self.nodes[path[:-1]]
            parent_node.tensor = cq.contract(
                R_bonds + "," + node_bonds + "->" + result_bonds,
                R,
                parent_node.tensor,
                options=options,
                optimize={"path": [(0, 1)]},
            )
            # The canonical form of the parent node is lost
            parent_node.canonical_form = None

            self._logger.debug(f"Node canonicalised. Shape: {Q.shape}")

        # Canonicalise the rest of the nodes, from the root up to the center
        for path in sorted(towards_child, key=len):
            # Identify the direction of the canonicalisation
            target_direction = target_path[len(path)]
            # Sanity checks
            assert not self.nodes[path].is_leaf
            assert target_direction != DirTTN.PARENT

            self._logger.debug(
                f"Canonicalising node at {path} towards {str(target_direction)}."
            )

            # If already in the desired canonical form, do nothing
            if self.nodes[path].canonical_form == target_direction:
                self._logger.debug("Skipping, already in canonical form.")
                continue

            # Otherwise, apply QR decomposition
            if target_direction == DirTTN.LEFT:
                Q_bonds = "srp"
                R_bonds = "ls"
            else:
                Q_bonds = "lsp"
                R_bonds = "rs"
            node_bonds = "lrp"

            Q, R = tensor.decompose(
                node_bonds + "->" + Q_bonds + "," + R_bonds,
                self.nodes[path].tensor,
                method=tensor.QRMethod(),
                options=options,
            )

            # If the child bond is not the center yet, contract R with child node
            child_path = tuple(list(path) + [target_direction])
            if child_path != target_path:
                child_node = self.nodes[child_path]

                # Contract R with the child node
                child_node.tensor = cq.contract(
                    "lrp,ps->lrs",
                    child_node.tensor,
                    R,
                    options=options,
                    optimize={"path": [(0, 1)]},
                )

                # The canonical form of the child node is lost
                child_node.canonical_form = None
                # Update the tensor
                self.nodes[path].tensor = Q
                self.nodes[path].canonical_form = target_direction

                self._logger.debug(f"Node canonicalised. Shape: {Q.shape}")

        # If ``center`` is not a physical bond, we are done canonicalising and R is
        # the tensor to return. Otherwise, we need to do a final contraction and QR
        # decomposition on the leaf node corresponding to ``target_path``.
        if isinstance(center, Qubit):
            self._logger.debug(
                f"Applying QR decomposition on leaf node at {target_path}."
            )

            leaf_node = self.nodes[target_path]
            n_qbonds = len(leaf_node.tensor.shape) - 1  # Number of qubit bonds
            q_bonds = "".join(chr(x) for x in range(n_qbonds))
            node_bonds = q_bonds + "p"
            new_bonds = q_bonds + "s"
            R_bonds = "ps"

            # Contract R with the leaf node
            leaf_node.tensor = cq.contract(
                node_bonds + "," + R_bonds + "->" + new_bonds,
                leaf_node.tensor,
                R,
                options=options,
                optimize={"path": [(0, 1)]},
            )

            # The canonical form of the leaf node is lost
            leaf_node.canonical_form = None
            # Update the parent tensor
            parent_path = target_path[:-1]
            self.nodes[parent_path].tensor = Q
            self.nodes[parent_path].canonical_form = target_path[-1]
            self._logger.debug(f"Node canonicalised. Shape: {Q.shape}")

            # Finally, apply QR decomposition on the leaf_node to obtain the R
            # tensor to be returned
            target_bond = self.qubit_position[center][1]
            Q_bonds = node_bonds[:target_bond] + "s" + node_bonds[target_bond + 1 :]
            R_bonds = chr(target_bond) + "s"

            Q, R = tensor.decompose(
                node_bonds + "->" + Q_bonds + "," + R_bonds,
                leaf_node.tensor,
                method=tensor.QRMethod(),
                options=options,
            )
            # Note: Since R is not contracted with any other tensor, we cannot update
            #   the leaf node to Q. That'd change the state represented by the TTN.

        # Otherwise, if ``unsafe`` is enabled, update the last tensor to Q
        elif unsafe:
            self.nodes[target_path[:-1]].tensor = Q
            self.nodes[target_path[:-1]].canonical_form = target_path[-1]

            self._logger.debug(f"Node canonicalised (unsafe!). Shape: {Q.shape}")

        self._logger.debug(
            f"Finished canonicalisation. Returning R tensor of shape {R.shape}"
        )
        return R

    def vdot(self, other: TTN) -> complex:  # type: ignore
        """Obtain the inner product of the two TTN: ``<self|other>``.

        It can be used to compute the squared norm of a TTN ``ttn`` as
        ``ttn.vdot(ttn)``. The tensors within the TTN are not modified.

        Note:
            The state that is conjugated is ``self``.

        Args:
            other: The other TTN.

        Returns:
            The resulting complex number.

        Raises:
            RuntimeError: If the two TTNs do not have the same qubits.
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
        """
        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        if len(self.qubit_position) != len(other.qubit_position):
            raise RuntimeError("Number of qubits do not match.")
        if self.get_qubits() != other.get_qubits():
            raise RuntimeError(
                "The sets of qubits are not the same."
                "\n\tself has {self.get_qubits()}"
                "\n\tother has {other.get_qubits()}"
            )
        if len(self.qubit_position) == 0:
            raise RuntimeError("There are no qubits in the TTN.")

        self._logger.debug("Applying vdot between two TTNs.")

        # We convert both TTNs to their interleaved representation and
        # contract them using cuQuantum. A single sample is enough for
        # contraction path optimisation, since there is little to optimise.
        ttn1 = self.get_interleaved_representation(conj=True)
        ttn2 = other.get_interleaved_representation(conj=False)
        interleaved_rep = ttn1 + ttn2 + [[]]  # Discards dim=1 bonds with []
        result = cq.contract(
            *interleaved_rep,
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"samples": 0},  # There is little to no optimisation to be done
        )

        self._logger.debug(f"Result from vdot={result}")
        return complex(result)

    def sample(self) -> dict[Qubit, int]:
        """Returns a sample from a Z measurement applied on every qubit.

        Notes:
            The contents of ``self`` are not updated. This is equivalent to applying
            ``state = self.copy()`` then ``state.measure(state.get_qubits())``.

        Returns:
            A dictionary mapping each qubit in the state to its 0 or 1 outcome.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    def measure(self, qubits: set[Qubit], destructive: bool = True) -> dict[Qubit, int]:
        """Applies a Z measurement on each of the ``qubits``.

        Notes:
            After applying this function, ``self`` will contain the normalised
            projected state.

        Args:
            qubits: The subset of qubits to be measured.
            destructive: If ``True``, the resulting state will not contain the
                measured qubits. If ``False``, these qubits will appear on the
                state corresponding to the measurement outcome. Defaults to ``True``.

        Returns:
            A dictionary mapping the given ``qubits`` to their measurement outcome,
            i.e. either ``0`` or ``1``.

        Raises:
            ValueError: If an element in ``qubits`` is not a qubit in the state.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    def postselect(self, qubit_outcomes: dict[Qubit, int]) -> float:
        """Applies a postselection, updates the states and returns its probability.

        Notes:
            After applying this function, ``self`` will contain the projected
            state over the non-postselected qubits.

            The resulting state has been normalised.

        Args:
            qubit_outcomes: A dictionary mapping a subset of qubits to their
                desired outcome value (either ``0`` or ``1``).

        Returns:
            The probability of this postselection to occur in a measurement.

        Raises:
            ValueError: If a key in ``qubit_outcomes`` is not a qubit in the state.
            ValueError: If a value in ``qubit_outcomes`` is other than ``0`` or ``1``.
            ValueError: If all of the qubits in the state are being postselected.
                Instead, you may wish to use ``get_amplitude()``.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    def expectation_value(self, pauli_string: QubitPauliString) -> float:
        """Obtains the expectation value of the Pauli string observable.

        Args:
            pauli_string: A pytket object representing a tensor product of Paulis.

        Returns:
            The expectation value.

        Raises:
            ValueError: If a key in ``pauli_string`` is not a qubit in the state.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    def get_fidelity(self) -> float:
        """Returns the current fidelity of the state."""
        return self.fidelity

    def get_statevector(self) -> np.ndarray:
        """Returns the statevector represented by the TTN, with qubits ordered
        in Increasing Lexicographic Order (ILO).
        Raises:
            ValueError: If there are no qubits left in the TTN.
        """
        if len(self.get_qubits()) == 0:
            raise ValueError("There are no qubits left in this TTN.")

        # Create the interleaved representation with all tensors
        interleaved_rep = self.get_interleaved_representation()

        # Specify the output bond IDs in ILO order
        output_bonds = []
        for q in sorted(self.get_qubits()):
            output_bonds.append(str(q))
        interleaved_rep.append(output_bonds)

        # Contract
        result_tensor = cq.contract(
            *interleaved_rep,
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"samples": 0},  # There is little to no optimisation to be done
        )

        # Convert to numpy vector and flatten
        statevector: np.ndarray = cp.asnumpy(result_tensor).flatten()
        return statevector

    def get_amplitude(self, state: int) -> complex:
        """Returns the amplitude of the chosen computational state.

        Notes:
            The result is equivalent to ``self.get_statevector[b]``, but this method
            is faster when querying a single amplitude.

        Args:
            state: The integer whose bitstring describes the computational state.
                The qubits in the bitstring are in increasing lexicographic order.

        Returns:
            The amplitude of the computational state in the TTN.
        """

        interleaved_rep = self.get_interleaved_representation()
        ilo_qubits = sorted(self.get_qubits())

        for i, q in enumerate(ilo_qubits):
            # Create the tensors for each qubit in ``state``
            bitvalue = 1 if state & 2 ** (len(ilo_qubits) - i - 1) else 0
            tensor = cp.zeros(shape=(2,), dtype=self._cfg._complex_t)
            tensor[bitvalue] = 1
            # Append it to the interleaved representation
            interleaved_rep.append(tensor)
            interleaved_rep.append([str(q)])  # The bond
        # Ignore the dim=1 tensors in the output
        interleaved_rep.append([])

        # Contract
        result = cq.contract(
            *interleaved_rep,
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"samples": 0},  # There is little to no optimisation to be done
        )

        self._logger.debug(f"Amplitude of state {state} is {result}.")
        return complex(result)

    def get_qubits(self) -> set[Qubit]:
        """Returns the set of qubits that this TTN is defined on."""
        return set(self.qubit_position.keys())

    def get_interleaved_representation(
        self, conj: bool = False
    ) -> list[Union[Tensor, str]]:
        """Returns the interleaved representation of the TTN used by cuQuantum.

        Args:
            conj: If True, all tensors are conjugated and bonds IDs are prefixed
                with * (except physical bonds). Defaults to False.
        """
        self._logger.debug("Creating interleaved representation...")

        # Auxiliar dictionary of physical bonds to qubit IDs
        qubit_id = {
            location: str(qubit) for qubit, location in self.qubit_position.items()
        }

        interleaved_rep = []
        for path, node in self.nodes.items():
            # Append the tensor
            if conj:
                interleaved_rep.append(node.tensor.conj())
            else:
                interleaved_rep.append(node.tensor)

            # Create the ID for the parent bond
            parentID = "".join(str(int(d)) for d in path)
            if conj:
                parentID = "*" + parentID

            # Append the bonds
            if node.is_leaf:
                bonds = []
                for b in range(len(node.tensor.shape) - 1):
                    bonds.append(qubit_id[(path, b)])
                bonds.append(parentID)
            else:
                bonds = [parentID + "0", parentID + "1", parentID]

            interleaved_rep.append(bonds)
            self._logger.debug(f"Bond IDs: {bonds}")

        return interleaved_rep

    def get_dimension(self, path: RootPath, direction: DirTTN) -> int:
        """Returns the dimension of bond ``dir`` of the node at ``path``.

        Args:
            path: The path to a node in the TTN.
            direction: The direction of the bond.

        Returns:
            The dimension of the specified bond.

        Raises:
            ValueError: If ``path`` is not in the TTN.
        """
        if path not in self.nodes:
            raise ValueError(f"The path {path} is not in the TTN.")

        dim: int = self.nodes[path].tensor.shape[direction]
        return dim

    def get_byte_size(self) -> int:
        """
        Returns:
            The number of bytes the TTN currently occupies in GPU memory.
        """
        return sum(node.tensor.nbytes for node in self.nodes.values())

    def get_device_id(self) -> int:
        """
        Returns:
            The identifier of the device (GPU) where the tensors are stored.
        """
        return int(self.nodes[tuple()].tensor.device)

    def update_libhandle(self, libhandle: CuTensorNetHandle) -> None:
        """Update the ``CuTensorNetHandle`` used by this ``TTN`` object. Multiple
        objects may use the same handle.

        Args:
            libhandle: The new cuTensorNet library handle.

        Raises:
            RuntimeError: If the device (GPU) where ``libhandle`` was initialised
                does not match the one where the tensors of the TTN are stored.
        """
        if libhandle.device_id != self.get_device_id():
            raise RuntimeError(
                "Device of libhandle is not the one where the TTN is stored.",
                f"{libhandle.device_id} != {self.get_device_id()}",
            )
        self._lib = libhandle

    def copy(self) -> TTN:
        """
        Returns:
            A deep copy of the TTN on the same device.
        """

        # Create a dummy object
        new_ttn = TTN(self._lib, qubit_partition=dict(), config=self._cfg.copy())
        # Copy all data
        new_ttn.fidelity = self.fidelity
        new_ttn.nodes = {path: node.copy() for path, node in self.nodes.items()}
        new_ttn.qubit_position = self.qubit_position.copy()

        # If the user has set a seed, assume that they'd want every copy
        # to behave in the same way, so we copy the RNG state
        if self._cfg.seed is not None:
            # Setting state (rather than just copying the seed) allows for the
            # copy to continue from the same point in the sequence of random
            # numbers as the original copy
            new_ttn._rng.setstate(self._rng.getstate())
        # Otherwise, samples will be different between copies, since their
        # self._rng will be initialised from system randomnes when seed=None.

        self._logger.debug(
            "Successfully copied a TTN "
            f"of size {new_ttn.get_byte_size() / 2**20} MiB."
        )
        return new_ttn

    def _apply_1q_unitary(self, unitary: cp.ndarray, qubit: Qubit) -> TTN:
        raise NotImplementedError(
            "TTN is a base class with no contraction algorithm implemented."
            + " You must use a subclass of TTN, such as TTNxGate."
        )

    def _apply_2q_unitary(self, unitary: cp.ndarray, q0: Qubit, q1: Qubit) -> TTN:
        raise NotImplementedError(
            "TTN is a base class with no contraction algorithm implemented."
            + " You must use a subclass of TTN, such as TTNxGate."
        )

    def _flush(self) -> None:
        # Does nothing in the general MPS case; but children classes with batched
        # gate contraction will redefine this method so that the last batch of
        # gates is applied.
        return None
