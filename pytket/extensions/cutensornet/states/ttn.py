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
from __future__ import annotations  # type: ignore
import warnings
from typing import Optional
from enum import IntEnum

from random import random  # type: ignore
import math  # type: ignore
import numpy as np  # type: ignore

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
try:
    import cuquantum as cq  # type: ignore
    from cuquantum.cutensornet import tensor  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)

from pytket.circuit import Command, Op, OpType, Qubit
from pytket.pauli import Pauli, QubitPauliString

from pytket.extensions.cutensornet.general import set_logger

from .general import CuTensorNetHandle, Tensor, Config


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


class TTN:
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
    ):
        """Initialise a TTN on the computational state ``|0>``.

        Note:
            A ``libhandle`` should be created via a ``with CuTensorNet() as libhandle:``
            statement. The device where the TTN is stored will match the one specified
            by the library handle.

            The current implementation requires the keys of ``qubit_partition`` to be
            integers from ``0`` to ``2^l - 1`` for some ``l``. The cost of applying
            gates between qubits on ``qubit_partition[i]`` and ``qubit_partition[j]``
            scales exponentially on ``|i-j|``.

        Args:
            libhandle: The cuTensorNet library handle that will be used to carry out
                tensor operations on the TTN.
            qubits: A partition of the qubits in the circuit into disjoint groups.
            config: The object describing the configuration for simulation.

        Raises:
            ValueError: If the keys of ``qubit_partition`` do not range from ``0`` to
                ``2^l - 1`` for some ``l``.
            ValueError: If a ``Qubit`` is repeated in ``qubit_partition``.
            NotImplementedError: If the value of ``truncation_fidelity`` in ``config``
                is smaller than one.
        """
        self._lib = libhandle
        self._cfg = config
        self._logger = set_logger("TTN", level=config.loglevel)
        self.fidelity = 1.0
        self.nodes: dict[RootPath, TreeNode] = dict()
        self.qubit_position: dict[Qubit, tuple[RootPath, int]] = dict()

        if self._cfg.truncation_fidelity < 1:
            raise NotImplementedError(
                "Truncation fidelity mode not currently implemented on TTN."
            )

        n_groups = len(qubit_partition)
        if n_groups == 0:  # There's no initialisation to be done
            return None

        n_levels = math.floor(math.log2(n_groups))
        if n_groups != 2**n_levels:
            raise ValueError(
                "The number of entries in qubit_partition must be a power of two."
            )

        # Create the TreeNodes of the different groups of qubits
        for k, qubits in qubit_partition.items():
            if k < 0 or k >= n_groups:
                raise ValueError(
                    f"The keys of qubit_partition must range from 0 to {n_groups-1}."
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
                        f"Qubit {q} appears in multiple entries of qubit_partition."
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

    def apply_gate(self, gate: Command) -> TTN:
        """Apply the gate to the TTN.

        Note:
            Only single-qubit gates and two-qubit gates are supported.

        Args:
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
            RuntimeError: If gate acts on more than 2 qubits.
        """
        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        self._logger.debug(f"Applying gate {gate}")

        if len(gate.qubits) == 1:
            self._apply_1q_gate(gate.qubits[0], gate.op)

        elif len(gate.qubits) == 2:
            self._apply_2q_gate(gate.qubits[0], gate.qubits[1], gate.op)

        else:
            # NOTE: This could be supported if gate acts on same group of qubits
            raise RuntimeError(
                "Gates must act on only 1 or 2 qubits! "
                + f"This is not satisfied by {gate}."
            )

        return self

    def vdot(self, other: TTN) -> complex:
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
        if self.get_qubits != other.get_qubits:
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
        interleaved_rep = tt1 + tt2 + [[]]  # Discards dim=1 bonds with []
        result = cq.contract(
            *interleaved_rep,
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"samples": 1}
        )

        self._logger.debug(f"Result from vdot={result}")
        return complex(result)

    def get_interleaved_representation(self, conj: bool = False) -> list[Union[Tensor, str]]:
        """Returns the interleaved representation of the TTN used by cuQuantum.

        Args:
            conj: If True, all tensors are conjugated and bonds IDs are prefixed
                with * (except physical bonds). Defaults to False.
        """
        self._logger.debug("Creating interleaved representation...")

        # Auxiliar dictionary of physical bonds to qubit IDs
        qubit_id = {location: str(qubit) for qubit, location in self.qubit_position.items()}

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
            if node.isleaf:
                bonds = []
                for b in range(len(node.tensor.shape) - 1):
                    bonds.append(qubit_id[(path,b)])
                bonds.append(parentID)
            else:
                bonds = [parentID+"0", parentID+"1", parentID]

            interleaved_rep.append(bonds)
            self._logger.debug(f"Bond IDs: {bonds}")

        return interleaved_rep

    def get_qubits(self) -> set[Qubit]:
        """Returns the set of qubits that this TTN is defined on."""
        return set(self.qubit_position.keys())

    def get_dimension(self, path: RootPath, direction: DirTTN) -> int:
        """Returns the dimension of bond ``dir`` of the node at ``path``.

        Args:
            path: The path to a node in the TTN.
            direction: The direction of the bond.

        Returns:
            The dimension of the bond between the node and its parent.

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

        self._logger.debug(
            "Successfully copied a TTN "
            f"of size {new_ttn.get_byte_size() / 2**20} MiB."
        )
        return new_ttn

    def _apply_1q_gate(self, qubit: Qubit, gate: Op) -> TTN:
        raise NotImplementedError(
            "TTN is a base class with no contraction algorithm implemented."
            + " You must use a subclass of TTN, such as TTNxGate."
        )

    def _apply_2q_gate(self, q0: Qubit, q1: Qubit, gate: Op) -> TTN:
        raise NotImplementedError(
            "TTN is a base class with no contraction algorithm implemented."
            + " You must use a subclass of TTN, such as TTNxGate."
        )
