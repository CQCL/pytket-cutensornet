# Copyright 2019 Cambridge Quantum Computing
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
from __future__ import annotations  # type: ignore
from enum import Enum  # type: ignore
import math
import random

import cupy as cp  # type: ignore
import numpy as np  # type: ignore
import cuquantum as cq  # type: ignore
import cuquantum.cutensornet as cutn  # type: ignore

from pytket.circuit import Command, Op, Qubit  # type: ignore
from .generic import Bond, Tensor


# An enum to refer to relative addresses from a tensor in the TTN.
TreeDir = Enum("TreeDir", ["LEFT", "RIGHT", "PARENT", "NONE"])


class TreeTensor(Tensor):
    """A wrapper around ``Tensor`` that contains information regarding its
    neighbouring tensors in a Tree Tensor Network.

    Attributes:
        neighbours (dict[TreeDir, TreeTensor]): Maps each direction to the
            ``TreeTensor`` adjacent to ``self`` in that direction. The root of
            the TTN does not have a ``PARENT`` and leaf nodes do not have ``LEFT``
            nor ``RIGHT``.
        canonicalised (TreeDir): If the tensor is an isometry (i.e. it has been
            canonicalised), indicate which bond is the one that remains open when
            applying the dagger isometry.
    """
    # The ``Tensor`` bonds in ``self.bonds`` are always ordered as follows:
    # [PARENT, LEFT, RIGHT]. These are present even in the case of the root
    # of the TTN, where PARENT is just a bond of dimension 1.

    def __init__(self, data: cp.ndarray, bonds: list[Bond]):
        super().__init__(data, bonds)
        self.neighbours: dict[TreeDir, TreeTensor] = dict()
        self.canonicalised = TreeDir.NONE

    def get_bond_at(self, direction: TreeDir) -> Bond:
        """Retrieve the ID of the bond of ``self`` in the requested direction.

        Args:
            direction: The relative direction of the bond within ``self``.

        Returns:
            The ID of the requested bond.
        """
        if direction == TreeDir.PARENT:
            return self.bonds[0]
        elif direction == TreeDir.LEFT:
            return self.bonds[1]
        elif direction == TreeDir.RIGHT:
            return self.bonds[2]
        else:
            raise RuntimeError(f"Invalid direction: {direction}")

    def set_bond_at(self, direction: TreeDir, bond: Bond) -> None:
        """Set the ID of the bond of ``self`` in the requested direction.

        Args:
            direction: The relative direction of the bond within ``self``.
            bond: The new ID of said bond.
        """
        if direction == TreeDir.PARENT:
            self.bonds[0] = bond
        elif direction == TreeDir.LEFT:
            self.bonds[1] = bond
        elif direction == TreeDir.RIGHT:
            self.bonds[2] = bond
        else:
            raise RuntimeError(f"Invalid direction: {direction}")

    def copy(self) -> TreeTensor:
        """Deep copy of the tensor.

        Returns:
            A deep copy of the TreeTensor.
        """
        new_tensor = TreeTensor(self.data.copy(), self.bonds.copy())
        new_tensor.neighbours = self.neighbours.copy()
        new_tensor.canonicalised = self.canonicalised
        return new_tensor


class TTN:
    """Parent class for state-based simulation using Tree Tensor Network
    representation.

    Attributes:
        chi (int): The maximum allowed dimension of a virtual bond.
        leaf_nodes (list[TensorNode]): A list of the leaf nodes in the TTN. Each
            leaf node has two physical bonds (two qubits).
        qubit_bond (dict[Qubit, Bond]): A dictionary mapping circuit qubits
            to the physical bond they correspond to in the TTN.
        fidelity (float): A lower bound of the fidelity, obtained by multiplying
            the fidelities after each contraction. The fidelity of a contraction
            corresponds to |<psi|phi>|^2 where |psi> and |phi> are the states
            before and after truncation (assuming both are normalised).
    """

    def __init__(
        self, qubits: list[Qubit], chi: int, float_precision: Optional[str] = None
    ):
        """Initialise a TTN on the computational state 0.

        Note:
            Use as ``with TTN(..) as ttn:`` so that cuQuantum
            handles are automatically destroyed at the end of execution.

        Args:
            qubits: The list of qubits of the circuit the TTN will simulate.
            chi: The maximum value the dimension of the virtual bonds
                is allowed to take. Higher implies better approximation but
                more computational resources.
            float_precision: Either 'float32' for single precision (32 bits per
                real number) or 'float64' for double precision (64 bits per real).
                Each complex number is represented using two of these real numbers.
                Default is 'float64'.
        """
        if chi < 2:
            raise Exception("The max virtual bond dimension (chi) must be >= 2.")

        allowed_precisions = ["float32", "float64"]
        if float_precision is None:
            float_precision = "float64"
        elif float_precision not in allowed_precisions:
            raise Exception(f"Value of float_precision must be in {allowed_precisions}")

        if float_precision == "float32":  # Single precision
            self._real_t = np.float32  # type: ignore
            self._complex_t = np.complex64  # type: ignore
        elif float_precision == "float64":  # Double precision
            self._real_t = np.float64  # type: ignore
            self._complex_t = np.complex128  # type: ignore

        #################################################
        # Create CuTensorNet library and memory handles #
        #################################################
        # TODO: Move this stuff (shared with MPS) to generic.py
        self._libhandle = cutn.create()
        self._stream = cp.cuda.Stream()
        dev = cp.cuda.Device()  # get current device

        if cp.cuda.runtime.runtimeGetVersion() < 11020:
            raise RuntimeError("Requires CUDA 11.2+.")
        if not dev.attributes["MemoryPoolsSupported"]:
            raise RuntimeError("Device does not support CUDA Memory pools")

        # Avoid shrinking the pool
        mempool = cp.cuda.runtime.deviceGetDefaultMemPool(dev.id)
        if int(cp.__version__.split(".")[0]) >= 10:
            # this API is exposed since CuPy v10
            cp.cuda.runtime.memPoolSetAttribute(
                mempool,
                cp.cuda.runtime.cudaMemPoolAttrReleaseThreshold,
                0xFFFFFFFFFFFFFFFF,  # = UINT64_MAX
            )

        # A device memory handler lets CuTensorNet manage its own GPU memory
        def malloc(size, stream):  # type: ignore
            return cp.cuda.runtime.mallocAsync(size, stream)

        def free(ptr, size, stream):  # type: ignore
            cp.cuda.runtime.freeAsync(ptr, stream)

        memhandle = (malloc, free, "memory_handler")
        cutn.set_device_mem_handler(self._libhandle, memhandle)

        #######################################
        # Initialise the TTN with a |0> state #
        #######################################

        self.chi = chi
        self.fidelity = 1.0

        n_qubits = len(qubits)
        if n_qubits == 0:  # There's no initialisation to be done
            return None
        elif n_qubits == 1:
            raise RuntimeError("Please, provide at least two qubits.")

        # If the number of qubits is not a power of 2, add dummies to the
        # ``qubits`` list so that it is a power of 2. Add these dummies
        # uniformly throughout to avoid biasing the size of subtrees.
        l = math.ceil(math.log(n_qubits, 2))  # Tree height
        dummies = [None for i in range(2**l - n_qubits)]
        qubits_and_dummies = []
        # The proportion of actual qubits vs qubits+dummies
        prop_q = n_qubits / (n_qubits + len(dummies))
        # Add qubits or dummies trying to maintain the proportion
        for _ in range(2**l):
            # If the proportion of qubits left is small, we need to add dummies
            if len(qubits) / (len(qubits) + len(dummies)) < prop_q:
                qubits_and_dummies.append(dummies.pop())
            else:
                qubits_and_dummies.append(qubits.pop())

        # Create the leaf nodes
        self.leaf_nodes = []
        self.qubit_bond = dict()
        # Each leaf node will contain a left bond and a right bond
        left_bonds = [q for i, q in enumerate(qubits_and_dummies) if i % 2 == 0]
        right_bonds = [q for i, q in enumerate(qubits_and_dummies) if i % 2 == 1]
        assert len(left_bonds) == len(right_bonds)

        for i, (q_left, q_right) in enumerate(zip(left_bonds, right_bonds)):
            # For each ``k``, ``qubits_and_dummies[k]`` is assigned to bond ID ``k+1``
            # Notice that ``qubits_and_dummies`` is in reverse order, but this should
            # not cause any issues.
            bond_ids = [2**l + 1 + i, 2*i + 1, 2*i + 2]  # [PARENT, LEFT, RIGHT]
            self.qubit_bond[q_left] = bond_ids[1]
            self.qubit_bond[q_right] = bond_ids[2]

            # Create the tensor of shape [PARENT, LEFT, RIGHT] on state |0>
            if q_left is None and q_right is None:  # Both are dummies
                leaf_tensor = cp.zeros(shape=(1, 1, 1), dtype=self._complex_t)
            elif q_left is None:  # Only the left one is a dummy
                leaf_tensor = cp.zeros(shape=(1, 1, 2), dtype=self._complex_t)
            elif q_right is None:  # Only the right one is a dummy
                leaf_tensor = cp.zeros(shape=(1, 2, 1), dtype=self._complex_t)
            else:  # Neither is a dummy
                leaf_tensor = cp.zeros(shape=(1, 2, 2), dtype=self._complex_t)
            # The state |0> has a 1 on the same entry and 0 in the rest
            leaf_tensor[0][0][0] = 1

            # Append the leaf node to the list
            self.leaf_nodes.append(TreeTensor(leaf_tensor, bond_ids))

        # Create all of the other tensors
        previous_layer = self.leaf_nodes
        for j in range(1, l):  # Create one layer at a time

            left_tensors = [t for i, t in enumerate(previous_layer) if i % 2 == 0]
            right_tensors = [t for i, t in enumerate(previous_layer) if i % 2 == 1]
            assert len(left_tensors) == len(right_tensors)
            previous_layer = []  # Clean up to start creating current layer

            for i, (t_left, t_right) in enumerate(zip(left_tensors, right_tensors)):
                # The tensor is just a placeholder for now, all dimensions are 1
                tensor_d = cp.ones(shape=(1, 1, 1), dtype=self._complex_t)
                bond_ids = [
                    j * 2**l + i + 1,  # PARENT bond ID (it skips some numbers)
                    t_left.get_bond_at(TreeDir.PARENT),  # Connect with left tensor
                    t_right.get_bond_at(TreeDir.PARENT),  # Connect with right tensor
                ]
                tensor = TreeTensor(tensor_d, bond_ids)
                # Give the references of children to parent and vice versa
                tensor.neighbours[TreeDir.LEFT] = t_left
                tensor.neighbours[TreeDir.RIGHT] = t_right
                t_left.neighbours[TreeDir.PARENT] = tensor
                t_right.neighbours[TreeDir.PARENT] = tensor

                # Keep track of this layer's tensors
                previous_layer.append(tensor)

        assert len(previous_layer) == 1  # Last layer is just the root tensor




