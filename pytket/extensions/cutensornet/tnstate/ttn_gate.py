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
import logging

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

from pytket.circuit import Op, Qubit
from .general import Tensor
from .ttn import TTN, DirTTN, RootPath


class TTNxGate(TTN):
    """Implements a gate-by-gate contraction algorithm to calculate the output state
    of a circuit as a ``TTN``.
    """

    def _apply_1q_gate(self, qubit: Qubit, gate: Op) -> TTNxGate:
        """Applies the 1-qubit gate to the TTN.

        This does not increase the dimension of any bond.

        Args:
            qubit: The qubit that this gate is applied to.
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.
        """

        # Load the gate's unitary to the GPU memory
        gate_unitary = gate.get_unitary().astype(dtype=self._cfg._complex_t, copy=False)
        gate_tensor = cp.asarray(gate_unitary, dtype=self._cfg._complex_t)

        path, target = self.qubit_position[qubit]
        node_tensor = self.nodes[path].tensor
        n_qbonds = (
            len(node_tensor.shape) - 1
        )  # Total number of physical bonds in this node

        # Glossary of bond IDs
        # qX -> where X is the X-th physical bond (qubit) in the TTN node
        # p  -> the parent bond of the TTN node
        # i  -> the input bond of the gate
        # o  -> the output bond of the gate

        node_bonds = [f"q{x}" for x in range(n_qbonds)] + ["p"]
        result_bonds = node_bonds.copy()
        node_bonds[target] = "i"  # Target bond must match with the gate input bond
        result_bonds[target] = "o"  # After contraction it matches the output bond

        interleaved_rep = [
            # The tensor of the TTN
            node_tensor,
            node_bonds,
            # The tensor of the gate
            gate_tensor,
            ["o", "i"],
            # The bonds of the resulting tensor
            result_bonds,
        ]

        # Contract
        new_tensor = cq.contract(*interleaved_rep)

        # Update ``self.nodes``
        # NOTE: Canonicalisation of the node does not change
        self.nodes[path].tensor = new_tensor
        return self

    def _apply_2q_gate(self, q0: Qubit, q1: Qubit, gate: Op) -> TTNxGate:
        """Applies the 2-qubit gate to the TTN.

        Truncation is automatically applied according to the paremeters
        in the ``Config`` object passed to this ``TTN``.
        The TTN is converted to canonical form before truncating.

        Args:
            q0: The 0-th qubit the gate acts upon.
            q1: The 1-st qubit the gate acts upon.
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.
        """

        # Load the gate's unitary to the GPU memory
        gate_unitary = gate.get_unitary().astype(dtype=self._cfg._complex_t, copy=False)
        gate_tensor = cp.asarray(gate_unitary, dtype=self._cfg._complex_t)
        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(gate_tensor, (2, 2, 2, 2))

        (path_q0, bond_q0) = self.qubit_position[q0]
        (path_q1, bond_q1) = self.qubit_position[q1]

        # If the two qubits are in the same leaf node, contract the gate with it.
        # There is no truncation needed.
        if path_q0 == path_q1:
            n_qbonds = (
                len(self.nodes[path_q0].tensor.shape) - 1
            )  # Total number of physical bonds in this node

            # Glossary of bond IDs
            # qX -> where X is the X-th physical bond (qubit) in the TTN node
            # p  -> the parent bond of the TTN node
            # i0 -> the input bond of the gate on q0
            # o0 -> the output bond of the gate on q0
            # i1 -> the input bond of the gate on q1
            # o1 -> the output bond of the gate on q1
            gate_bond = ["o0", "o1", "i0", "i1"]

            aux_bonds = [f"q{x}" for x in range(n_qbonds)] + ["p"]
            node_bonds = aux_bonds.copy()
            node_bonds[bond_q0] = "i0"
            node_bonds[bond_q1] = "i1"
            result_bonds = aux_bonds.copy()
            result_bonds[bond_q0] = "o0"
            result_bonds[bond_q1] = "o1"

            self.nodes[path_q0].tensor = cq.contract(
                self.nodes[path_q0].tensor,
                node_bonds,
                gate_tensor,
                gate_bond,
                result_bonds,
            )

            self._logger.debug(
                "The qubits the gate acts on are on the same group. "
                "Gate trivially applied, no dimensions changed."
            )
            return self

        # Otherwise, we must include the gate in the common ancestor tensor and
        # rewire the inputs and outputs. First, identify common path and direction
        common_dirs = []
        for d0, d1 in zip(path_q0, path_q1):
            if d0 == d1:
                common_dirs.append(d0)
            else:
                break
        common_path = tuple(common_dirs)
        (qL, qR) = (q0, q1) if path_q0[len(common_path)] == DirTTN.LEFT else (q1, q0)

        # Glossary of bond IDs
        # l -> the input bond of the gate on qL
        # r -> the input bond of the gate on qR
        # L -> the output bond of the gate on qL
        # R -> the output bond of the gate on qR
        # c -> a child bond of the TTN node
        # C -> another child bond of the TTN node
        # p -> the parent bond of the TTN node
        # f -> the output bond of a funnel
        # F -> the output bond of another funnel
        if qL == q0:
            gate_bonds = "LRlr"
        else:  # Implicit swap
            gate_bonds = "RLrl"

        # Update the common ancestor tensor with the gate and funnels
        left_funnel = self._create_funnel_tensor(common_path, DirTTN.LEFT)
        right_funnel = self._create_funnel_tensor(common_path, DirTTN.RIGHT)

        self._logger.debug(f"Including gate in tensor at {common_path}.")
        self.nodes[common_path].tensor = cq.contract(
            "cCp,fclL,FCrR," + gate_bonds + "->fFp",
            self.nodes[common_path].tensor,
            left_funnel,
            right_funnel,
            gate_tensor,
        )
        self.nodes[common_path].canonical_form = None
        self._logger.debug(
            f"New tensor size (MiB)={self.nodes[common_path].tensor.nbytes / 2**20}"
        )

        # For each bond along the path, add two identity dim 2 wires
        mid_paths = [
            (path_qX[:i], path_qX[i])
            for path_qX in [path_q0, path_q1]
            for i in range(len(common_path) + 1, len(path_qX))
        ]

        for path, child_dir in mid_paths:
            child_funnel = self._create_funnel_tensor(path, child_dir)
            parent_funnel = self._create_funnel_tensor(path, DirTTN.PARENT)

            if child_dir == DirTTN.LEFT:
                node_bonds = "cCp"
                result_bonds = "fCF"
            else:
                node_bonds = "Ccp"
                result_bonds = "CfF"

            self._logger.debug(f"Adding funnels at {path}.")
            self.nodes[path].tensor = cq.contract(
                "fcab,Fpab," + node_bonds + "->" + result_bonds,
                child_funnel,
                parent_funnel,
                self.nodes[path].tensor,
            )
            self.nodes[path].canonical_form = None
            self._logger.debug(
                f"New tensor size (MiB)={self.nodes[path].tensor.nbytes / 2**20}"
            )

        # Update the leaf tensors containing qL and qR
        for path, bond in [self.qubit_position[q] for q in [qL, qR]]:
            leaf_node = self.nodes[path]
            n_qbonds = (
                len(leaf_node.tensor.shape) - 1
            )  # Total number of physical bonds in this node

            aux_bonds = [f"q{x}" for x in range(n_qbonds)] + ["p"]
            node_bonds = aux_bonds.copy()
            node_bonds[bond] = "a"
            result_bonds = aux_bonds.copy()
            result_bonds[bond] = "b"

            self._logger.debug(f"Adding funnels at leaf node at {path}.")
            leaf_node.tensor = cq.contract(
                leaf_node.tensor,
                node_bonds,
                self._create_funnel_tensor(path, DirTTN.PARENT),
                ["f", "p", "a", "b"],
                result_bonds,
            )
            leaf_node.canonical_form = None
            self._logger.debug(
                f"New tensor size (MiB)={leaf_node.tensor.nbytes / 2**20}"
            )

        # Truncate to chi (if needed) on all bonds along the path from qL to qR
        # TODO!

    def _create_funnel_tensor(self, path: RootPath, direction: DirTTN) -> Tensor:
        """Creates a funnel tensor for the given bond.

        A funnel tensor is a reshape of an identity, merging three bonds to one.
        A funnel tensor has four bonds. It satisfies ``funnel.shape[0] == 4*dim``
        where ``dim`` is the dimension of the bond of ``path`` and ``direction``.
        Hence, the first bond of the funnel is the "merged" bond. The other three
        satisfy ``funnel.shape[1] == dim`` (this the original bond) and
        ``funnel.shape[x] == 2`` for ``x`` 2 and 3.
        """
        dim = self.get_dimension(path, direction)
        identity = cp.eye(4 * dim, dtype=self._cfg._complex_t)
        return cp.reshape(identity, (4 * dim, dim, 2, 2))
