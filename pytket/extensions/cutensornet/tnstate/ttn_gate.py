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

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
try:
    import cuquantum as cq  # type: ignore
    from cuquantum.cutensornet import tensor  # type: ignore
    from cuquantum.cutensornet.experimental import contract_decompose  # type: ignore
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

        # Contract
        new_tensor = cq.contract(
            node_tensor,
            node_bonds,
            gate_tensor,
            ["o", "i"],
            result_bonds,
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"path": [(0, 1)]},
        )

        # Update ``self.nodes``
        # NOTE: Canonicalisation of the node does not change
        self.nodes[path].tensor = new_tensor
        return self

    def _apply_2q_gate(self, q0: Qubit, q1: Qubit, gate: Op) -> TTNxGate:
        """Applies the 2-qubit gate to the TTN.

        Truncation is automatically applied according to the parameters
        in the ``Config`` object passed to this ``TTN``.
        The TTN is converted to canonical form before truncating.

        Args:
            q0: The 0-th qubit the gate acts upon.
            q1: The 1-st qubit the gate acts upon.
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

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
            result_bonds = aux_bonds
            result_bonds[bond_q0] = "o0"
            result_bonds[bond_q1] = "o1"

            self.nodes[path_q0].tensor = cq.contract(
                self.nodes[path_q0].tensor,
                node_bonds,
                gate_tensor,
                gate_bond,
                result_bonds,
                options=options,
                optimize={"path": [(0, 1)]},
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

        # We begin by canonicalising to the left child bond of the common ancestor.
        # This canonicalisation could be done later (just before truncation), but
        # doing it now will prevent the need to recanonicalise the tensors that have
        # grown (by a factor of x16) when introducing this gate.
        # The choice of the left child bond is arbitrary, any bond in the TTN that
        # is in the arc connecting qL to qR would have worked.
        #
        # NOTE: In fact, none of the tensors that are affected by the gate need to
        #   be canonicalised ahead of time, but I don't expect the saving would be
        #   particularly noticeable, and it'd require some non-trivial refactoring
        #   of `canonicalise()`.
        self.canonicalise(
            center=(*common_path, DirTTN.LEFT)
        )

        self._logger.debug(f"Applying gate to the TTN.")

        # Glossary of bond IDs
        # a -> the input bond of the gate on q0
        # b -> the input bond of the gate on q1
        # A -> the output bond of the gate on q0
        # B -> the output bond of the gate on q1
        # l -> left child bond of the TTN node
        # r -> right child bond of the TTN node
        # p -> the parent bond of the TTN node
        # s -> the shared bond resulting from a decomposition
        # chr(x) -> bond of the x-th qubit in a leaf node
        gate_bonds = "ABab"

        # The overall strategy is to connect the `a` bond of the gate tensor to
        # the corresponding bond for `q0` in the TTN (so that its bond `A`) becomes
        # the new physical bond for `q0`. However, bonds `b` and `B` corresponding to
        # `q1` are left open. We combine this `gate_tensor` with the leaf node of `q0`
        # and QR-decompose the result; where the Q tensor will be the new
        # (canonicalised) leaf node and R becomes our `msg_tensor`. The latter contains
        # the open bonds `b` and `B` and our objective is to "push" this `msg_tensor`
        # through the TTN towards the leaf node of `q1`. Here, "push through" means
        # contract with the next tensor, and apply QR decomposition, so that the
        # `msg_tensor` carrying `b` and `B` ends up one bond closer to `q1`.
        # Once `msg_tensor` is directly connected to the leaf node containing `q1`, we
        # just need to contract them, connecting `b` to `q1`, with `B` becoming the
        # new physical bond.
        #
        # The `msg_tensor` has four bonds. Our convention will be that the first bond
        # always corresponds to `B`, the second bond is `b`, the third bond connects
        # it to the TTN in the child direction and the fourth connects it to the TTN
        # in the `DirTTN.PARENT` direction. If we label the third bond with `l`, then
        # the fourth bond will be labelled `L` (and vice versa). Same for `r` and `p`.

        # We begin applying the gate to the TTN by contracting `gate_tensor` into the
        # leaf node containing `q0`, with the `b` and `B` bonds of the latter left open.
        # We immediately QR-decompose the resulting tensor,
        leaf_node = self.nodes[path_q0]
        n_qbonds = len(leaf_node.tensor.shape) - 1  # Num of qubit bonds
        leaf_bonds = "".join(
            "a" if x == bond_q0 else chr(x)
            for x in range(n_qbonds)
        ) + "p"
        Q_bonds = "".join(
            "A" if x == bond_q0 else chr(x)
            for x in range(n_qbonds)
        ) + "s"
        R_bonds = "Bbsp"  # The `msg_tensor`

        # Apply the contraction followed by a QR decomposition
        leaf_node.tensor, msg_tensor = contract_decompose(
            f"{leaf_bonds},{gate_bonds}->{Q_bonds},{R_bonds}",
            leaf_node.tensor,
            gate_tensor,
            algorithm={"qr_method": True},
            options=options,
            optimize={"path": [(0, 1)]},
        )
        # Update the canonical form of the leaf node
        leaf_node.canonical_form = DirTTN.PARENT

        # We must push the `msg_tensor` all the way to the common ancestor
        # of `q0` and `q1`.
        bonds_from_q0_to_ancestor = [
            path_q0[:i] for i in reversed(range(len(common_path) + 1, len(path_q0) + 1))
        ]
        # Sanity checks:
        assert all(len(bond_addresses) != len(common_path))
        assert len(bond_addresses[0]) == len(path_q0)
        assert len(bond_addresses[1]) < len(bond_addresses[0])

        # For all of these nodes; push `msg_tensor` through to their parent bond
        for child_bond in bond_addresses[:-1]:  # Doesn't do it on common ancestor!
            child_dir = child_bond[-1]
            parent_bond = child_bond[:-1]
            node = self.nodes[parent_bond]

            node_bonds = "lrp"
            msg_bonds = "BbLl" if child_dir == DirTTN.LEFT else "BbRr"
            Q_bonds = "Lrs" if child_dir == DirTTN.LEFT else "lRs"
            R_bonds = "Bbsp"  # The new `msg_tensor`

            # Apply the contraction followed by a QR decomposition
            node.tensor, msg_tensor = contract_decompose(
                f"{node_bonds},{msg_bonds}->{Q_bonds},{R_bonds}",
                node.tensor,
                msg_tensor,
                algorithm={"qr_method": True},
                options=options,
                optimize={"path": [(0, 1)]},
            )
            # Update the canonical form of the node
            node.canonical_form = DirTTN.PARENT

        # The `msg_tensor` is now on a child bond of the common ancestor.
        # We must push it through to the other child node.
        child_bond = bond_addresses[-1]  # This is where msg_tensor currently is
        child_dir = child_bond[-1]
        parent_bond = child_bond[:-1]
        common_ancestor_node = self.nodes[parent_bond]

        node_bonds = "lrp"
        msg_bonds = "BbLl" if child_dir == DirTTN.LEFT else "BbRr"
        Q_bonds = "Lsp" if child_dir == DirTTN.LEFT else "sRp"
        R_bonds = "Bbrs" if child_dir == DirTTN.LEFT else "Bbls"  # The new `msg_tensor`

        # Apply the contraction followed by a QR decomposition
        common_ancestor_node.tensor, msg_tensor = contract_decompose(
            f"{node_bonds},{msg_bonds}->{Q_bonds},{R_bonds}",
            common_ancestor_node.tensor,
            msg_tensor,
            algorithm={"qr_method": True},
            options=options,
            optimize={"path": [(0, 1)]},
        )
        # Update the canonical form of the node
        if child_dir == DirTTN.LEFT:
            common_ancestor_node.canonical_form = DirTTN.RIGHT
        else:
            common_ancestor_node.canonical_form = DirTTN.LEFT

        # We must push the `msg_tensor` from the common ancestor to the leaf node
        # containing `q1`.
        bonds_addresses = [
            path_q1[:i] for i in range(len(common_path) + 1, len(path_q1) + 1)
        ]
        # Sanity checks:
        assert all(len(bond_addresses) != len(common_path))
        assert len(bond_addresses[-1]) == len(path_q1)
        assert len(bond_addresses[0]) < len(bond_addresses[1])

        # For all of these nodes; push `msg_tensor` through to their child bond
        for child_bond in bond_addresses[1:]:  # Skip common ancestor: already pushed
            child_dir = child_bond[-1]
            parent_bond = child_bond[:-1]
            node = self.nodes[parent_bond]

            node_bonds = "lrp"
            msg_bonds = "BbpP"
            Q_bonds = "srP" if child_dir == DirTTN.LEFT else "lsP"
            R_bonds = "Bbls" if child_dir == DirTTN.LEFT else "Bbrs"  # New `msg_tensor`

            # Apply the contraction followed by a QR decomposition
            node.tensor, msg_tensor = contract_decompose(
                f"{node_bonds},{msg_bonds}->{Q_bonds},{R_bonds}",
                node.tensor,
                msg_tensor,
                algorithm={"qr_method": True},
                options=options,
                optimize={"path": [(0, 1)]},
            )
            # Update the canonical form of the node
            node.canonical_form = child_dir

        # Finally, the `msg_tensor` is in the parent bond of the leaf node of `q1`.
        # All we need to do is contract the `msg_tensor` into the leaf.
        leaf_node = self.nodes[path_q1]
        n_qbonds = len(leaf_node.tensor.shape) - 1  # Num of qubit bonds
        leaf_bonds = "".join(
            "b" if x == bond_q1 else chr(x)  # Connect `b` to `q1`
            for x in range(n_qbonds)
        ) + "p"
        msg_bonds = "BbpP"
        result_bonds = "".join(
            "B" if x == bond_q1 else chr(x)  # `B` becomes the new physical bond `q1`
            for x in range(n_qbonds)
        ) + "P"

        # Apply the contraction
        leaf_node.tensor = cq.contract(
            f"{leaf_bonds},{msg_bonds}->{result_bonds}",
            leaf_node.tensor,
            msg_tensor
            options=options,
            optimize={"path": [(1, 2), (0, 1)]},
        )
        # The leaf node lost its canonical form
        leaf_node.canonical_form = None

        # Truncate (if needed) bonds along the arc from `q1` to `q0`.
        # We truncate in this direction to take advantage of the canonicalisation
        # of the TTN we achieved while pushing the `msg_tensor` from `q0` to `q1`.
        trunc_paths = [  # From q1 to the common ancestor
            path_q1[:i] for i in reversed(range(len(common_path) + 1, len(path_q1) + 1))
        ]
        trunc_paths += [  # From the common ancestor to q0
            path_q0[:i] for i in range(len(common_path) + 1, len(path_q0) + 1)
        ]

        towards_root = True
        for path in trunc_paths:
            # Canonicalise to this bond (unsafely, so we must reintroduce bond_tensor)
            bond_tensor = self.canonicalise(path, unsafe=True)

            # Apply SVD decomposition on bond_tensor and truncate up to
            # `self._cfg.chi`. Ask cuTensorNet to contract S directly into U/V and
            # normalise the singular values so that the sum of its squares is equal
            # to one (i.e. the TTN is a normalised state after truncation).
            self._logger.debug(
                f"Truncating at {path} to (or below) chosen chi={self._cfg.chi}"
            )

            options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                max_extent=self._cfg.chi,
                partition="V" if towards_root else "U",
                normalization="L2",  # Sum of squares equal 1
            )

            U, S, V, svd_info = tensor.decompose(
                "cp->cs,sp",
                bond_tensor,
                method=svd_method,
                options=options,
                return_info=True,
            )
            assert S is None  # Due to "partition" option in SVDMethod

            # discarded_weight is calculated within cuTensorNet as:
            #                             sum([s**2 for s in S'])
            #     discarded_weight = 1 - -------------------------
            #                             sum([s**2 for s in S])
            # where S is the list of original singular values and S' is the set of
            # singular values that remain after truncation (before normalisation).
            # It can be shown that the fidelity |<psi|phi>|^2 (for |phi> and |psi>
            # unit vectors before and after truncation) is equal to 1 - disc_weight.
            #
            # We multiply the fidelity of the current step to the overall fidelity
            # to keep track of a lower bound for the fidelity.
            this_fidelity = 1.0 - svd_info.discarded_weight
            self.fidelity *= this_fidelity

            # Contract V to the parent node of the bond
            direction = path[-1]
            if direction == DirTTN.LEFT:
                indices = "lrp,sl->srp"
            else:
                indices = "lrp,sr->lsp"
            self.nodes[path[:-1]].tensor = cq.contract(
                indices,
                self.nodes[path[:-1]].tensor,
                V,
                options=options,
                optimize={"path": [(0, 1)]},
            )

            # Contract U to the child node of the bond
            if self.nodes[path].is_leaf:
                n_qbonds = (
                    len(self.nodes[path].tensor.shape) - 1
                )  # Total number of physical bonds in this node
                node_bonds = [f"q{x}" for x in range(n_qbonds)] + ["p"]
            else:
                node_bonds = ["l", "r", "p"]
            result_bonds = node_bonds.copy()
            result_bonds[-1] = "s"

            self.nodes[path].tensor = cq.contract(
                self.nodes[path].tensor,
                node_bonds,
                U,
                ["p", "s"],
                result_bonds,
                options=options,
                optimize={"path": [(0, 1)]},
            )
            # With these two contractions, bond_tensor has been reintroduced, as
            # required when calling ``canonicalise(.., unsafe=True)``

            # The next node in the path towards qR loses its canonical form, since
            # S was contracted to it (either via U or V)
            if towards_root:
                self.nodes[path[:-1]].canonical_form = None
            else:
                self.nodes[path].canonical_form = None

            # Report to logger
            self._logger.debug(f"Truncation done. Truncation fidelity={this_fidelity}")
            self._logger.debug(
                f"Reduced bond dimension from {bond_tensor.shape[0]} to {V.shape[0]}."
            )

        return self
