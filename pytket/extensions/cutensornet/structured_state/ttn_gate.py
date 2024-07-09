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

from pytket.circuit import Qubit
from .ttn import TTN, DirTTN, RootPath


class TTNxGate(TTN):
    """Implements a gate-by-gate contraction algorithm to calculate the output state
    of a circuit as a ``TTN``.
    """

    def _apply_1q_unitary(self, unitary: cp.ndarray, qubit: Qubit) -> TTNxGate:
        """Applies the 1-qubit gate to the TTN.

        This does not increase the dimension of any bond.

        Args:
            unitary: The unitary to be applied.
            qubit: The qubit the unitary acts on.

        Returns:
            ``self``, to allow for method chaining.
        """
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
            unitary,
            ["o", "i"],
            result_bonds,
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"path": [(0, 1)]},
        )

        # Update ``self.nodes``
        # NOTE: Canonicalisation of the node does not change
        self.nodes[path].tensor = new_tensor
        return self

    def _apply_2q_unitary(self, unitary: cp.ndarray, q0: Qubit, q1: Qubit) -> TTNxGate:
        """Applies the 2-qubit gate to the TTN.

        The TTN is converted to canonical and truncation is applied if necessary.

        Args:
            unitary: The unitary to be applied.
            q0: The first qubit in the tuple |q0>|q1> the unitary acts on.
            q1: The second qubit in the tuple |q0>|q1> the unitary acts on.

        Returns:
            ``self``, to allow for method chaining.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(unitary, (2, 2, 2, 2))

        (path_q0, bond_q0) = self.qubit_position[q0]
        (path_q1, bond_q1) = self.qubit_position[q1]

        # Glossary of bond IDs
        # a -> the input bond of the gate on q0
        # b -> the input bond of the gate on q1
        # A -> the output bond of the gate on q0
        # B -> the output bond of the gate on q1
        # S -> the shared bond of the gate tensor's SVD
        # l -> left child bond of the TTN node
        # r -> right child bond of the TTN node
        # p -> the parent bond of the TTN node
        # s -> the shared bond resulting from a decomposition
        # chr(x) -> bond of the x-th qubit in a leaf node
        gate_bonds = "ABab"

        # If the two qubits are in the same leaf node, contract the gate with it.
        # No truncation is required.
        if path_q0 == path_q1:
            leaf_node = self.nodes[path_q0]
            n_qbonds = len(leaf_node.tensor.shape) - 1  # Num of qubit bonds
            aux_bonds = [chr(x) for x in range(n_qbonds)]
            aux_bonds[bond_q0] = "a"
            aux_bonds[bond_q1] = "b"
            leaf_bonds = "".join(aux_bonds) + "p"
            aux_bonds[bond_q0] = "A"
            aux_bonds[bond_q1] = "B"
            result_bonds = "".join(aux_bonds) + "p"

            self.nodes[path_q0].tensor = cq.contract(
                f"{leaf_bonds},{gate_bonds}->{result_bonds}",
                leaf_node.tensor,
                gate_tensor,
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
        self.canonicalise(center=(*common_path, DirTTN.LEFT))

        # Apply SVD on the gate tensor to remove any zero singular values ASAP
        svd_method = tensor.SVDMethod(
            abs_cutoff=self._cfg.zero,
            partition="U",  # Contract S directly into U
        )
        # Apply the SVD decomposition using the configuration defined above
        U, S, V = tensor.decompose(
            f"{gate_bonds}->SAa,SBb", gate_tensor, method=svd_method, options=options
        )
        assert S is None  # Due to "partition" option in SVDMethod

        # The overall strategy is to connect the `U` tensor above with the physical bond
        # for `q0` in the TTN, so that its bond `A` becomes the new physical bond and
        # the bond `S` is left dangling (open). We combine this `gate_tensor` with the
        # leaf node of `q0` and QR-decompose the result; where the Q tensor will be the
        # new (canonicalised) leaf node and R becomes our `msg_tensor`. The latter
        # contains the open bond `S` and our objective is to "push" this `msg_tensor`
        # through the TTN towards the leaf node of `q1`. Here, "push through" means
        # contract with the next tensor, and apply QR decomposition, so that the
        # `msg_tensor` carrying `b` and `B` ends up one bond closer to `q1`.
        # Once `msg_tensor` is directly connected to the leaf node containing `q1`, we
        # just need to apply the `V` tensor above to `q1` and connect its `S` bond with
        # that of the `msg_tensor`.
        bonds_to_q0 = [  # Bonds in the "arc" from the common ancestor to `q0`
            path_q0[:i] for i in range(len(common_path) + 1, len(path_q0) + 1)
        ]
        # Sanity checks:
        assert all(
            len(bond_address) != len(common_path) for bond_address in bonds_to_q0
        )
        assert len(bonds_to_q0) == 1 or len(bonds_to_q0[0]) < len(bonds_to_q0[1])
        assert len(bonds_to_q0[-1]) == len(path_q0)

        bonds_to_q1 = [  # Bonds in the "arc" from the common ancestor to `q1`
            path_q1[:i] for i in range(len(common_path) + 1, len(path_q1) + 1)
        ]
        # Sanity checks:
        assert all(
            len(bond_address) != len(common_path) for bond_address in bonds_to_q1
        )
        assert len(bonds_to_q1) == 1 or len(bonds_to_q1[0]) < len(bonds_to_q1[1])
        assert len(bonds_to_q1[-1]) == len(path_q1)

        # The `msg_tensor` has three bonds. Our convention will be that the first bond
        # always corresponds to `S`, the second bond connects the `msg_tensor`
        # to the TTN in the child direction and the third connects it to the TTN
        # in the `DirTTN.PARENT` direction. If we label the second bond with `l`, then
        # the third bond will be labelled `L` (and vice versa). Same for `r` and `p`.

        # We begin applying the gate to the TTN by contracting `U` into the
        # leaf node containing `q0`, with the `S` bond of the former left open.
        # We immediately QR-decompose the resulting tensor, so that Q becomes the new
        # (canonicalised) leaf node and R becomes the `msg_tensor` that we will be
        # "pushing" through the rest of the arc towards `q1`.
        leaf_node = self.nodes[path_q0]
        n_qbonds = len(leaf_node.tensor.shape) - 1  # Num of qubit bonds
        aux_bonds = [chr(x) for x in range(n_qbonds)]
        aux_bonds[bond_q0] = "a"
        leaf_bonds = "".join(aux_bonds) + "p"
        aux_bonds[bond_q0] = "A"
        Q_bonds = "".join(aux_bonds) + "s"
        R_bonds = "Ssp"  # The `msg_tensor`
        U_bonds = "SAa"

        # Apply the contraction followed by a QR decomposition
        leaf_node.tensor, msg_tensor = contract_decompose(
            f"{leaf_bonds},{U_bonds}->{Q_bonds},{R_bonds}",
            leaf_node.tensor,
            U,
            algorithm={"qr_method": tensor.QRMethod()},
            options=options,
            optimize={"path": [(0, 1)]},
        )
        # Update the canonical form of the leaf node
        leaf_node.canonical_form = DirTTN.PARENT

        # We must push the `msg_tensor` all the way to the common ancestor
        # of `q0` and `q1`.
        bond_addresses = list(reversed(bonds_to_q0))  # From `q0` to the ancestor

        # For all of these nodes; push `msg_tensor` through to their parent bond
        for child_bond in bond_addresses[:-1]:  # Doesn't do it on common ancestor!
            child_dir = child_bond[-1]
            parent_bond = child_bond[:-1]
            node = self.nodes[parent_bond]

            node_bonds = "lrp"
            msg_bonds = "SLl" if child_dir == DirTTN.LEFT else "SRr"
            Q_bonds = "Lrs" if child_dir == DirTTN.LEFT else "lRs"
            R_bonds = "Ssp"  # The new `msg_tensor`

            self._logger.debug(
                f"Pushing msg_tensor ({msg_tensor.nbytes // 2**20} MiB) through node "
                f"({node.tensor.nbytes // 2**20} MiB) at {parent_bond}."
            )

            # Apply the contraction followed by a QR decomposition
            node.tensor, msg_tensor = contract_decompose(
                f"{node_bonds},{msg_bonds}->{Q_bonds},{R_bonds}",
                node.tensor,
                msg_tensor,
                algorithm={"qr_method": tensor.QRMethod()},
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
        msg_bonds = "SLl" if child_dir == DirTTN.LEFT else "SRr"
        Q_bonds = "Lsp" if child_dir == DirTTN.LEFT else "sRp"
        R_bonds = "Srs" if child_dir == DirTTN.LEFT else "Sls"  # The new `msg_tensor`

        self._logger.debug(
            f"Pushing msg_tensor ({msg_tensor.nbytes // 2**20} MiB) through node "
            f"({common_ancestor_node.tensor.nbytes // 2**20} MiB) at {parent_bond}."
        )

        # Apply the contraction followed by a QR decomposition
        common_ancestor_node.tensor, msg_tensor = contract_decompose(
            f"{node_bonds},{msg_bonds}->{Q_bonds},{R_bonds}",
            common_ancestor_node.tensor,
            msg_tensor,
            algorithm={"qr_method": tensor.QRMethod()},
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
        bond_addresses = bonds_to_q1  # From ancestor to `q1`

        # For all of these nodes; push `msg_tensor` through to their child bond
        for child_bond in bond_addresses[1:]:  # Skip common ancestor: already pushed
            child_dir = child_bond[-1]
            parent_bond = child_bond[:-1]
            node = self.nodes[parent_bond]

            node_bonds = "lrp"
            msg_bonds = "SpP"
            Q_bonds = "srP" if child_dir == DirTTN.LEFT else "lsP"
            R_bonds = "Sls" if child_dir == DirTTN.LEFT else "Srs"  # New `msg_tensor`

            self._logger.debug(
                f"Pushing msg_tensor ({msg_tensor.nbytes // 2**20} MiB) through node "
                f"({node.tensor.nbytes // 2**20} MiB) at {parent_bond}."
            )

            # Apply the contraction followed by a QR decomposition
            node.tensor, msg_tensor = contract_decompose(
                f"{node_bonds},{msg_bonds}->{Q_bonds},{R_bonds}",
                node.tensor,
                msg_tensor,
                algorithm={"qr_method": tensor.QRMethod()},
                options=options,
                optimize={"path": [(0, 1)]},
            )
            # Update the canonical form of the node
            node.canonical_form = child_dir

        # Finally, the `msg_tensor` is in the parent bond of the leaf node of `q1`.
        # All we need to do is contract the `msg_tensor` and `V` into the leaf.
        leaf_node = self.nodes[path_q1]
        n_qbonds = len(leaf_node.tensor.shape) - 1  # Num of qubit bonds
        aux_bonds = [chr(x) for x in range(n_qbonds)]
        aux_bonds[bond_q1] = "b"  # Connect `b` to `q1`
        leaf_bonds = "".join(aux_bonds) + "p"
        msg_bonds = "SpP"
        V_bonds = "SBb"
        aux_bonds[bond_q1] = "B"  # `B` becomes the new physical bond `q1`
        result_bonds = "".join(aux_bonds) + "P"

        # Apply the contraction
        leaf_node.tensor = cq.contract(
            f"{leaf_bonds},{V_bonds},{msg_bonds}->{result_bonds}",
            leaf_node.tensor,
            V,
            msg_tensor,
            options=options,
            optimize={"path": [(0, 1), (0, 1)]},
        )
        # The leaf node lost its canonical form
        leaf_node.canonical_form = None

        # Truncate (if needed) bonds along the arc from `q1` to `q0`.
        # We truncate in this direction to take advantage of the canonicalisation
        # of the TTN we achieved while pushing the `msg_tensor` from `q0` to `q1`.
        if self._cfg.truncation_fidelity < 1:
            # Truncate as much as possible before violating the truncation fidelity
            self._fidelity_bound_sequential_weighted_truncation(
                list(reversed(bonds_to_q1)), bonds_to_q0
            )

        else:
            # Truncate so that all bonds have dimension less or equal to chi
            self._chi_sequential_truncation(list(reversed(bonds_to_q1)), bonds_to_q0)

        return self

    def _fidelity_bound_sequential_weighted_truncation(
        self,
        bonds_from_q1_to_ancestor: list[RootPath],
        bonds_from_ancestor_to_q0: list[RootPath],
    ) -> None:
        """Truncate as much as possible up to the truncation fidelity.

        Our target is to assign a local truncation fidelity `f_i` to each bond `i` in
        the input lists so that the lower bound of the fidelity satisfies:

        real_fidelity > self.fidelity * prod(f_i) > self.fidelity * trunc_fidelity  (A)

        Let e_i = 1 - f_i, where we refer to `e_i` as the "truncation error at bond i".
        We can use that when 0 < e_i < 1, the bound:

            prod(1 - e_i)   >   1 - sum(e_i)                                        (B)

        is fairly tight, with an inaccuracy of an additive O(e_i^2) term. Hence, as long
        as we satisfy

            1 - sum(e_i)    >    truncation_fidelity                                (C)

        the inqualities at (A) will be satisfied for our chosen f_i. Let

            admissible_error   =   1 - truncation_fidelity                          (D)

        and assign each e_i = w_i * admissible_error where 0 < w_i < 1 is a weight
        factor such that sum(w_i) = 1. With these choice, inequality (C) holds and,
        consequently, so does (A).

        Args:
            bonds_from_q1_to_ancestor: A list of bonds (each as their RootPath address).
                These bonds will be truncated. The list must be ordered in such a way
                that consecutive bonds share a common tensor and such that the first
                bond in the list corresponds to the leaf node that `q0` is assigned to
                and the last bond in the list corresponds to child bond of the common
                ancestor between the leaves of `q0` and `q1`.
            bonds_from_ancestor_q1: Same as above, but the list starts from the other
                child bond of the common ancestor and ends at the leaf node that `q1`
                is assigned to. Together, these two lists provide a path in the TTN
                from the leaf node of `q0` to the leaf node of `q1`.
        """
        self._logger.debug("Starting sequential weighted truncation (fidelity bound).")
        initial_fidelity = self.fidelity

        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
        admissible_error = 1 - self._cfg.truncation_fidelity

        # Combine the two lists of bonds, but remember at which entry the direction
        # of the path is switched from going towards root to going towards leaves.
        truncation_bonds = bonds_from_q1_to_ancestor + bonds_from_ancestor_to_q0
        switch_direction_at = len(bonds_from_q1_to_ancestor)
        towards_root = True  # First half of truncation_bonds is path towards ancestor

        # Obtain the dimension of each bond
        dimensions = [
            self.get_dimension(bond, DirTTN.PARENT) for bond in truncation_bonds
        ]
        # Assign the weight `w_i` of each bond.
        # NOTE: currently uses w_i = dim_i / sum(dim_i), for no other reason that it is
        #   simple. Better weight functions may exist and research on this is desirable.
        weights = [dim / sum(dimensions) for dim in dimensions]
        # Assign a fidelity `f_i` to each bond.
        bond_fidelities = [1 - w * admissible_error for w in weights]

        # Truncate each bond as much as possible up to its assigned bond fidelity
        for i, bond_address in enumerate(truncation_bonds):
            dimension_before = self.get_dimension(bond_address, DirTTN.PARENT)

            # Canonicalise to this bond (unsafely, so we must reintroduce bond_tensor)
            bond_tensor = self.canonicalise(bond_address, unsafe=True)

            # Flip ``towards_root`` if we have reached the common ancestor
            # i.e. if the ``bond_tensor`` needs to go towards a child tensor rather
            # than towards the parent
            if switch_direction_at == i:
                towards_root = False

            # Apply SVD decomposition to truncate as much as possible before exceeding
            # a `discarded_weight_cutoff` of `1 - f_i`. Contract S directly into U/V and
            # normalise the singular values so that the sum of its squares is equal
            # to one (i.e. the TTN is a normalised state after truncation).
            self._logger.debug(
                f"Truncating at {bond_address} to target fidelity={bond_fidelities[i]}"
            )

            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                discarded_weight_cutoff=1 - bond_fidelities[i],
                partition="V" if towards_root else "U",  # Contract S to parent or child
                normalization="L2",  # Sum of squares singular values must equal 1
            )

            # Apply the SVD decomposition using the configuration defined above
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
            dimension_after = V.shape[0]

            # Contract U and V into the TTN. This reintroduces the data of bond_tensor
            # back into the TTN, as required by ``canonicalise(.., unsafe=True)``.
            self._contract_decomp_bond_tensor_into_ttn(U, V, bond_address)

            # The next node in the path towards qR loses its canonical form, since
            # S was contracted to it (either via U or V)
            if towards_root:
                self.nodes[bond_address[:-1]].canonical_form = None
            else:
                self.nodes[bond_address].canonical_form = None

            # Report to logger
            self._logger.debug(f"Truncation done. Truncation fidelity={this_fidelity}")
            self._logger.debug(
                f"Reduced bond dimension from {dimension_before} to {dimension_after}."
            )

        self._logger.debug(
            "Finished sequential weighted truncation (fidelity bound). "
            f"Fidelity factor = {self.fidelity / initial_fidelity}"
        )

        # Sanity check: reached the common ancestor and changed direction
        assert not towards_root

    def _chi_sequential_truncation(
        self,
        bonds_from_q1_to_ancestor: list[RootPath],
        bonds_from_ancestor_to_q0: list[RootPath],
    ) -> None:
        """Truncate all bonds in the input lists to have a dimension of chi or lower.

        The lists of bonds are explored sequentially, truncating the bonds
        one by one.

        Args:
            bonds_from_q1_to_ancestor: A list of bonds (each as their RootPath address).
                These bonds will be truncated. The list must be ordered in such a way
                that consecutive bonds share a common tensor and such that the first
                bond in the list corresponds to the leaf node that `q0` is assigned to
                and the last bond in the list corresponds to child bond of the common
                ancestor between the leaves of `q0` and `q1`.
            bonds_from_ancestor_q1: Same as above, but the list starts from the other
                child bond of the common ancestor and ends at the leaf node that `q1`
                is assigned to. Together, these two lists provide a path in the TTN
                from the leaf node of `q0` to the leaf node of `q1`.
        """
        self._logger.debug("Starting sequential truncation (chi bound).")
        initial_fidelity = self.fidelity

        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        # Combine the two lists of bonds, but remember at which entry the direction
        # of the path is switched from going towards root to going towards leaves.
        truncation_bonds = bonds_from_q1_to_ancestor + bonds_from_ancestor_to_q0
        switch_direction_at = len(bonds_from_q1_to_ancestor)
        towards_root = True  # First half of truncation_bonds is path towards ancestor

        for i, bond_address in enumerate(truncation_bonds):
            dimension_before = self.get_dimension(bond_address, DirTTN.PARENT)

            # Canonicalise to this bond (unsafely, so we must reintroduce bond_tensor)
            bond_tensor = self.canonicalise(bond_address, unsafe=True)

            # Flip ``towards_root`` if we have reached the common ancestor
            # i.e. if the ``bond_tensor`` needs to go towards a child tensor rather
            # than towards the parent
            if switch_direction_at == i:
                towards_root = False

            # Apply SVD decomposition on bond_tensor and truncate up to
            # `self._cfg.chi`. Ask cuTensorNet to contract S directly into U/V and
            # normalise the singular values so that the sum of its squares is equal
            # to one (i.e. the TTN is a normalised state after truncation).
            self._logger.debug(
                f"Truncating at {bond_address} to (or below) chosen chi={self._cfg.chi}"
            )

            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                max_extent=self._cfg.chi,
                partition="V" if towards_root else "U",  # Contract S to parent or child
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
            dimension_after = V.shape[0]

            # Contract U and V into the TTN. This reintroduces the data of bond_tensor
            # back into the TTN, as required by ``canonicalise(.., unsafe=True)``.
            self._contract_decomp_bond_tensor_into_ttn(U, V, bond_address)

            # The next node in the path towards qR loses its canonical form, since
            # S was contracted to it (either via U or V)
            if towards_root:
                self.nodes[bond_address[:-1]].canonical_form = None
            else:
                self.nodes[bond_address].canonical_form = None

            # Report to logger
            self._logger.debug(f"Truncation done. Truncation fidelity={this_fidelity}")
            self._logger.debug(
                f"Reduced bond dimension from {dimension_before} to {dimension_after}."
            )

        self._logger.debug(
            "Finished sequential truncation (chi bound). "
            f"Fidelity factor = {self.fidelity / initial_fidelity}"
        )

        # Sanity check: reached the common ancestor and changed direction
        assert not towards_root

    def _contract_decomp_bond_tensor_into_ttn(
        self, U: cp.ndarray, V: cp.ndarray, bond_address: RootPath
    ) -> None:
        """Contracts a decomposed bond_tensor back into the TTN.

        Args:
            U: The tensor of the decomposition adjacent to the child node of the bond.
            V: The tensor of the decomposition adjacent to the parent node of the bond.
            bond_address: The address to the bond that was decomposed; explicitly, the
                DirTTN.PARENT bond of the corresponding child node.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        # Contract V to the parent node of the bond
        direction = bond_address[-1]
        if direction == DirTTN.LEFT:
            indices = "lrp,sl->srp"
        else:
            indices = "lrp,sr->lsp"
        self.nodes[bond_address[:-1]].tensor = cq.contract(
            indices,
            self.nodes[bond_address[:-1]].tensor,
            V,
            options=options,
            optimize={"path": [(0, 1)]},
        )

        # Contract U to the child node of the bond
        if self.nodes[bond_address].is_leaf:
            n_qbonds = (
                len(self.nodes[bond_address].tensor.shape) - 1
            )  # Total number of physical bonds in this node
            node_bonds = [f"q{x}" for x in range(n_qbonds)] + ["p"]
        else:
            node_bonds = ["l", "r", "p"]
        result_bonds = node_bonds.copy()
        result_bonds[-1] = "s"

        self.nodes[bond_address].tensor = cq.contract(
            self.nodes[bond_address].tensor,
            node_bonds,
            U,
            ["p", "s"],
            result_bonds,
            options=options,
            optimize={"path": [(0, 1)]},
        )
