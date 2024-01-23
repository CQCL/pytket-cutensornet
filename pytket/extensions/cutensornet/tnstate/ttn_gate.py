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
            options=options,
            optimize={"path": [(0, 1), (1, 2), (0, 1)]},
        )
        self.nodes[common_path].canonical_form = None
        self._logger.debug(
            f"New tensor of shape {self.nodes[common_path].tensor.shape} and "
            f"size (MiB)={self.nodes[common_path].tensor.nbytes / 2**20}"
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
                indices = "fcab,Fpab,cCp->fCF"
            else:
                indices = "fcab,Fpab,Ccp->CfF"

            self._logger.debug(f"Adding funnels at {path}.")
            self.nodes[path].tensor = cq.contract(
                indices,
                child_funnel,
                parent_funnel,
                self.nodes[path].tensor,
                options=options,
                optimize={"path": [(1, 2), (0, 1)]},
            )
            self.nodes[path].canonical_form = None
            self._logger.debug(
                f"New tensor of shape {self.nodes[path].tensor.shape} and "
                f"size (MiB)={self.nodes[path].tensor.nbytes / 2**20}"
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
            result_bonds = aux_bonds
            result_bonds[bond] = "b"
            result_bonds[-1] = "f"

            self._logger.debug(f"Adding funnels at leaf node at {path}.")
            leaf_node.tensor = cq.contract(
                leaf_node.tensor,
                node_bonds,
                self._create_funnel_tensor(path, DirTTN.PARENT),
                ["f", "p", "a", "b"],
                result_bonds,
                options=options,
                optimize={"path": [(0, 1)]},
            )
            leaf_node.canonical_form = None
            self._logger.debug(
                f"New tensor of shape {leaf_node.tensor.shape} and "
                f"size (MiB)={leaf_node.tensor.nbytes / 2**20}"
            )

        # Truncate (if needed) bonds along the path from q0 to q1
        bonds_from_q0_to_ancestor = [
            path_q0[:i] for i in reversed(range(len(common_path) + 1, len(path_q0) + 1))
        ]
        bonds_from_ancestor_to_q1 = [
            path_q1[:i] for i in range(len(common_path) + 1, len(path_q1) + 1)
        ]

        if self._cfg.truncation_fidelity < 1:
            # Truncate as much as possible before violating the truncation fidelity
            self._fidelity_bound_sequential_weighted_truncation(
                bonds_from_q0_to_ancestor, bonds_from_ancestor_to_q1
            )

        else:
            # Truncate so that all bonds have dimension less or equal to chi
            self._chi_sequential_truncation(
                bonds_from_q0_to_ancestor, bonds_from_ancestor_to_q1
            )

        return self

    def _fidelity_bound_sequential_weighted_truncation(
        self,
        bonds_from_q0_to_ancestor: list[RootPath],
        bonds_from_ancestor_to_q1: list[RootPath],
    ) -> None:
        """Truncate as much as possible up to the truncation fidelity.

        Our target is to assign a local truncation fidelity `f_i` to each bond `i` in
        the input lists so that the lower bound of the fidelity satisfies:

            self.fidelity * prod(f_i) < self.fidelity * truncation_fidelity         (A)

        Let e_i = 1 - f_i, where we refer to `e_i` as the "truncation error at bond i".
        We can use that when e_i is close to zero, the bound:

            prod(1 - e_i)   >   1 - sum(e_i)                                        (B)

        is fairly tight, with an inaccuracy of an additive O(e_i^2) term. Hence, for
        simplicity we take prod(f_i) ~ 1 - sum(e_i). Let

            `admissible_error`   =   1 - `truncation_fidelity`                 (C)

        and assign each e_i = w_i * `admissible_error` where 0 < w_i < 1 is a weight
        factor such that sum(w_i) = 1. Thus, if each bond `i` is truncated to a fidelity

            f_i = 1 - w_i * `admissible_error`                                      (D)

        then the total fidelity factor on the LHS of equation (A) should approximate
        `truncation_fidelity`. There is risk of overshooting with truncation and
        endinf up with a new `self.fidelity` slightly lower than the target, but this
        should be fine in practice, since `self.fidelity` is a lower bound anyway.
        Each of the `w_i` weight factors is assigned depending on the bond dimension,
        with larger bonds given higher weight, so they are truncated more aggressively.

        Args:
            bonds_from_q0_to_ancestor: A list of bonds (each as their RootPath address).
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
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
        admissible_error = 1 - self._cfg.truncation_fidelity

        # Combine the two lists of bonds, but remember at which entry the direction
        # of the path is switched from going towards root to going towards leaves.
        truncation_bonds = bonds_from_q0_to_ancestor + bonds_from_ancestor_to_q1
        switch_direction_at = len(bonds_from_q0_to_ancestor)
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
        # Sanity check: reached the common ancestor and changed direction
        assert not towards_root

    def _chi_sequential_truncation(
        self,
        bonds_from_q0_to_ancestor: list[RootPath],
        bonds_from_ancestor_to_q1: list[RootPath],
    ) -> None:
        """Truncate all bonds in the input lists to have a dimension of chi or lower.

        The lists of bonds are explored sequentially, truncating the bonds
        one by one.

        Args:
            bonds_from_q0_to_ancestor: A list of bonds (each as their RootPath address).
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
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        # Combine the two lists of bonds, but remember at which entry the direction
        # of the path is switched from going towards root to going towards leaves.
        truncation_bonds = bonds_from_q0_to_ancestor + bonds_from_ancestor_to_q1
        switch_direction_at = len(bonds_from_q0_to_ancestor)
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
