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
        truncation_bonds = [  # From q0 to the common ancestor
            path_q0[:i] for i in reversed(range(len(common_path) + 1, len(path_q0) + 1))
        ]
        truncation_bonds += [  # From the common ancestor to q1
            path_q1[:i] for i in range(len(common_path) + 1, len(path_q1) + 1)
        ]

        if self._cfg.truncation_fidelity < 1:
            # Truncate as much as possible before violating the truncation fidelity
            self._fidelity_bound_truncation_largest_first(truncation_bonds)

        else:
            # Truncate so that all bonds have dimension less or equal to chi
            self._chi_sequential_bond_truncation(truncation_bonds)

        return self

    def _fidelity_bound_truncation_largest_first(
        self,
        truncation_bonds: list[RootPath],
    ) -> None:
        """Truncate as much as possible before exceeding the truncation fidelity.

        We first identify the bond in ``truncation_bonds`` with the largest dimension
        and truncate it as much as possible before violating the truncation fidelity
        and update the bound to the fidelity with the induced error. Then, we repeat
        with the next largest bond (now with tighter fidelity threshold) and stop when
        we find the first bond that cannot have a single dimension truncated.

        Args:
            truncation_bonds: A list of bonds (provided by their RootPath address).
                This is expected to provide the bonds in the path in the TTN from one
                leaf node to another.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        # Sort the bonds in decreasing order in terms of their dimension
        truncation_bonds.sort(
            key=lambda bond: self.get_dimension(bond, DirTTN.PARENT),
            reverse=True,
        )

        # Keep track of the fidelity threshold (tighten it after each truncation)
        fidelity_threshold = self._cfg.truncation_fidelity

        # Truncate each bond (in order) down to the fidelity threshold
        for bond_address in truncation_bonds:
            dimension_before = self.get_dimension(bond_address, DirTTN.PARENT)

            self._logger.debug(
                "Truncating bond of dimension " f"{dimension_before} at {bond_address}."
            )

            # Canonicalise to this bond (unsafely, so we must reintroduce bond_tensor)
            bond_tensor = self.canonicalise(bond_address, unsafe=True)

            # Apply SVD decomposition to truncate as much as possible before exceeding
            # a `discarded_weight_cutoff` of `1 - fidelity_threshold`.
            self._logger.debug(f"Truncating to target fidelity={fidelity_threshold}")

            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                discarded_weight_cutoff=1 - fidelity_threshold,
                partition="U",  # Contract S directly into U (towards the child node)
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
            trunc_fidelity = 1.0 - svd_info.discarded_weight
            self.fidelity *= trunc_fidelity
            dimension_after = V.shape[0]

            # Report to logger
            self._logger.debug(f"Truncation done. Truncation fidelity={trunc_fidelity}")
            self._logger.debug(
                f"Reduced bond dimension from {dimension_before} to {dimension_after}."
            )

            # We require that self._cfg.truncation_fidelity > prod(each_trunc_fidelity)
            # since the RHS lower bounds the actual update factor to the fidelity. We
            # ensure this inequality is satisfied by dividing the fidelity_threshold
            # by the fidelity of this truncation. This makes the threshold tighter,
            # since trunc_fidelity is less than 1, so division causes a higher value
            # of fidelity_threshold.
            fidelity_threshold /= trunc_fidelity

            # Contract U and V into the TTN. This reintroduces the data of bond_tensor
            # back into the TTN, as required by ``canonicalise(.., unsafe=True)``.
            self._contract_decomp_bond_tensor_into_ttn(U, V, bond_address)

            # Since S was contracted with U, and the U contracted with the child node,
            # the child node loses its canonical form.
            self.nodes[bond_address].canonical_form = None

            # Stop truncating if there was no reduction of dimension in this iteration
            if dimension_before == dimension_after:
                break

    def _chi_sequential_bond_truncation(
        self,
        truncation_bonds: list[RootPath],
    ) -> None:
        """Truncate all bonds in the input list to have a dimension of chi or lower.

        The list ``truncation_bonds`` is explored sequentially, truncating the bonds
        one by one.

        Args:
            truncation_bonds: A list of bonds (provided by their RootPath address).
                The list must be ordered in such a way that consecutive bonds have
                a common tensor and such that the first and last bonds correspond to
                the parent bonds of leaf nodes. Hence, this provides the bonds in the
                path in the TTN from one leaf node to another.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        # Identify what is the level in the tree at which the ancestor node of all
        # of these bonds is
        ancestor_level = min(len(bond_address) for bond_address in truncation_bonds)
        towards_root = True  # First half of truncation_bonds is path towards ancestor

        for bond_address in truncation_bonds:
            # Canonicalise to this bond (unsafely, so we must reintroduce bond_tensor)
            bond_tensor = self.canonicalise(bond_address, unsafe=True)

            # Flip ``towards_root`` if we have reached the common ancestor
            # i.e. if the ``bond_tensor`` needs to go towards a child tensor rather
            # than towards the parent
            if len(bond_address) == ancestor_level:
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
                f"Reduced bond dimension from {bond_tensor.shape[0]} to {V.shape[0]}."
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
