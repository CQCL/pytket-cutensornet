# Copyright Quantinuum
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

import logging
import warnings

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)  # noqa: B028
try:
    from cuquantum.tensornet import contract, tensor  # type: ignore
    from cuquantum.tensornet.experimental import contract_decompose  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)  # noqa: B028

from pytket.circuit import Op, OpType, Qubit
from pytket.pauli import Pauli, QubitPauliString

from .mps import MPS, DirMPS


class MPSxGate(MPS):
    """Implements a gate-by-gate contraction algorithm to calculate the output state
    of a circuit as an ``MPS``. The algorithm is described in:
    https://arxiv.org/abs/2002.07730
    """

    def _apply_1q_unitary(self, unitary: cp.ndarray, qubit: Qubit) -> MPSxGate:
        """Applies the 1-qubit unitary to the MPS.

        This does not increase the dimension of any bond.

        Args:
            unitary: The unitary to be applied.
            qubit: The qubit the unitary acts on.

        Returns:
            ``self``, to allow for method chaining.
        """
        position = self.qubit_position[qubit]

        # Glossary of bond IDs
        # p -> physical bond of the MPS tensor
        # v -> one of the virtual bonds of the MPS tensor
        # V -> the other virtual bond of the MPS tensor
        # o -> the output bond of the gate

        T_bonds = "vVp"
        result_bonds = "vVo"
        gate_bonds = "op"

        # Contract
        new_tensor = contract(
            gate_bonds + "," + T_bonds + "->" + result_bonds,
            unitary,
            self.tensors[position],
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"path": [(0, 1)]},
        )

        # Update ``self.tensors``
        self.tensors[position] = new_tensor
        return self

    def _apply_2q_unitary(self, unitary: cp.ndarray, q0: Qubit, q1: Qubit) -> MPSxGate:
        """Applies the 2-qubit unitary to the MPS.

        Args:
            unitary: The unitary to be applied.
            q0: The first qubit in the tuple |q0>|q1> the unitary acts on.
            q1: The second qubit in the tuple |q0>|q1> the unitary acts on.

        Returns:
            ``self``, to allow for method chaining.
        """

        # If qubits are not adjacent, use an alternative approach
        if abs(self.qubit_position[q0] - self.qubit_position[q1]) != 1:
            return self._apply_2q_unitary_nonadjacent(unitary, q0, q1)
        # Otherwise, proceed as normal

        positions = [self.qubit_position[q0], self.qubit_position[q1]]
        l_pos = min(positions)
        r_pos = max(positions)

        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        # Always canonicalise. Even in the case of exact simulation (no truncation)
        # canonicalisation may reduce the bond dimension (thanks to reduced QR).
        self.canonicalise(l_pos, r_pos)

        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(unitary, (2, 2, 2, 2))

        # Glossary of bond IDs
        # l -> physical bond of the left tensor in the MPS
        # r -> physical bond of the right tensor in the MPS
        # L -> left bond of the outcome of the gate
        # R -> right bond of the outcome of the gate
        # S -> shared bond of the gate tensor's SVD
        # a,b,c -> the virtual bonds of the tensors

        if l_pos == positions[0]:  # noqa: SIM108
            gate_bonds = "LRlr"
        else:  # Implicit swap
            gate_bonds = "RLrl"

        # Apply SVD on the gate tensor to remove any zero singular values ASAP
        svd_method = tensor.SVDMethod(
            abs_cutoff=self._cfg.zero,
            partition="U",  # Contract S directly into U
        )
        # Apply the SVD decomposition using the configuration defined above
        U, S, V = tensor.decompose(
            f"{gate_bonds}->SLl,SRr", gate_tensor, method=svd_method, options=options
        )
        assert S is None  # Due to "partition" option in SVDMethod

        # Contract
        self._logger.debug("Contracting the two-qubit gate with its site tensors...")
        T = contract(
            "SLl,abl,SRr,bcr->acLR",
            U,
            self.tensors[l_pos],
            V,
            self.tensors[r_pos],
            options=options,
            optimize={"path": [(0, 1), (0, 1), (0, 1)]},
        )
        self._logger.debug(f"Intermediate tensor of size (MiB)={T.nbytes / 2**20}")  # noqa: G004

        if self._cfg.truncation_fidelity < 1:
            # Apply SVD decomposition to truncate as much as possible before exceeding
            # a `discarded_weight_cutoff` of `1 - self._cfg.truncation_fidelity`.
            self._logger.debug(
                f"Truncating to target fidelity={self._cfg.truncation_fidelity}"  # noqa: G004
            )

            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                discarded_weight_cutoff=1 - self._cfg.truncation_fidelity,
                partition="U",  # Contract S directly into U (named L in our case)
                normalization="L2",  # Sum of squares singular values must equal 1
            )

        else:
            # Apply SVD decomposition and truncate up to a `max_extent` (for the shared
            # bond) of `self._cfg.chi`.
            # If user did not provide a value for `chi`, this is still given a
            # default value that is so large that it causes no truncation at all.
            # Nevertheless, we apply SVD so that singular values below `self._cfg.zero`
            # are truncated.
            self._logger.debug(f"Truncating to (or below) chosen chi={self._cfg.chi}")  # noqa: G004

            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                max_extent=self._cfg.chi,
                partition="U",  # Contract S directly into U (named L in our case)
                normalization="L2",  # Sum of squares singular values must equal 1
            )

        # Apply the SVD decomposition using the configuration defined above
        L, S, R, svd_info = tensor.decompose(
            "acLR->asL,scR", T, method=svd_method, options=options, return_info=True
        )
        assert S is None  # Due to "partition" option in SVDMethod

        # Update fidelity if there was some truncation
        #
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
        # Report to logger
        self._logger.debug(f"Truncation done. Truncation fidelity={this_fidelity}")  # noqa: G004
        self._logger.debug(
            "Reduced virtual bond dimension from "  # noqa: G004
            f"{svd_info.full_extent} to {svd_info.reduced_extent}."
        )

        self.tensors[l_pos] = L
        self.tensors[r_pos] = R

        # If requested, provide info about memory usage.
        if self._logger.isEnabledFor(logging.INFO):
            # If-statetement used so that we only call `get_byte_size` if needed.
            self._logger.info(f"MPS size (MiB)={self.get_byte_size() / 2**20}")  # noqa: G004
            self._logger.info(f"MPS fidelity={self.fidelity}")  # noqa: G004

        return self

    def _apply_2q_unitary_nonadjacent(
        self, unitary: cp.ndarray, q0: Qubit, q1: Qubit
    ) -> MPSxGate:
        """Applies the 2-qubit unitary to the MPS between non-adjacent qubits.

        Args:
            unitary: The unitary to be applied.
            q0: The first qubit in the tuple |q0>|q1> the unitary acts on.
            q1: The second qubit in the tuple |q0>|q1> the unitary acts on.

        Returns:
            ``self``, to allow for method chaining.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        positions = [self.qubit_position[q0], self.qubit_position[q1]]
        l_pos = min(positions)
        r_pos = max(positions)

        # Always canonicalise. Even in the case of exact simulation (no truncation)
        # canonicalisation may reduce the bond dimension (thanks to reduced QR).
        self.canonicalise(l_pos, r_pos)

        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(unitary, (2, 2, 2, 2))

        # Glossary of bond IDs
        # p -> some physical bond of the MPS
        # l -> physical bond of the left tensor in the MPS
        # r -> physical bond of the right tensor in the MPS
        # L -> left bond of the outcome of the gate
        # R -> right bond of the outcome of the gate
        # s -> shared bond of the gate tensor's SVD
        # a,b -> virtual bonds of the MPS
        # m,M -> virtual bonds connected to the "message tensor"

        if l_pos == positions[0]:  # noqa: SIM108
            gate_bonds = "LRlr"
        else:  # Implicit swap
            gate_bonds = "RLrl"

        # Apply SVD on the gate tensor to remove any zero singular values ASAP
        svd_method = tensor.SVDMethod(
            abs_cutoff=self._cfg.zero,
            partition="U",  # Contract S directly into U (i.e. l_gate_tensor)
        )
        # Apply the SVD decomposition using the configuration defined above
        l_gate_tensor, S, r_gate_tensor = tensor.decompose(
            f"{gate_bonds}->sLl,sRr", gate_tensor, method=svd_method, options=options
        )
        assert S is None  # Due to "partition" option in SVDMethod

        # First, contract `l_gate_tensor` with the
        # MPS site tensor on the left position. Then, decompose the new tensor so
        # that a "message tensor" carrying the bond `s` is sent through the right
        # virtual bond. We do these two steps in a single `contract_decompose`
        self.tensors[l_pos], msg_tensor = contract_decompose(
            "sLl,abl->amL,smb",
            l_gate_tensor,
            self.tensors[l_pos],
            algorithm={"qr_method": tensor.QRMethod()},
            options=options,
            optimize={"path": [(0, 1)]},
        )

        # The site tensor is now in canonical form
        self.canonical_form[l_pos] = DirMPS.LEFT  # type: ignore

        # Next, "push" the `msg_tensor` through all site tensors between `l_pos`
        # and `r_pos`. Once again, this is just contract_decompose on each.
        for pos in range(l_pos + 1, r_pos):
            # Report to logger
            self._logger.debug(
                f"Pushing msg_tensor ({msg_tensor.nbytes // 2**20} MiB) through site "  # noqa: G004
                f"tensor ({self.tensors[pos].nbytes // 2**20} MiB) in position {pos}."
            )

            self.tensors[pos], msg_tensor = contract_decompose(
                "sam,mbp->aMp,sMb",
                msg_tensor,
                self.tensors[pos],
                algorithm={"qr_method": tensor.QRMethod()},
                options=options,
                optimize={"path": [(0, 1)]},
            )

            # The site tensor is now in canonical form
            self.canonical_form[pos] = DirMPS.LEFT  # type: ignore

        # Finally, contract the `msg_tensor` with the site tensor in `r_pos` and the
        # `r_gate_tensor` from the decomposition of `gate_tensor`
        self.tensors[r_pos] = contract(
            "sam,mbr,sRr->abR",
            msg_tensor,
            self.tensors[r_pos],
            r_gate_tensor,
            options=options,
            optimize={"path": [(0, 2), (0, 1)]},
        )
        # The site tensor is not in canonical form anymore
        self.canonical_form[r_pos] = None

        # Apply truncations between the leftmost and rightmost qubits
        self._truncate_path(l_pos, r_pos)

        return self

    def apply_cnx(self, controls: list[Qubit], target: Qubit) -> MPSxGate:
        """Applies a CnX gate to the MPS.

        Args:
            controls: The control qubits
            target: The target qubit

        Returns:
            ``self``, to allow for method chaining.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        pos_qubit_map = {self.qubit_position[q]: q for q in controls + [target]}
        l_pos = min(pos_qubit_map.keys())
        r_pos = max(pos_qubit_map.keys())
        t_pos = self.qubit_position[target]

        # Canonicalise
        self.canonicalise(l_pos, r_pos)

        # Glossary of bond IDs
        # p -> some physical bond of the MPS
        # P -> the new physical bond after application of the gate
        # a,b,c -> virtual bonds of the MPS
        # s -> a shared bond after decomposition
        # m,M -> virtual bonds connected to the "message tensor"

        # Define the "connection" tensors depending
        connection = {  # bonds PMpm
            False: cp.eye(4, dtype=self._cfg._complex_t),  # No connection, just IxI
            True: cp.asarray(  # Map |p,m> to |P,M> = |p,m&p>
                [
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=self._cfg._complex_t,
            ),
        }
        for p, t in connection.items():
            connection[p] = cp.reshape(t, (2, 2, 2, 2))

        # Create an initial value for "message tensor" from the left; state |1>
        lmsg_tensor = cp.zeros(2, dtype=self._cfg._complex_t)
        lmsg_tensor[1] = 1
        # Apply a Kronecker product with the identity of left bond of l_pos
        l_dim = self.get_virtual_dimensions(l_pos)[0]
        lmsg_tensor = contract(
            "m,ab->mab",
            lmsg_tensor,
            cp.eye(l_dim, dtype=self._cfg._complex_t),
            options=options,
            optimize={"path": [(0, 1)]},
        )

        # Update all of the tensor sites from `l_pos` to `t_pos` - 1
        for pos in range(l_pos, t_pos):
            # Identify if this is a control qubit
            is_control = pos in pos_qubit_map

            # Push the message tensor through the site tensor, updating it
            self.tensors[pos], lmsg_tensor = contract_decompose(
                "mab,bcp,PMpm->asP,Msc",
                lmsg_tensor,
                self.tensors[pos],
                connection[is_control],
                algorithm={"qr_method": tensor.QRMethod()},
                options=options,
                optimize={"path": [(0, 1), (0, 1)]},
            )

            # The site tensor is now in canonical form
            self.canonical_form[pos] = DirMPS.LEFT  # type: ignore

        # Repeat, but the other way from `r_pos` to `t_pos`
        # Create an initial value for "message tensor" from the right; state |1>
        rmsg_tensor = cp.zeros(2, dtype=self._cfg._complex_t)
        rmsg_tensor[1] = 1
        # Apply a Kronecker product with the identity of right bond of r_pos
        r_dim = self.get_virtual_dimensions(r_pos)[1]
        rmsg_tensor = contract(
            "m,bc->mbc",
            rmsg_tensor,
            cp.eye(r_dim, dtype=self._cfg._complex_t),
            options=options,
            optimize={"path": [(0, 1)]},
        )

        # Update all of the tensor sites from `r_pos` to `t_pos` + 1
        for pos in range(r_pos, t_pos, -1):
            # Identify if this is a control qubit
            is_control = pos in pos_qubit_map

            # Push the message tensor through the site tensor, updating it
            rmsg_tensor, self.tensors[pos] = contract_decompose(
                "abp,mbc,PMpm->Mas,scP",
                self.tensors[pos],
                rmsg_tensor,
                connection[is_control],
                algorithm={"qr_method": tensor.QRMethod()},
                options=options,
                optimize={"path": [(0, 1), (0, 1)]},
            )

            # The site tensor is now in canonical form
            self.canonical_form[pos] = DirMPS.RIGHT  # type: ignore

        # Contract both `lmsg_tensor` and `rmsg_tensor` with an AND to obtain the final
        # control signal and contract it with a tensor that applies X conditionally on
        # said signal (i.e. XOR).
        and_tensor = cp.asarray(  # Map |x,y> to |x&y>
            [
                [1, 1, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=self._cfg._complex_t,
        )
        and_tensor = cp.reshape(and_tensor, (2, 2, 2))  # Bonds Zxy (where Z is result)

        xor_tensor = cp.asarray(  # Map |q,c> to |q XOR c>
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=self._cfg._complex_t,
        )
        xor_tensor = cp.reshape(xor_tensor, (2, 2, 2))  # Bonds Zxy (where Z is result)

        self.tensors[t_pos] = contract(
            "lab,rcd,bcp,PpM,Mlr->adP",
            lmsg_tensor,
            rmsg_tensor,
            self.tensors[t_pos],
            xor_tensor,
            and_tensor,
            options=options,
            optimize={"path": [(3, 4), (0, 2), (0, 2), (0, 1)]},
        )
        # The site tensor is not in canonical form anymore
        self.canonical_form[t_pos] = None

        # Apply truncations between the leftmost and rightmost qubits
        self._truncate_path(l_pos, r_pos)

        return self

    def apply_pauli_gadget(self, pauli_str: QubitPauliString, angle: float) -> MPSxGate:
        """Applies the Pauli gadget to the MPS.

        Args:
            pauli_str: The Pauli string of the Pauli gadget
            angle: The angle in half turns

        Returns:
            ``self``, to allow for method chaining.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        pos_qubit_map = {
            self.qubit_position[q]: q
            for q, pauli in pauli_str.map.items()
            if pauli != Pauli.I
        }
        l_pos = min(pos_qubit_map.keys())
        r_pos = max(pos_qubit_map.keys())

        # Canonicalise
        self.canonicalise(l_pos, r_pos)

        # Glossary of bond IDs
        # p -> some physical bond of the MPS
        # P -> the new physical bond after application of the Pauli gadget
        # a,b,c -> virtual bonds of the MPS
        # s -> a shared bond after decomposition
        # m,M -> virtual bonds connected to the "message tensor"

        # First, create a "message tensor" containing the angle of the Pauli gadget
        msg_tensor = cp.zeros(2, dtype=self._cfg._complex_t)
        phase = 1j * cp.pi * angle / 2
        msg_tensor[0] = cp.exp(-phase)
        msg_tensor[1] = cp.exp(phase)
        # Apply a Kronecker product with the identity of left bond of l_pos
        l_dim = self.get_virtual_dimensions(l_pos)[0]
        msg_tensor = contract(
            "m,ab->mab",
            msg_tensor,
            cp.eye(l_dim, dtype=self._cfg._complex_t),
            options=options,
            optimize={"path": [(0, 1)]},
        )

        # Define the "connection" tensors
        connection = {  # bonds PMpm
            Pauli.I: cp.eye(4, dtype=self._cfg._complex_t),  # No connection, just IxI
            Pauli.Z: cp.asarray(  # A CX gate so that IZ in pm becomes ZZ in PM
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=self._cfg._complex_t,
            ),
            Pauli.X: cp.asarray(  # A H*CX*H so that IZ in pm becomes XZ in PM
                [
                    [0.5, 0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5, 0.5],
                ],
                dtype=self._cfg._complex_t,
            ),
            Pauli.Y: cp.asarray(  # S*H*CX*H*Sdg so that IZ in pm becomes YZ in PM
                [
                    [0.5, 0.5, -0.5j, 0.5j],
                    [0.5, 0.5, 0.5j, -0.5j],
                    [0.5j, -0.5j, 0.5, 0.5],
                    [-0.5j, 0.5j, 0.5, 0.5],
                ],
                dtype=self._cfg._complex_t,
            ),
        }
        for p, t in connection.items():
            connection[p] = cp.reshape(t, (2, 2, 2, 2))

        # Update all of the tensor sites from `l_pos` to `r_pos` - 1
        for pos in range(l_pos, r_pos):
            # Identify which is the Pauli in this position
            if pos not in pos_qubit_map:
                pauli = Pauli.I
            else:
                pauli = pauli_str.map[pos_qubit_map[pos]]

            # Push the message tensor through the site tensor, updating it
            self.tensors[pos], msg_tensor = contract_decompose(
                "mab,bcp,PMpm->asP,Msc",
                msg_tensor,
                self.tensors[pos],
                connection[pauli],
                algorithm={"qr_method": tensor.QRMethod()},
                options=options,
                optimize={"path": [(0, 1), (0, 1)]},
            )

            # The site tensor is now in canonical form
            self.canonical_form[pos] = DirMPS.LEFT  # type: ignore

        # Finally, contract the `msg_tensor` with the site tensor in `r_pos` and
        # cap off the dangling virtual bond from the "connection" tensor with a |0>
        pauli = pauli_str.map[pos_qubit_map[r_pos]]
        trivial_tensor = cp.asarray([1, 0], dtype=self._cfg._complex_t)

        self.tensors[r_pos] = contract(
            "mab,bcp,PMpm,M->acP",
            msg_tensor,
            self.tensors[r_pos],
            connection[pauli],
            trivial_tensor,
            options=options,
            optimize={"path": [(2, 3), (0, 1), (0, 1)]},
        )
        # The site tensor is not in canonical form anymore
        self.canonical_form[r_pos] = None

        # Apply truncations between the leftmost and rightmost qubits
        self._truncate_path(l_pos, r_pos)

        return self

    def _truncate_path(self, l_pos: int, r_pos: int) -> None:
        """Truncates all bonds between `l_pos` and `r_pos` according to the policy
        chosen by the user.
        """
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
        orig_fidelity = self.fidelity

        ############################################################
        ### Setup SVD configuration depending on user's settings ###
        ############################################################

        if self._cfg.truncation_fidelity < 1:
            # Apply SVD decomposition to truncate as much as possible before exceeding
            # a `discarded_weight_cutoff` of `1 - self._cfg.truncation_fidelity`.
            self._logger.debug(
                f"Truncating to target fidelity={self._cfg.truncation_fidelity}"  # noqa: G004
            )

            # When there are multiple virtual bonds between the two MPS tensors where
            # the gate is applied (i.e. non-adjacent qubits) we need to distribute the
            # allowed truncation error among the different bonds.
            # Our target is to assign a local truncation fidelity `f_i` to each bond
            # `i` in the input lists so that the lower bound of the fidelity satisfies:
            #
            #   real_fidelity > self.fidelity*prod(f_i) > self.fidelity*trunc_fidelity
            #
            # Let e_i = 1 - f_i, where we refer to `e_i` as the "truncation error at i".
            # We can use that when 0 < e_i < 1, it holds that:
            #
            #   prod(1 - e_i)   >   1 - sum(e_i)
            #
            # Hence, as long as we satisfy
            #
            #   1 - sum(e_i)    >    truncation_fidelity
            #
            # the target inquality at the top will be satisfied for our chosen f_i.
            # We achieve this by defining e_i = (1 - trunc_fid) / k, where k is the
            # number of bonds between the two tensors.
            distance = r_pos - l_pos
            if distance == 0:
                local_truncation_error = 1 - self._cfg.truncation_fidelity
            else:
                local_truncation_error = (1 - self._cfg.truncation_fidelity) / distance
            self._logger.debug(
                f"There are {distance} bonds between the qubits. Each of these will "  # noqa: G004
                f"be truncated to target fidelity={1 - local_truncation_error}"
            )

            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                discarded_weight_cutoff=local_truncation_error,
                partition="U",  # Contract S directly into U (to the "left")
                normalization="L2",  # Sum of squares singular values must equal 1
            )

        else:
            # Apply SVD decomposition and truncate up to a `max_extent` (for the shared
            # bond) of `self._cfg.chi`.
            # If the user did not explicitly ask for truncation, `self._cfg.chi` will be
            # set to a very large default number, so it's like no `max_extent` was set.
            # Still, we remove any singular values below ``self._cfg.zero``.
            self._logger.debug(f"Truncating to (or below) chosen chi={self._cfg.chi}")  # noqa: G004

            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                max_extent=self._cfg.chi,
                partition="U",  # Contract S directly into U (to the "left")
                normalization="L2",  # Sum of squares singular values must equal 1
            )

        ############################################################
        ### Apply truncation to all bonds between the two qubits ###
        ############################################################

        # From right to left, so that we can use the current canonical form.
        for pos in reversed(range(l_pos, r_pos)):
            self.tensors[pos], S, self.tensors[pos + 1], info = contract_decompose(
                "abl,bcr->abl,bcr",  # Note: doesn't follow the glossary above.
                self.tensors[pos],
                self.tensors[pos + 1],
                algorithm={"svd_method": svd_method, "qr_method": False},
                options=options,
                optimize={"path": [(0, 1)]},
                return_info=True,
            )
            assert S is None  # Due to "partition" option in SVDMethod

            # Since we are contracting S to the "left" in `svd_method`, the site tensor
            # at `pos+1` is canonicalised, whereas the site tensor at `pos` is the one
            # where S has been contracted to and, hence, is not in canonical form
            self.canonical_form[pos + 1] = DirMPS.RIGHT  # type: ignore
            self.canonical_form[pos] = None
            # Update fidelity lower bound
            this_fidelity = 1.0 - info.svd_info.discarded_weight
            self.fidelity *= this_fidelity
            # Report to logger
            self._logger.debug(
                f"Truncation done between positions {pos} and {pos + 1}. "  # noqa: G004
                f"Truncation fidelity={this_fidelity}"
            )
            self._logger.debug(
                "Reduced virtual bond dimension from "  # noqa: G004
                f"{info.svd_info.full_extent} to {info.svd_info.reduced_extent}."
            )

        if self._cfg.truncation_fidelity < 1:
            # Sanity check: user's requested lower bound of fidelity satisfied
            assert self.fidelity > orig_fidelity * self._cfg.truncation_fidelity

        # If requested, provide info about memory usage.
        if self._logger.isEnabledFor(logging.INFO):
            # If-statetement used so that we only call `get_byte_size` if needed.
            self._logger.info(f"MPS size (MiB)={self.get_byte_size() / 2**20}")  # noqa: G004
            self._logger.info(f"MPS fidelity={self.fidelity}")  # noqa: G004

    def measure_pauli_string(self, pauli_string: QubitPauliString) -> int:
        """Measure the Pauli string and return `0` or `1` accordingly.
        The MPS is collapsed and renormalised.
        """
        # Add an ancilla qubit
        idx = 0
        while Qubit("_aux_mps", idx) in self.get_qubits():
            idx += 1
        aux_q = Qubit("_aux_mps", idx)
        self.add_qubit(aux_q, position=0, state=0)

        # Fix truncation parameters to prevent approximation
        orig_chi = self._cfg.chi
        orig_tfid = self._cfg.truncation_fidelity
        self._cfg.chi = 2**60
        self._cfg.truncation_fidelity = 1.0

        # Push the qubit through the MPS and apply the necessary entangling gates
        sorted_qubits = sorted(self.get_qubits(), key=lambda q: self.qubit_position[q])
        for q in sorted_qubits[1:]:  # Skip aux_q, which is in the leftmost position
            self._apply_command(Op.create(OpType.SWAP), [aux_q, q], [], [])
            if q in pauli_string.map and pauli_string.map[q] != Pauli.I:
                match pauli_string.map[q]:
                    case Pauli.X:
                        self._apply_command(Op.create(OpType.CX), [q, aux_q], [], [])
                    case Pauli.Z:
                        self._apply_command(Op.create(OpType.H), [q], [], [])
                        self._apply_command(Op.create(OpType.CX), [q, aux_q], [], [])
                        self._apply_command(Op.create(OpType.H), [q], [], [])
                    case Pauli.Y:
                        self._apply_command(Op.create(OpType.Sdg), [q], [], [])
                        self._apply_command(Op.create(OpType.CX), [q, aux_q], [], [])
                        self._apply_command(Op.create(OpType.S), [q], [], [])

        # Measure the ancilla qubit destructively
        result = self.measure({aux_q}, destructive=True)

        # Restore truncation parameters
        self._cfg.chi = orig_chi
        self._cfg.truncation_fidelity = orig_tfid

        return result[aux_q]

    def get_entanglement_entropy(self, position: int) -> float:
        """Returns the entanglement entropy of the virtual bond to the right of ``position``.

        Args:
            position: A position in the MPS.

        Returns:
            The entanglement entropy.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """
        if position < 0 or position >= len(self) - 1:
            raise RuntimeError(f"Position {position} is out of bounds.")

        # Canonicalise to tensor[position]
        self.canonicalise(position, position + 1)

        # Contract tensor[position] with tensor[position+1]
        # Apply SVD to obtain the singular values at the virtual bond
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
        svd_method = tensor.SVDMethod(
            abs_cutoff=self._cfg.zero,  # Remove zero singular values
        )
        _, S, _ = contract_decompose(
            "abl,bcr->abl,bcr",  # Note: doesn't follow the glossary above.
            self.tensors[position],
            self.tensors[position + 1],
            algorithm={"svd_method": svd_method, "qr_method": False},
            options=options,
            optimize={"path": [(0, 1)]},
        )

        # Compute the entanglement entropy
        entropy = -sum(s**2 * cp.log(s**2) for s in S)
        return float(entropy)
