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
import logging

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
        new_tensor = cq.contract(
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

        The MPS is converted to canonical and truncation is applied if necessary.

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

        if l_pos == positions[0]:
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
        T = cq.contract(
            f"SLl,abl,SRr,bcr->acLR",
            U,
            self.tensors[l_pos],
            V,
            self.tensors[r_pos],
            options=options,
            optimize={"path": [(0, 1), (0, 1), (0, 1)]},
        )
        self._logger.debug(f"Intermediate tensor of size (MiB)={T.nbytes / 2**20}")

        if self._cfg.truncation_fidelity < 1:
            # Apply SVD decomposition to truncate as much as possible before exceeding
            # a `discarded_weight_cutoff` of `1 - self._cfg.truncation_fidelity`.
            self._logger.debug(
                f"Truncating to target fidelity={self._cfg.truncation_fidelity}"
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
            self._logger.debug(f"Truncating to (or below) chosen chi={self._cfg.chi}")

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
        self._logger.debug(f"Truncation done. Truncation fidelity={this_fidelity}")
        self._logger.debug(
            "Reduced virtual bond dimension from "
            f"{svd_info.full_extent} to {svd_info.reduced_extent}."
        )

        self.tensors[l_pos] = L
        self.tensors[r_pos] = R

        # If requested, provide info about memory usage.
        if self._logger.isEnabledFor(logging.INFO):
            # If-statetement used so that we only call `get_byte_size` if needed.
            self._logger.info(f"MPS size (MiB)={self.get_byte_size() / 2**20}")
            self._logger.info(f"MPS fidelity={self.fidelity}")

        return self

    def _apply_2q_unitary_nonadjacent(
        self, unitary: cp.ndarray, q0: Qubit, q1: Qubit
    ) -> MPSxGate:
        """Applies the 2-qubit unitary to the MPS between non-adjacent qubits.

        The MPS is converted to canonical and truncation is applied if necessary.

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
        self.canonicalise(l_pos, l_pos)

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

        if l_pos == positions[0]:
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

        #################################
        ### Apply the gate to the MPS ###
        #################################

        orig_fidelity = self.fidelity

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
                f"Pushing msg_tensor ({msg_tensor.nbytes // 2**20} MiB) through site "
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
        self.tensors[r_pos] = cq.contract(
            "sam,mbr,sRr->abR",
            msg_tensor,
            self.tensors[r_pos],
            r_gate_tensor,
            options=options,
            optimize={"path": [(0, 2), (0, 1)]},
        )

        # The site tensor is not in canonical form anymore
        self.canonical_form[r_pos] = None

        ############################################################
        ### Setup SVD configuration depending on user's settings ###
        ############################################################

        if self._cfg.truncation_fidelity < 1:
            # Apply SVD decomposition to truncate as much as possible before exceeding
            # a `discarded_weight_cutoff` of `1 - self._cfg.truncation_fidelity`.
            self._logger.debug(
                f"Truncating to target fidelity={self._cfg.truncation_fidelity}"
            )

            # When there are multiple virtual bonds between the two MPS tensors where
            # the gate is applied (i.e. non-adjacent qubits) we need to distributed the
            # allowed truncation error between the different bonds.
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
            local_truncation_error = (1 - self._cfg.truncation_fidelity) / distance
            self._logger.debug(
                f"The are {distance} bond between the qubits. Each of these will "
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
            self._logger.debug(f"Truncating to (or below) chosen chi={self._cfg.chi}")

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
                f"Truncation done between positions {pos} and {pos+1}. "
                f"Truncation fidelity={this_fidelity}"
            )
            self._logger.debug(
                "Reduced virtual bond dimension from "
                f"{info.svd_info.full_extent} to {info.svd_info.reduced_extent}."
            )

        if self._cfg.truncation_fidelity < 1:
            # Sanity check: user's requested lower bound of fidelity satisfied
            assert self.fidelity > orig_fidelity * self._cfg.truncation_fidelity

        # If requested, provide info about memory usage.
        if self._logger.isEnabledFor(logging.INFO):
            # If-statetement used so that we only call `get_byte_size` if needed.
            self._logger.info(f"MPS size (MiB)={self.get_byte_size() / 2**20}")
            self._logger.info(f"MPS fidelity={self.fidelity}")

        return self
