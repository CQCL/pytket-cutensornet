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
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)

from pytket.circuit import Qubit
from .mps import MPS


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
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}

        positions = [self.qubit_position[q0], self.qubit_position[q1]]
        l_pos = min(positions)
        r_pos = max(positions)

        # Always canonicalise. Even in the case of exact simulation (no truncation)
        # canonicalisation may reduce the bond dimension (thanks to reduced QR).
        self.canonicalise(l_pos, r_pos)

        # Figure out the new dimension of the shared virtual bond
        new_dim = 2 * min(
            self.get_virtual_dimensions(l_pos)[0],
            self.get_virtual_dimensions(r_pos)[1],
        )

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

        # Get the template of the MPS tensors involved
        L = self.tensors[l_pos]
        R = self.tensors[r_pos]

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

        elif new_dim > self._cfg.chi:
            # Apply SVD decomposition and truncate up to a `max_extent` (for the shared
            # bond) of `self._cfg.chi`.
            self._logger.debug(f"Truncating to (or below) chosen chi={self._cfg.chi}")

            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                max_extent=self._cfg.chi,
                partition="U",  # Contract S directly into U (named L in our case)
                normalization="L2",  # Sum of squares singular values must equal 1
            )

        else:
            # The user did not explicitly ask for truncation, but it is advantageous to
            # remove any singular values below ``self._cfg.zero``.
            self._logger.debug(f"Truncating singular values below={self._cfg.zero}.")
            if self._cfg.zero > self._cfg._atol / 1000:
                self._logger.info(  # This was raised as a warning in Config already
                    "Your chosen value_of_zero is relatively large. "
                    "Faithfulness of final fidelity estimate is not guaranteed."
                )

            # NOTE: There is no guarantee of canonical form in this case. This is fine
            # since canonicalisation is just meant to detect the optimal singular values
            # to truncate, but if we find values that are essentially zero, we are safe
            # to remove them.
            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                partition="U",  # Contract S directly into U (named L in our case)
                normalization=None,  # Without canonicalisation we must not normalise
            )

        # Apply the SVD decomposition using the configuration defined above
        L, S, R, svd_info = tensor.decompose(
            "acLR->asL,scR", T, method=svd_method, options=options, return_info=True
        )
        assert S is None  # Due to "partition" option in SVDMethod

        # Update fidelity if there was some truncation (of non-zero singular values)
        if new_dim > self._cfg.chi or self._cfg.truncation_fidelity < 1:
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
            self._logger.debug(f"Truncation done. Truncation fidelity={this_fidelity}")

        else:
            self._logger.debug(f"Truncation done. Fidelity estimate unchanged.")

        self._logger.debug(
            f"Reduced virtual bond dimension from {new_dim} to {R.shape[0]}."
        )

        self.tensors[l_pos] = L
        self.tensors[r_pos] = R

        # If requested, provide info about memory usage.
        if self._logger.isEnabledFor(logging.INFO):
            # If-statetement used so that we only call `get_byte_size` if needed.
            self._logger.info(f"MPS size (MiB)={self.get_byte_size() / 2**20}")
            self._logger.info(f"MPS fidelity={self.fidelity}")

        return self
