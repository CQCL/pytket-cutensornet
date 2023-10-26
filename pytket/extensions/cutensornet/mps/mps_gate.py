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

from pytket.circuit import Op
from .mps import MPS


class MPSxGate(MPS):
    """Implements a gate-by-gate contraction algorithm to calculate the output state
    of a circuit as an ``MPS``. The algorithm is described in:
    https://arxiv.org/abs/2002.07730
    """

    def _apply_1q_gate(self, position: int, gate: Op) -> MPSxGate:
        """Applies the 1-qubit gate to the MPS.

        This does not increase the dimension of any bond.

        Args:
            position: The position of the MPS tensor that this gate
                is applied to.
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.
        """

        # Load the gate's unitary to the GPU memory
        gate_unitary = gate.get_unitary().astype(dtype=self._cfg._complex_t, copy=False)
        gate_tensor = cp.asarray(gate_unitary, dtype=self._cfg._complex_t)

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
            gate_tensor,
            self.tensors[position],
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"path": [(0, 1)]},
        )

        # Update ``self.tensors``
        self.tensors[position] = new_tensor
        return self

    def _apply_2q_gate(self, positions: tuple[int, int], gate: Op) -> MPSxGate:
        """Applies the 2-qubit gate to the MPS.

        If doing so increases the virtual bond dimension beyond ``chi``;
        truncation is automatically applied.
        The MPS is converted to canonical form before truncating.

        Args:
            positions: The position of the MPS tensors that this gate
                is applied to. They must be contiguous.
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.
        """
        l_pos = min(positions)
        r_pos = max(positions)

        # Figure out the new dimension of the shared virtual bond
        new_dim = 2 * min(
            self.get_virtual_dimensions(l_pos)[0],
            self.get_virtual_dimensions(r_pos)[1],
        )

        # Canonicalisation may be required if `new_dim` is larger than `chi`
        # or if set by `truncation_fidelity`
        if new_dim > self._cfg.chi or self._cfg.truncation_fidelity < 1:
            # If truncation required, convert to canonical form before
            # contracting. Avoids the need to apply gauge transformations
            # to the larger tensor resulting from the contraction.
            self.canonicalise(l_pos, r_pos)

            # Since canonicalisation may change the dimension of the bonds,
            # we need to recalculate the value of `new_dim`
            new_dim = 2 * min(
                self.get_virtual_dimensions(l_pos)[0],
                self.get_virtual_dimensions(r_pos)[1],
            )

        # Load the gate's unitary to the GPU memory
        gate_unitary = gate.get_unitary().astype(dtype=self._cfg._complex_t, copy=False)
        gate_tensor = cp.asarray(gate_unitary, dtype=self._cfg._complex_t)

        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(gate_tensor, (2, 2, 2, 2))

        # Glossary of bond IDs
        # l -> physical bond of the left tensor in the MPS
        # r -> physical bond of the right tensor in the MPS
        # L -> left bond of the outcome of the gate
        # R -> right bond of the outcome of the gate
        # a,b,c -> the virtual bonds of the tensors

        if l_pos == positions[0]:
            gate_bonds = "LRlr"
        else:  # Implicit swap
            gate_bonds = "RLrl"

        left_bonds = "abl"
        right_bonds = "bcr"
        result_bonds = "acLR"

        # Contract
        self._logger.debug("Contracting the two-qubit gate with its site tensors...")
        T = cq.contract(
            gate_bonds + "," + left_bonds + "," + right_bonds + "->" + result_bonds,
            gate_tensor,
            self.tensors[l_pos],
            self.tensors[r_pos],
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"path": [(0, 1), (0, 1)]},
        )
        self._logger.debug(f"Intermediate tensor of size (MiB)={T.nbytes / 2**20}")

        # Get the template of the MPS tensors involved
        L = self.tensors[l_pos]
        l_shape = list(L.shape)
        R = self.tensors[r_pos]
        r_shape = list(R.shape)

        if self._cfg.truncation_fidelity < 1:
            # Carry out SVD decomposition first with NO truncation
            # to figure out where to apply the dimension cutoff.
            # Then, apply S normalisation and contraction of S and L manually.
            #
            # TODO: As soon as cuQuantum 23.09 is released, replace this
            # unintuitive code with a simple update to SVDConfig so that it
            # uses REL_SUM2_CUTOFF. Then the code in the `else` block should
            # be run; i.e. use standard cuTensorNet API to do the SVD
            # including normalisation and contraction of S with L.
            self._logger.debug(
                f"Truncating to target fidelity={self._cfg.truncation_fidelity}"
            )

            options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
            svd_method = tensor.SVDMethod(abs_cutoff=self._cfg.zero)
            L, S, R = tensor.decompose(
                "acLR->asL,scR", T, method=svd_method, options=options
            )

            # Use the fact that the entries of S are sorted in decreasing
            # order and calculate the number of singular values `new_dim` to
            # keep so that
            #                             sum([s**2 for s in S'])
            #   truncation_fidelity  <=  -------------------------
            #                             sum([s**2 for s in S])
            #
            # where S is the list of original singular values and S' is the set of
            # singular values that remain after truncation (before normalisation).
            denom = float(sum(cp.square(S)))  # Element-wise squaring
            numer = 0.0
            old_dim = new_dim
            new_dim = 0

            # Take singular values until we surpass the target fidelity
            while self._cfg.truncation_fidelity > numer / denom:
                numer += float(S[new_dim] ** 2)
                new_dim += 1
            this_fidelity = numer / denom

            # Reshape tensors down to `new_dim` for the virtual bond
            # No data is copied or moved around, we're changing the ndarray bounds
            l_shape[-2] = new_dim
            # pylint: disable = unexpected-keyword-arg   # Disable pylint for next line
            L = cp.ndarray(
                l_shape,
                dtype=self._cfg._complex_t,
                memptr=L.data,
                strides=L.strides,
            )
            r_shape[0] = new_dim
            # pylint: disable = unexpected-keyword-arg   # Disable pylint for next line
            R = cp.ndarray(
                r_shape,
                dtype=self._cfg._complex_t,
                memptr=R.data,
                strides=R.strides,
            )
            # pylint: disable = unexpected-keyword-arg   # Disable pylint for next line
            S = cp.ndarray(new_dim, dtype=self._cfg._real_t, memptr=S.data)

            # Normalise
            S *= np.sqrt(1 / this_fidelity)

            # Contract S into L
            S = S.astype(dtype=self._cfg._complex_t, copy=False)
            # Use some einsum index magic: since the virtual bond "s" appears in the
            # list of bonds of the output, it is not summed over.
            # This causes S to act as the intended diagonal matrix.
            L = cq.contract(
                "asL,s->asL",
                L,
                S,
                options={"handle": self._lib.handle, "device_id": self._lib.device_id},
                optimize={"path": [(0, 1)]},
            )

            # We multiply the fidelity of the current step to the overall fidelity
            # to keep track of a lower bound for the fidelity.
            self.fidelity *= this_fidelity

            # Report to logger
            self._logger.debug(f"Truncation done. Truncation fidelity={this_fidelity}")
            self._logger.debug(
                f"Reduced virtual bond dimension from {old_dim} to {new_dim}."
            )

        elif new_dim > self._cfg.chi:
            # Apply SVD decomposition and truncate up to a `max_extent` (for the shared
            # bond) of `self._cfg.chi`. Ask cuTensorNet to contract S directly into the
            # L tensor and normalise the singular values so that the sum of its squares
            # is equal to one (i.e. the MPS is a normalised state after truncation).
            self._logger.debug(f"Truncating to (or below) chosen chi={self._cfg.chi}")

            options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                max_extent=self._cfg.chi,
                partition="U",  # Contract S directly into U (named L in our case)
                normalization="L2",  # Sum of squares equal 1
            )

            L, S, R, svd_info = tensor.decompose(
                "acLR->asL,scR", T, method=svd_method, options=options, return_info=True
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

            # Report to logger
            self._logger.debug(f"Truncation done. Truncation fidelity={this_fidelity}")
            self._logger.debug(
                f"Reduced virtual bond dimension from {new_dim} to {R.shape[0]}."
            )

        else:
            # The user did not explicitly ask for truncation, but it is advantageous to
            # remove any singular values below ``self._cfg.zero``.
            self._logger.debug(f"Truncating singular values below={self._cfg.zero}.")
            if self._cfg.zero > self._cfg._atol / 1000:
                self._logger.info(  # This was raised as a warning in ConfigMPS already
                    "Your chosen value_of_zero is relatively large. "
                    "Faithfulness of final fidelity estimate is not guaranteed."
                )

            # NOTE: There is no guarantee of canonical form in this case. This is fine
            # since canonicalisation is just meant to detect the optimal singular values
            # to truncate, but if we find values that are essentially zero, we are safe
            # to remove them.
            options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
            svd_method = tensor.SVDMethod(
                abs_cutoff=self._cfg.zero,
                partition="U",  # Contract S directly into U (named L in our case)
                normalization=None,  # Without canonicalisation we must not normalise
            )
            L, S, R = tensor.decompose(
                "acLR->asL,scR", T, method=svd_method, options=options
            )
            assert S is None  # Due to "partition" option in SVDMethod

            # Report to logger
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
