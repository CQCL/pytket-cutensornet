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
import warnings

import numpy as np  # type: ignore
try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
try:
    import cuquantum as cq  # type: ignore
    import cuquantum.cutensornet as cutn  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)

from pytket.circuit import Op  # type: ignore
from .mps import Tensor, MPS


class MPSxGate(MPS):
    """Class for state-based simulation using an Matrix Product State (MPS)
    representation, with gate-by-gate contraction.

        Attributes:
        chi (int): The maximum allowed dimension of a virtual bond.
        truncation_fidelity (float): The target fidelity of SVD truncation.
        tensors (list[Tensor]): A list of tensors in the MPS; tensors[0] is
            the leftmost and tensors[len(self)-1] is the rightmost; tensors[i]
            and tensors[i+1] are connected in the MPS via a bond.
        fidelity (float):  A lower bound of the fidelity, obtained by multiplying
            the fidelities after each contraction. The fidelity of a contraction
            corresponds to |<psi|phi>|^2 where |psi> and |phi> are the states
            before and after truncation (assuming both are normalised).
    """

    def _apply_1q_gate(self, position: int, gate: Op) -> MPSxGate:
        """Apply the 1-qubit gate to the MPS. This does not increase the
        dimension of any bond.

        Args:
            position: The position of the MPS tensor that this gate
                is applied to.
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.
        """
        if self.get_physical_dimension(position) != 2:
            raise RuntimeError(
                "Gates can only be applied to tensors with physical"
                + " bond dimension of 2."
            )

        # Load the gate's unitary to the GPU memory
        gate_tensor = cp.empty(shape=(2, 2), dtype=self._complex_t)
        gate_tensor.set(gate.get_unitary(), self._stream)

        # Identify the ID of the bonds involved
        virtual_bonds = self.get_virtual_bonds(position)
        physical_bond = self.get_physical_bond(position)
        new_bond = -1  # Temporary ID for new physical bond

        # Contract
        new_tensor = cq.contract(
            gate_tensor,
            [new_bond, physical_bond],
            self.tensors[position].data,
            self.tensors[position].bonds,
            virtual_bonds + [new_bond],
        )

        # Update ``self.tensors``
        self.tensors[position].data = new_tensor
        return self

    def _apply_2q_gate(self, positions: list[int], gate: Op) -> MPSxGate:
        """Apply the 2-qubit gate to the MPS. If doing so increases the
        virtual bond dimension beyond ``chi``; truncation is automatically
        applied. The MPS is converted to canonical form before truncating.

        Args:
            positions: The position of the MPS tensors that this gate
                is applied to. They must be contiguous.
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.
        """
        if any(self.get_physical_dimension(pos) != 2 for pos in positions):
            raise RuntimeError(
                "Gates can only be applied to tensors with physical"
                + " bond dimension of 2."
            )
        l_pos = min(positions)
        r_pos = max(positions)

        # Figure out the new dimension of the shared virtual bond
        if l_pos == 0 or r_pos == len(self) - 1:
            new_dim = 2
        else:
            new_dim = 2 * min(
                self.get_virtual_dimensions(l_pos)[0],
                self.get_virtual_dimensions(r_pos)[1],
            )

        # Truncation will be required if `new_dim` is larger than `chi`
        # or if set by `truncation_fidelity`
        truncating = new_dim > self.chi or self.truncation_fidelity < 1
        if truncating:
            # If truncation required, convert to canonical form before
            # contracting. Avoids the need to apply gauge transformations
            # to the larger tensor resulting from the contraction.
            self.canonicalise(l_pos, r_pos)

        # Load the gate's unitary to the GPU memory
        gate_tensor = cp.empty(shape=(4, 4), dtype=self._complex_t)
        gate_tensor.set(gate.get_unitary(), self._stream)

        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(gate_tensor, (2, 2, 2, 2))

        # Assign bond IDs
        left_p_bond = self.get_physical_bond(l_pos)
        right_p_bond = self.get_physical_bond(r_pos)
        left_new_bond = -2  # Temporary ID for left physical bond
        right_new_bond = -1  # Temporary ID for right physical bond

        if l_pos == positions[0]:
            gate_bonds = [left_new_bond, right_new_bond, left_p_bond, right_p_bond]
        else:  # Implicit swap
            gate_bonds = [right_new_bond, left_new_bond, right_p_bond, left_p_bond]

        T_bonds = [
            v_bond
            for v_bond in self.get_virtual_bonds(l_pos) + self.get_virtual_bonds(r_pos)
            if v_bond != r_pos  # The bond between the left and right tensors
        ] + [left_new_bond, right_new_bond]

        # Contract
        T_d = cq.contract(
            gate_tensor,
            gate_bonds,
            self.tensors[l_pos].data,
            self.tensors[l_pos].bonds,
            self.tensors[r_pos].data,
            self.tensors[r_pos].bonds,
            T_bonds,
        )
        # Reassign auxiliar bond ID
        T_bonds[-2] = left_p_bond
        T_bonds[-1] = right_p_bond
        # Create tensor object
        T = Tensor(T_d, T_bonds)

        # Get the template of the MPS tensors involved
        L = self.tensors[l_pos]
        l_shape = list(L.data.shape)
        R = self.tensors[r_pos]
        r_shape = list(R.data.shape)

        if self.chi < new_dim:
            new_dim = self.chi
        if new_dim != r_shape[0]:
            # We need to change the shape of the tensors
            l_shape[-2] = new_dim
            L.data = cp.empty(l_shape, dtype=self._complex_t)
            r_shape[0] = new_dim
            R.data = cp.empty(r_shape, dtype=self._complex_t)

        # Reserve space for the tensor of singular values
        S_d = cp.empty(new_dim, dtype=self._real_t)

        # Create tensor descriptors
        T_desc = T.get_tensor_descriptor(self._libhandle)
        L_desc = L.get_tensor_descriptor(self._libhandle)
        R_desc = R.get_tensor_descriptor(self._libhandle)

        # Create SVDConfig with default configuration
        svd_config = cutn.create_tensor_svd_config(self._libhandle)
        # Create SVDInfo to record truncation information
        svd_info = cutn.create_tensor_svd_info(self._libhandle)

        if self.truncation_fidelity < 1:
            # Carry out SVD decomposition first with NO truncation
            # to figure out where to apply the dimension cutoff.
            #
            # NOTE: Unfortunately, it looks like there is no option
            # in the cuTensorNet API to ask for a cutoff in terms of
            # `discarded_weight`; ABS_CUTOFF and REL_CUTOFF aren't it.
            # It'd be best to request them to add this feature.
            #
            # TODO: Benchmark if this workflow that does two SVDs on
            # the same tensor is slower than if I were to do the
            # tensor reshaping, normalisation of S and contraction myself.
            # It's likely, since SVD is costly, but this is simpler.

            cutn.tensor_svd(
                self._libhandle,
                T_desc,
                T.data.data.ptr,
                L_desc,
                L.data.data.ptr,
                S_d.data.ptr,
                R_desc,
                R.data.data.ptr,
                svd_config,
                svd_info,
                0,  # 0 means let cuQuantum manage mem itself
                self._stream.ptr,  # type: ignore
            )
            self._stream.synchronize()  # type: ignore

            # Use the fact that the entries of S_d are sorted in decreasing
            # order and calculate the number of singular values `new_dim` to
            # keep so that
            #                             sum([s**2 for s in S'])
            #   truncation_fidelity  <=  -------------------------
            #                             sum([s**2 for s in S])
            #
            # where S is the list of original singular values and S' is the set of
            # singular values that remain after truncation (before normalisation).
            denom = float(sum(cp.square(S_d)))  # Element-wise squaring
            numer = 0.0
            new_dim = 0

            while self.truncation_fidelity > numer / denom:
                numer += float(S_d[new_dim] ** 2)
                new_dim += 1

            # We need to change the shape of the tensors
            l_shape[-2] = new_dim
            L.data = cp.empty(l_shape, dtype=self._complex_t)
            r_shape[0] = new_dim
            R.data = cp.empty(r_shape, dtype=self._complex_t)
            S_d = cp.empty(new_dim, dtype=self._real_t)

            # Create new tensor descriptors
            cutn.destroy_tensor_descriptor(L_desc)
            cutn.destroy_tensor_descriptor(R_desc)
            L_desc = L.get_tensor_descriptor(self._libhandle)
            R_desc = R.get_tensor_descriptor(self._libhandle)

        # Configure SVD parameters
        svd_config_attributes = [
            # TensorSVDPartition.US asks that cuTensorNet automatically
            # contracts the tensor of singular values (S) into one of the
            # two tensors (U), named L in our case.
            (
                cutn.TensorSVDConfigAttribute.S_PARTITION,
                cutn.TensorSVDPartition.US,
            ),
        ]

        if truncating:
            # Renormalise after truncation. Thanks to using canonical form of
            # the MPS and the fact that the original state is a unit vector,
            # we know that the L2 norm of the singular values (i.e. sum of its
            # squares) must be equal to 1. We ask cuTensorNet to renormalise
            # the truncated S so that this is satisfied after truncation, thus
            # making sure the resulting state is normalised.
            svd_config_attributes.append(
                (
                    cutn.TensorSVDConfigAttribute.S_NORMALIZATION,
                    cutn.TensorSVDNormalization.L2,
                )
            )

        for attr, value in svd_config_attributes:
            attr_dtype = cutn.tensor_svd_config_get_attribute_dtype(attr)
            value = np.array([value], dtype=attr_dtype)
            cutn.tensor_svd_config_set_attribute(
                self._libhandle,
                svd_config,
                attr,
                value.ctypes.data,
                value.dtype.itemsize,
            )

        # Apply SVD decomposition; truncation will be applied if needed
        cutn.tensor_svd(
            self._libhandle,
            T_desc,
            T.data.data.ptr,
            L_desc,
            L.data.data.ptr,
            S_d.data.ptr,
            R_desc,
            R.data.data.ptr,
            svd_config,
            svd_info,
            0,  # 0 means let cuQuantum manage mem itself
            self._stream.ptr,  # type: ignore
        )
        self._stream.synchronize()  # type: ignore

        # Get an error estimate
        discarded_weight_dtype = cutn.tensor_svd_info_get_attribute_dtype(
            cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT
        )
        discarded_weight = np.empty(1, dtype=discarded_weight_dtype)
        cutn.tensor_svd_info_get_attribute(
            self._libhandle,
            svd_info,
            cutn.TensorSVDInfoAttribute.DISCARDED_WEIGHT,
            discarded_weight.ctypes.data,
            discarded_weight.itemsize,
        )
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
        if self.truncation_fidelity < 1:
            assert self.truncation_fidelity <= 1.0 - float(discarded_weight)
        self.fidelity *= 1.0 - float(discarded_weight)

        # Destroy handles
        cutn.destroy_tensor_descriptor(T_desc)
        cutn.destroy_tensor_descriptor(L_desc)
        cutn.destroy_tensor_descriptor(R_desc)
        cutn.destroy_tensor_svd_config(svd_config)
        cutn.destroy_tensor_svd_info(svd_info)

        # The L and R tensors have already been updated and these correspond
        # to the entries of l_pos and r_pos in self.tensors
        assert self.tensors[l_pos] is L and self.tensors[r_pos] is R
        return self
