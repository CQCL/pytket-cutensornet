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

import cupy as cp
import numpy as np

import cuquantum as cq
import cuquantum.cutensornet as cutn
from pytket.circuit import Op

from .mps import Tensor, MPS


class MPSxGate(MPS):
    """Class for state-based simulation using an Matrix Product State (MPS)
    representation, with gate-by-gate contraction.

        Attributes:
        chi (int): The maximum allowed dimension of a virtual bond.
        tensors (list[Tensor]): A list of tensors in the MPS; tensors[0] is
            the leftmost and tensors[len(self)-1] is the rightmost; tensors[i]
            and tensors[i+1] are connected in the MPS via a bond.
        fidelity (float): An estimate of the fidelity, obtained by multiplying
            the fidelities after each contraction.
    """

    def apply_1q_gate(self, position: int, gate: Op):
        """Apply the 1-qubit gate to the MPS. This does not increase the
        dimension of any bond.

                Args:
                position: The position of the MPS tensor that this gate
                is applied to.
                gate: The gate to be applied.
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
        contract_bond = self.get_physical_bond(position)
        new_bond = self._new_bond_id()

        # Contract
        new_tensor = cq.contract(
            gate_tensor,
            [new_bond, contract_bond],
            self.tensors[position].data,
            virtual_bonds + [contract_bond],
            virtual_bonds + [new_bond],
        )

        # Update ``self.tensors``
        self.tensors[position].data = new_tensor
        self.tensors[position].bonds = virtual_bonds + [new_bond]

    def apply_2q_gate(self, positions: tuple[int, int], gate: Op):
        """Apply the 2-qubit gate to the MPS. If doing so increases the
        virtual bond dimension beyond ``chi``; truncation is automatically
        applied. The MPS is converted to canonical form before truncating.

                Args:
                positions: The position of the MPS tensors that this gate
                is applied to. They must be contiguous.
                gate: The gate to be applied.
        """
        if any(self.get_physical_dimension(pos) != 2 for pos in positions):
            raise RuntimeError(
                "Gates can only be applied to tensors with physical"
                + " bond dimension of 2."
            )

        dist = positions[1] - positions[0]
        # We explicitly allow both dist==1 or dist==-1 so that non-symmetric
        # gates such as CX can use the same Op for the two ways it can be in.
        if dist not in [1, -1]:
            raise Exception("Gates must be applied to contiguous positions!")
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
        if new_dim > self.chi:  # Truncation will be required
            new_dim = self.chi
            # If truncation required, convert to canonical form before
            # contracting. Avoids the need to apply gauge transformations
            # to the larger tensor resulting from the contraction.
            self.canonicalise(l_pos, r_pos)

        # Load the gate's unitary to the GPU memory
        gate_tensor = cp.empty(shape=(4, 4), dtype=self._complex_t)
        gate_tensor.set(gate.get_unitary(), self._stream)

        # Reshape into a rank-4 tensor and assign bond IDs
        gate_tensor = cp.reshape(gate_tensor, (2, 2, 2, 2))
        left_input = self.get_physical_bond(l_pos)
        right_input = self.get_physical_bond(r_pos)
        left_output = self._new_bond_id()
        right_output = self._new_bond_id()

        if l_pos == positions[0]:
            gate_bonds = [left_output, right_output, left_input, right_input]
        else:  # Implicit swap
            gate_bonds = [right_output, left_output, right_input, left_input]
        T_bonds = [
            v_bond
            for v_bond in self.get_virtual_bonds(l_pos) + self.get_virtual_bonds(r_pos)
            if v_bond != r_pos  # The bond between the left and right tensors
        ] + [left_output, right_output]

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
        T = Tensor(T_d, T_bonds)

        # Get the template of the MPS tensors involved
        L = self.tensors[l_pos]
        l_shape = list(L.data.shape)
        R = self.tensors[r_pos]
        r_shape = list(R.data.shape)

        if new_dim != r_shape[0]:
            # We need to change the shape of the tensors
            l_shape[-2] = new_dim
            L.data = cp.empty(l_shape, dtype=self._complex_t)
            r_shape[0] = new_dim
            R.data = cp.empty(r_shape, dtype=self._complex_t)

        # Reassign bond IDs
        L.bonds[-1] = left_output
        R.bonds[-1] = right_output

        # Reserve space for the tensor of singular values
        S_d = cp.empty(new_dim, dtype=self._real_t)

        # Create tensor descriptors
        T_desc = T.get_tensor_descriptor(self._libhandle)
        L_desc = L.get_tensor_descriptor(self._libhandle)
        R_desc = R.get_tensor_descriptor(self._libhandle)

        # Configure SVD parameters
        svd_config = cutn.create_tensor_svd_config(self._libhandle)

        config_dtype = cutn.tensor_svd_config_get_attribute_dtype(
            cutn.TensorSVDConfigAttribute.S_PARTITION
        )
        config_value = np.array([cutn.TensorSVDPartition.UV_EQUAL], dtype=config_dtype)
        cutn.tensor_svd_config_set_attribute(
            self._libhandle,
            svd_config,
            cutn.TensorSVDConfigAttribute.S_PARTITION,
            config_value.ctypes.data,
            config_value.dtype.itemsize,
        )
        # Create SVDInfo to record truncation information
        svd_info = cutn.create_tensor_svd_info(self._libhandle)

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
            0,
            self._stream.ptr,  # 0 means let cuQuantum manage mem itself
        )
        self._stream.synchronize()

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
        self.fidelity *= 1 - float(discarded_weight)

        # Destroy descriptors
        cutn.destroy_tensor_descriptor(T_desc)
        cutn.destroy_tensor_descriptor(L_desc)
        cutn.destroy_tensor_descriptor(R_desc)
        cutn.destroy_tensor_svd_config(svd_config)
        cutn.destroy_tensor_svd_info(svd_info)

        # The L and R tensors have already been updated and these correspond
        # to the entries of l_pos and r_pos in self.tensors
        assert self.tensors[l_pos] is L and self.tensors[r_pos] is R

    def apply_postselection(self, position: int):
        """Apply a postselection of state 0 to the chosen tensor.
        Postselection of any other state can be applied by preceding this
        with the appropriate gates.
        The resulting physical bond of the tensor at ``position`` has
        dimension 1.

                Args:
                position: The position of the MPS tensor that this gate
                is applied to.
        """
        if self.get_physical_dimension(position) != 2:
            raise RuntimeError(
                "Postselection can only be applied to tensors with physical"
                + " bond dimension of 2."
            )

        # Create the tensor of the |0> postselection
        post_tensor = cp.empty(shape=(1, 2), dtype=self._complex_t)
        post_tensor[0][0] = 1
        post_tensor[0][1] = 0

        # Identify the ID of the bonds involved
        virtual_bonds = self.get_virtual_bonds(position)
        contract_bond = self.get_physical_bond(position)
        new_bond = self._new_bond_id()

        # Contract
        new_tensor = cq.contract(
            post_tensor,
            [new_bond, contract_bond],
            self.tensors[position].data,
            self.tensors[position].bonds,
            virtual_bonds + [new_bond],
        )

        # Update ``self.tensors``
        self.tensors[position].data = new_tensor
        self.tensors[position].bonds = virtual_bonds + [new_bond]
