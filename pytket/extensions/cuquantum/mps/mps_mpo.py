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

import cupy as cp  # type: ignore
import numpy as np  # type: ignore
import cuquantum as cq  # type: ignore
import cuquantum.cutensornet as cutn  # type: ignore

from pytket.circuit import Op  # type: ignore
from .mps import Tensor, MPS


class MPSxMPO(MPS):
    """Class for state-based simulation using an Matrix Product State (MPS)
    representation with MPS-MPO contraction, using a DMRG-like approach
    following the approach described in: https://arxiv.org/abs/2207.05612.

        Attributes:
        chi (int): The maximum allowed dimension of a virtual bond.
        tensors (list[Tensor]): A list of tensors in the MPS; tensors[0] is
            the leftmost and tensors[len(self)-1] is the rightmost; tensors[i]
            and tensors[i+1] are connected in the MPS via a bond.
        fidelity (float):  A lower bound of the fidelity, obtained by multiplying
            the fidelities after each contraction. The fidelity of a contraction
            corresponds to |<psi|phi>|^2 where |psi> and |phi> are the states
            before and after truncation (assuming both are normalised).
    """

    def __init__(self, n_tensors: int, chi: int, float_precision: str = "float64"):
        # Use the same docstring as in MPS.
        super().__init__(n_tensors, chi, float_precision)

        # Initialise the MPO data structure. This will keep a list of the gates
        # batched for application to the MPS; all of them will be applied at
        # once when deemed appropriate or when calling .flush(), removing them
        # from here. The gates are kept in a list of lists.
        #
        # One list per MPS position, containing all the tensors of the gates
        # acting on the corresponding position. These lists are originall empty.
        # The last element of each list corresponds to the last gate applied.
        #
        # Each of the tensors will have four bonds ordered as follows:
        # [input, left, right, output]
        self._mpo: list[list[Tensor]] = [[]]*n_tensors

        # Initialise the MPS that we will use as first approximation of the
        # variational algorithm.
        self._aux_mps = MPSxGate(n_tensors, chi, float_precision)

        self._mpo_bond_counter = 0

    def apply_1q_gate(self, position: int, gate: Op) -> None:
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

        # Apply the gate to the MPS with eager approximation
        _aux_mps.apply_1q_gate(self, position, gate)

        # Load the gate's unitary to the GPU memory
        gate_tensor = cp.empty(shape=(2, 2), dtype=self._complex_t)
        gate_tensor.set(gate.get_unitary(), self._stream)

        # Identify the tensor to contract the gate with
        if self._mpo[position]:  # Not empty
            last_tensor = self._mpo[position][-1]
        else:  # Use the MPS tensor
            last_tensor = self.tensors[position]

        # Identify the ID of the bonds involved
        open_bond = last_tensor.bonds[-1]
        other_bonds = last_tensor.bonds[:-1]
        new_bond = -1  # Temporary ID for new open bond

        # Contract
        new_tensor = cq.contract(
            gate_tensor,
            [new_bond, open_bond],
            last_tensor.data,
            last_tensor.bonds,
            other_bonds + [new_bond],
        )

        # Update the tensor; do so "in place" in the MPS-MPO data structures
        last_tensor.data = new_tensor

    def apply_2q_gate(self, positions: tuple[int, int], gate: Op) -> None:
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

        # Apply the gate to the MPS with eager approximation
        _aux_mps.apply_2q_gate(self, positions, gate)

        # Load the gate's unitary to the GPU memory
        gate_tensor = cp.empty(shape=(4, 4), dtype=self._complex_t)
        gate_tensor.set(gate.get_unitary(), self._stream)

        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(gate_tensor, (2, 2, 2, 2))

        # Assign bond IDs
        left_p_bond = self.get_physical_bond(l_pos)
        right_p_bond = self.get_physical_bond(r_pos)
        left_new_p_bond = self._new_bond_id()
        right_new_p_bond = self._new_bond_id()
        new_v_bond = self._new_bond_id()
        left_dummy_bond = self._new_bond_id()
        right_dummy_bond = self._new_bond_id()

        # Create the tensor object for the gate
        if l_pos == positions[0]:
            gate_bonds = [left_new_bond, right_new_p_bond, left_p_bond, right_p_bond]
        else:  # Implicit swap
            gate_bonds = [right_new_bond, left_new_p_bond, right_p_bond, left_p_bond]
        G = Tensor(gate_tensor, gate_bonds)

        # Template of tensors that will store the SVD decomposition of the gate tensor
        L = Tensor(
            cp.empty(shape=(2,1,4,2), dtype=self._complex_t),
            [left_p_bond, left_dummy_bond, new_v_bond, left_new_p_bond],
        )
        R = Tensor(
            cp.empty(shape=(2,4,1,2), dtype=self._complex_t),
            [right_p_bond, new_v_bond, right_dummy_bond, right_new_p_bond],
        )
        S_d = cp.empty(4, dtype=self._real_t)

        # Create tensor descriptors
        G_desc = G.get_tensor_descriptor(self._libhandle)
        L_desc = L.get_tensor_descriptor(self._libhandle)
        R_desc = R.get_tensor_descriptor(self._libhandle)

        # Configure SVD parameters
        svd_config = cutn.create_tensor_svd_config(self._libhandle)

        svd_config_attributes = [
            # Contract the rank-1 tensor of singular values (S) directly
            # into U and V. UV_EQUAL refers to applying U = U*sqrt(S) and
            # similarly for V. Here, U and V are L and R respectively.
            (
                cutn.TensorSVDConfigAttribute.S_PARTITION,
                cutn.TensorSVDPartition.UV_EQUAL,
            ),
        ]

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
        svd_info = cutn.create_tensor_svd_info(self._libhandle)

        # Apply SVD decomposition; no truncation takes place
        cutn.tensor_svd(
            self._libhandle,
            G_desc,
            G.data.data.ptr,
            L_desc,
            L.data.data.ptr,
            S_d.data.ptr,
            R_desc,
            R.data.data.ptr,
            svd_config,
            svd_info,
            0,  # 0 means let cuQuantum manage mem itself
            self._stream.ptr,
        )
        self._stream.synchronize()
        # TODO: these could be precomputed for all OpTypes and stored in a dictionary
        #   so that we only copy from it rather than apply SVD each time. However,
        #   gates with parameters such as ZZPhase might be a challenge.

        # Destroy handles
        cutn.destroy_tensor_descriptor(T_desc)
        cutn.destroy_tensor_descriptor(L_desc)
        cutn.destroy_tensor_descriptor(R_desc)
        cutn.destroy_tensor_svd_config(svd_config)
        cutn.destroy_tensor_svd_info(svd_info)

        # Store L and R
        self._mpo[l_pos].append(L)
        self._mpo[r_pos].append(R)

        # Check whether the MPO is large enough to flush it
        if any(4*len(self._mpo[pos]) > self.chi for pos in [l_pos, r_pos]):
            self._flush()
        # TODO: it'd be good to classify gates that can be SVD'd with virtual dimension
        #   2 and those which require dimension 4.

    def apply_postselection(self, position: int) -> None:
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

        # Apply the gate to the MPS with eager approximation
        _aux_mps.apply_postselection(self, position)

        # Create the tensor of the |0> postselection
        post_tensor = cp.empty(shape=(1, 2), dtype=self._complex_t)
        post_tensor[0][0] = 1
        post_tensor[0][1] = 0

        # Identify the tensor to contract with
        if self._mpo[position]:  # Not empty
            last_tensor = self._mpo[position][-1]
        else:  # Use the MPS tensor
            last_tensor = self.tensors[position]

        # Identify the ID of the bonds involved
        open_bond = last_tensor.bonds[-1]
        other_bonds = last_tensor.bonds[:-1]
        new_bond = -1  # Temporary ID for new open bond

        # Contract
        new_tensor = cq.contract(
            post_tensor,
            [new_bond, open_bond],
            last_tensor.data,
            last_tensor.bonds,
            other_bonds + [new_bond],
        )

        # Update the tensor; do so "in place" in the MPS-MPO data structures
        last_tensor.data = new_tensor

    def get_physical_bond(self, position: int) -> Bond:
        """Return the unique identifier of the physical bond at ``position``.

        Args
            position: A position in the MPS.

        Returns:
            The identifier of the physical bond.
        """
        if position < 0 or position >= len(self):
            raise Exception(f"Position {position} is out of bounds.")

        # Identify the tensor last tensor in the MPO
        if self._mpo[position]:  # Not empty
            last_tensor = self._mpo[position][-1]
        else:  # Use the MPS tensor
            last_tensor = self.tensors[position]

        # By construction, the open bond is the last one
        return last_tensor.bonds[-1]

    def get_physical_dimension(self, position: int) -> int:
        """Return the dimension of the physical bond at ``position``.

        Args:
            position: A position in the MPS.

        Returns:
            The dimension of the physical bond.
        """

        # Identify the tensor last tensor in the MPO
        if self._mpo[position]:  # Not empty
            last_tensor = self._mpo[position][-1]
        else:  # Use the MPS tensor
            last_tensor = self.tensors[position]

        # By construction, the open bond is the last one
        return last_tensor.data.shape[-1]

    def _flush(self) -> None:
        """
        """
        raise NotImplementedError()

    def _new_bond_id(self) -> Bond:
        self._mpo_bond_counter += 1
        return 2*len(self) + self._mpo_bond_counter
