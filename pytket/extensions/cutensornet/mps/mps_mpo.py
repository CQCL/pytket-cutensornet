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

from typing import Optional, Union

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

from pytket.circuit import Op, Qubit  # type: ignore
from .mps import CuTensorNetHandle, DirectionMPS, Bond, Tensor, MPS
from .mps_gate import MPSxGate


class MPSxMPO(MPS):
    """Implements a batched--gate contraction algorithm (DMRG-like) to calculate
    the output state of a circuit as an ``MPS``. The algorithm is described
    in: https://arxiv.org/abs/2207.05612.
    """

    def __init__(
        self,
        libhandle: CuTensorNetHandle,
        qubits: list[Qubit],
        chi: Optional[int] = None,
        truncation_fidelity: Optional[float] = None,
        k: Optional[int] = None,
        optim_delta: Optional[float] = None,
        float_precision: Optional[Union[np.float32, np.float64]] = None,
    ):
        """Initialise an MPS on the computational state ``|0>``.

        Note:
            A ``libhandle`` should be created via a ``with CuTensorNet() as libhandle:``
            statement. The device where the MPS is stored will match the one specified
            by the library handle.

            Providing both a custom ``chi`` and ``truncation_fidelity`` will raise an
            exception. Choose one or the other (or neither, for exact simulation).

        Args:
            libhandle: The cuTensorNet library handle that will be used to carry out
                tensor operations on the MPS.
            qubits: The list of qubits in the circuit to be simulated.
            chi: The maximum value allowed for the dimension of the virtual
                bonds. Higher implies better approximation but more
                computational resources. If not provided, ``chi`` will be set
                to ``2**(len(qubits) // 2)``, which is enough for exact contraction.
            truncation_fidelity: Every time a two-qubit gate is applied, the virtual
                bond will be truncated to the minimum dimension that satisfies
                ``|<psi|phi>|^2 >= trucantion_fidelity``, where ``|psi>`` and ``|phi>``
                are the states before and after truncation (both normalised).
                If not provided, it will default to its maximum value 1.
            k: The maximum number of layers the MPO is allowed to have before
                being contracted. Increasing this might increase fidelity, but
                it will also increase resource requirements exponentially.
                Default value is 4.
            optim_delta: Stopping criteria for the optimisation when contracting the
                ``k`` layers of MPO. Stops when the increase of fidelity between
                iterations is smaller than ``optim_delta``. Default value is ``1e-5``.
            float_precision: The floating point precision used in tensor calculations;
                choose from ``numpy`` types: ``np.float64`` or ``np.float32``.
                Complex numbers are represented using two of such
                ``float`` numbers. Default is ``np.float64``.
        """
        super().__init__(libhandle, qubits, chi, truncation_fidelity, float_precision)

        # Initialise the MPO data structure. This will keep a list of the gates
        # batched for application to the MPS; all of them will be applied at
        # once when deemed appropriate or when calling ._flush(), removing them
        # from here. The gates are kept in a list of lists.
        #
        # One list per MPS position, containing all the tensors of the gates
        # acting on the corresponding position. These lists are originally empty.
        # The last element of each list corresponds to the last gate applied.
        #
        # Each of the tensors will have four bonds ordered as follows:
        # [input, left, right, output]
        self._mpo: list[list[Tensor]] = [list() for _ in qubits]

        # Initialise the MPS that we will use as first approximation of the
        # variational algorithm.
        self._aux_mps = MPSxGate(
            libhandle, qubits, chi, truncation_fidelity, float_precision
        )

        if k is None:
            self.k = 4
        else:
            self.k = k
        if optim_delta is None:
            self.optim_delta = 1e-5
        else:
            self.optim_delta = optim_delta

        self._mpo_bond_counter = 0

    def update_libhandle(self, libhandle: CuTensorNetHandle) -> None:
        """Set the library handle used by this ``MPS`` object. Multiple objects
        may use the same library handle.

        Args:
            libhandle: The new cuTensorNet library handle.

        Raises:
            RuntimeError: If the device (GPU) where ``libhandle`` was initialised
                does not match the one where the tensors of the MPS are stored.
        """
        super().update_libhandle(libhandle)
        self._aux_mps.update_libhandle(libhandle)

    def _apply_1q_gate(self, position: int, gate: Op) -> MPSxMPO:
        """Applies the 1-qubit gate to the MPS.

        This does not increase the dimension of any bond.

        Args:
            position: The position of the MPS tensor that this gate
                is applied to.
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.
        """

        # Apply the gate to the MPS with eager approximation
        self._aux_mps._apply_1q_gate(position, gate)

        # Load the gate's unitary to the GPU memory
        gate_unitary = gate.get_unitary().astype(dtype=self._complex_t, copy=False)
        gate_tensor = cp.asarray(gate_unitary, dtype=self._complex_t)

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
        return self

    def _apply_2q_gate(self, positions: tuple[int, int], gate: Op) -> MPSxMPO:
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

        # Check whether the MPO is large enough to flush it
        if any(len(self._mpo[pos]) >= self.k for pos in [l_pos, r_pos]):
            self._flush()

        # TODO: I should try and do some kind of BFS for the gates.

        # Apply the gate to the MPS with eager approximation
        self._aux_mps._apply_2q_gate(positions, gate)

        # Load the gate's unitary to the GPU memory
        gate_unitary = gate.get_unitary().astype(dtype=self._complex_t, copy=False)
        gate_tensor = cp.asarray(gate_unitary, dtype=self._complex_t)

        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(gate_tensor, (2, 2, 2, 2))

        # Assign bond IDs
        left_gate_input = self.get_physical_bond(l_pos)
        right_gate_input = self.get_physical_bond(r_pos)
        left_gate_output = self._new_bond_id()
        right_gate_output = self._new_bond_id()
        gate_v_bond = self._new_bond_id()
        left_dummy = self._new_bond_id()
        right_dummy = self._new_bond_id()

        # Create the tensor object for the gate
        if l_pos == positions[0]:
            gate_bonds = [
                left_gate_output,
                right_gate_output,
                left_gate_input,
                right_gate_input,
            ]
        else:  # Implicit swap
            gate_bonds = [
                right_gate_output,
                left_gate_output,
                right_gate_input,
                left_gate_input,
            ]
        G = Tensor(gate_tensor, gate_bonds)

        # Template of tensors that will store the SVD decomposition of the gate tensor
        L = Tensor(
            cp.empty(shape=(2, 4, 2), dtype=self._complex_t),
            [left_gate_input, gate_v_bond, left_gate_output],
        )
        R = Tensor(
            cp.empty(shape=(2, 4, 2), dtype=self._complex_t),
            [right_gate_input, gate_v_bond, right_gate_output],
        )
        S_d = cp.empty(4, dtype=self._real_t)

        # Create tensor descriptors
        assert self._lib is not None
        G_desc = G.get_tensor_descriptor(self._lib)
        L_desc = L.get_tensor_descriptor(self._lib)
        R_desc = R.get_tensor_descriptor(self._lib)

        # Configure SVD parameters
        svd_config = cutn.create_tensor_svd_config(self._lib.handle)

        svd_config_attributes = [
            # TensorSVDPartition.US asks that cuTensorNet automatically
            # contracts the tensor of singular values (S) into one of the
            # two tensors (U), named L in our case.
            (
                cutn.TensorSVDConfigAttribute.S_PARTITION,
                cutn.TensorSVDPartition.US,
            ),
        ]

        for attr, value in svd_config_attributes:
            attr_dtype = cutn.tensor_svd_config_get_attribute_dtype(attr)
            value = np.array([value], dtype=attr_dtype)
            cutn.tensor_svd_config_set_attribute(
                self._lib.handle,
                svd_config,
                attr,
                value.ctypes.data,
                value.dtype.itemsize,
            )
        svd_info = cutn.create_tensor_svd_info(self._lib.handle)

        # Apply SVD decomposition; no truncation takes place
        cutn.tensor_svd(
            self._lib.handle,
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
            self._stream.ptr,  # type: ignore
        )
        self._stream.synchronize()  # type: ignore
        # TODO: these could be precomputed for all OpTypes and stored in a dictionary
        #   so that we only copy from it rather than apply SVD each time. However,
        #   gates with parameters such as ZZPhase might be a challenge.
        # TODO: it'd be good to classify gates that can be SVD'd with virtual dimension
        #   2 and those which require dimension 4.

        # Destroy handles
        cutn.destroy_tensor_descriptor(G_desc)
        cutn.destroy_tensor_descriptor(L_desc)
        cutn.destroy_tensor_descriptor(R_desc)
        cutn.destroy_tensor_svd_config(svd_config)
        cutn.destroy_tensor_svd_info(svd_info)

        # Add dummy bonds of dimension 1 to L and R so that they have the right shape
        L.data = cp.reshape(L.data, (2, 1, 4, 2))
        L.bonds = [left_gate_input, left_dummy, gate_v_bond, left_gate_output]
        R.data = cp.reshape(R.data, (2, 4, 1, 2))
        R.bonds = [right_gate_input, gate_v_bond, right_dummy, right_gate_output]

        # Store L and R
        self._mpo[l_pos].append(L)
        self._mpo[r_pos].append(R)
        return self

    def get_physical_bond(self, position: int) -> Bond:
        """Returns the unique identifier of the physical bond at ``position``.

        Args
            position: A position in the MPS.

        Returns:
            The identifier of the physical bond.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """
        if position < 0 or position >= len(self):
            raise RuntimeError(f"Position {position} is out of bounds.")

        # Identify the tensor last tensor in the MPO
        if self._mpo[position]:  # Not empty
            last_tensor = self._mpo[position][-1]
        else:  # Use the MPS tensor
            last_tensor = self.tensors[position]

        # By construction, the open bond is the last one
        return last_tensor.bonds[-1]

    def get_physical_dimension(self, position: int) -> int:
        """Returns the dimension of the physical bond at ``position``.

        Args:
            position: A position in the MPS.

        Returns:
            The dimension of the physical bond.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """

        # Identify the tensor last tensor in the MPO
        if self._mpo[position]:  # Not empty
            last_tensor = self._mpo[position][-1]
        else:  # Use the MPS tensor
            last_tensor = self.tensors[position]

        # By construction, the open bond is the last one
        return int(last_tensor.data.shape[-1])

    def _flush(self) -> None:
        """Applies all batched operations within ``self._mpo`` to the MPS.

        The method applies variational optimisation of the MPS until it
        converges. Based on https://arxiv.org/abs/2207.05612.
        """

        l_cached_tensors: list[Tensor] = []
        r_cached_tensors: list[Tensor] = []

        def update_sweep_cache(pos: int, direction: DirectionMPS) -> None:
            """Given a position in the MPS and a sweeping direction (see
            ``DirectionMPS``), calculate the tensor of the partial contraction
            of all MPS-MPO-vMPS* columns from ``pos`` towards ``direction``.
            Update the cache accordingly. Applies canonicalisation on the vMPS
            tensor before contracting.
            """

            # Canonicalise the tensor at ``pos``
            if direction == DirectionMPS.LEFT:
                self._aux_mps.canonicalise_tensor(pos, form=DirectionMPS.RIGHT)
            elif direction == DirectionMPS.RIGHT:
                self._aux_mps.canonicalise_tensor(pos, form=DirectionMPS.LEFT)

            # Get the interleaved representation
            interleaved_rep = [
                # The (conjugated) tensor of the variational MPS
                self._aux_mps.tensors[pos].data.conj(),
                [-b for b in self._aux_mps.tensors[pos].bonds],
                # The tensor of the MPS
                self.tensors[pos].data,
                self.tensors[pos].bonds,
            ]
            for mpo_tensor in self._mpo[pos]:
                # The MPO tensors at this position
                interleaved_rep.append(mpo_tensor.data)
                interleaved_rep.append(mpo_tensor.bonds)
            # The output bond of the last tensor must connect to the physical
            # bond of the corresponding ``self._aux_mps`` tensor
            interleaved_rep[-1][-1] = -self._aux_mps.get_physical_bond(pos)

            # Also contract the previous (cached) tensor during the sweep
            if direction == DirectionMPS.LEFT:
                if pos != len(self) - 1:  # Otherwise, there is nothing cached yet
                    interleaved_rep.append(r_cached_tensors[-1].data)
                    interleaved_rep.append(r_cached_tensors[-1].bonds)
            elif direction == DirectionMPS.RIGHT:
                if pos != 0:  # Otherwise, there is nothing cached yet
                    interleaved_rep.append(l_cached_tensors[-1].data)
                    interleaved_rep.append(l_cached_tensors[-1].bonds)

            # Figure out the ID of the bonds of the contracted tensor
            if direction == DirectionMPS.LEFT:
                # Take the left virtual bond of both of the MPS
                T_bonds = [
                    -self._aux_mps.get_virtual_bonds(pos)[0],
                    self.get_virtual_bonds(pos)[0],
                ]
                # Take the left bond of each of the MPO tensors
                for mpo_tensor in self._mpo[pos]:
                    T_bonds.append(mpo_tensor.bonds[1])
            elif direction == DirectionMPS.RIGHT:
                # Take the right virtual bond of both of the MPS
                T_bonds = [
                    -self._aux_mps.get_virtual_bonds(pos)[-1],
                    self.get_virtual_bonds(pos)[-1],
                ]
                # Take the right bond of each of the MPO tensors
                for mpo_tensor in self._mpo[pos]:
                    T_bonds.append(mpo_tensor.bonds[2])
            # Append the bond IDs of the resulting tensor to the interleaved_rep
            interleaved_rep.append(T_bonds)

            # Contract and store
            T = Tensor(
                cq.contract(*interleaved_rep),
                T_bonds,
            )
            if direction == DirectionMPS.LEFT:
                r_cached_tensors.append(T)
            elif direction == DirectionMPS.RIGHT:
                l_cached_tensors.append(T)

        def update_variational_tensor(
            pos: int, left_tensor: Optional[Tensor], right_tensor: Optional[Tensor]
        ) -> float:
            """Update the tensor at ``pos`` of the variational MPS using ``left_tensor``
            (and ``right_tensor``) which is meant to contain the contraction of all
            the left (and right) columns of the MPS-MPO-vMPS* network from ``pos``.
            Contract these with the MPS-MPO column at ``pos``.
            Return the current fidelity of this sweep.
            """
            interleaved_rep = [
                # The tensor of the MPS
                self.tensors[pos].data,
                self.tensors[pos].bonds,
            ]
            # The MPO tensors at position ``pos``
            for mpo_tensor in self._mpo[pos]:
                interleaved_rep.append(mpo_tensor.data)
                interleaved_rep.append(mpo_tensor.bonds)
            # The output bond of the last tensor must connect to the physical
            # bond of the corresponding ``self._aux_mps`` tensor
            interleaved_rep[-1][-1] = -self._aux_mps.get_physical_bond(pos)

            if left_tensor is not None:
                interleaved_rep.append(left_tensor.data)
                interleaved_rep.append(left_tensor.bonds)
            if right_tensor is not None:
                interleaved_rep.append(right_tensor.data)
                interleaved_rep.append(right_tensor.bonds)

            # Append the bond IDs of the resulting tensor
            F_bonds = [-b for b in self._aux_mps.tensors[pos].bonds]
            interleaved_rep.append(F_bonds)

            # Contract and store tensor
            F = Tensor(cq.contract(*interleaved_rep), self._aux_mps.tensors[pos].bonds)

            # Get the fidelity
            optim_fidelity = complex(
                cq.contract(F.data.conj(), F.bonds, F.data, F.bonds, [])
            )
            assert np.isclose(optim_fidelity.imag, 0.0, atol=self._atol)
            optim_fidelity = float(optim_fidelity.real)

            # Normalise F and update the variational MPS
            self._aux_mps.tensors[pos].data = F.data / np.sqrt(optim_fidelity)

            return optim_fidelity

        ##################################
        # Variational sweeping algorithm #
        ##################################

        # Begin by doing a sweep towards the left that does not update
        # the variational tensors, but simply loads up the ``r_cached_tensors``
        for pos in reversed(range(1, len(self))):
            update_sweep_cache(pos, direction=DirectionMPS.LEFT)

        prev_fidelity = -1.0  # Dummy value
        sweep_fidelity = 0.0  # Dummy value

        # Repeat sweeps until the fidelity converges
        sweep_direction = DirectionMPS.RIGHT
        while not np.isclose(prev_fidelity, sweep_fidelity, atol=self.optim_delta):
            prev_fidelity = sweep_fidelity

            if sweep_direction == DirectionMPS.RIGHT:
                sweep_fidelity = update_variational_tensor(
                    pos=0, left_tensor=None, right_tensor=r_cached_tensors.pop()
                )
                update_sweep_cache(pos=0, direction=DirectionMPS.RIGHT)

                for pos in range(1, len(self) - 1):
                    sweep_fidelity = update_variational_tensor(
                        pos=pos,
                        left_tensor=l_cached_tensors[-1],
                        right_tensor=r_cached_tensors.pop(),
                    )
                    update_sweep_cache(pos, direction=DirectionMPS.RIGHT)
                # The last variational tensor is not updated;
                # it'll be the first in the next sweep

                sweep_direction = DirectionMPS.LEFT

            elif sweep_direction == DirectionMPS.LEFT:
                sweep_fidelity = update_variational_tensor(
                    pos=len(self) - 1,
                    left_tensor=l_cached_tensors.pop(),
                    right_tensor=None,
                )
                update_sweep_cache(pos=len(self) - 1, direction=DirectionMPS.LEFT)

                for pos in reversed(range(1, len(self) - 1)):
                    sweep_fidelity = update_variational_tensor(
                        pos=pos,
                        left_tensor=l_cached_tensors.pop(),
                        right_tensor=r_cached_tensors[-1],
                    )
                    update_sweep_cache(pos, direction=DirectionMPS.LEFT)
                # The last variational tensor is not updated;
                # it'll be the first in the next sweep

                sweep_direction = DirectionMPS.RIGHT

        # Clear out the MPO
        self._mpo = [list() for _ in range(len(self))]
        self._mpo_bond_counter = 0

        # Update the MPS tensors
        self.tensors = [t.copy() for t in self._aux_mps.tensors]

        # Update the fidelity estimate
        self.fidelity *= sweep_fidelity
        self._aux_mps.fidelity = self.fidelity

    def _new_bond_id(self) -> Bond:
        self._mpo_bond_counter += 1
        return 2 * len(self) + self._mpo_bond_counter
