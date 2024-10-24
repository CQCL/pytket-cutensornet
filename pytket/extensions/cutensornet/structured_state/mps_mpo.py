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

from typing import Optional, Union

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

from pytket.circuit import Qubit, Bit
from pytket.extensions.cutensornet import CuTensorNetHandle
from .general import Tensor, Config
from .mps import (
    DirMPS,
    MPS,
)
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
        config: Config,
        bits: Optional[list[Bit]] = None,
    ):
        """Initialise an MPS on the computational state ``|0>``.

        Note:
            A ``libhandle`` should be created via a ``with CuTensorNet() as libhandle:``
            statement. The device where the MPS is stored will match the one specified
            by the library handle.

        Args:
            libhandle: The cuTensorNet library handle that will be used to carry out
                tensor operations on the MPS.
            qubits: The list of qubits in the circuit to be simulated.
            config: The object describing the configuration for simulation.
        """
        super().__init__(libhandle, qubits, config, bits=bits)

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
        # This ``_bond_ids`` store global bond IDs of MPO tensors, used by ``_flush()``
        self._bond_ids: list[list[tuple[int, int, int, int]]] = [list() for _ in qubits]

        # Initialise the MPS that we will use as first approximation of the
        # variational algorithm.
        self._aux_mps = MPSxGate(libhandle, qubits, config, bits=bits)

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

    def apply_qubit_relabelling(self, qubit_map: dict[Qubit, Qubit]) -> MPSxMPO:
        """Relabels each qubit ``q`` as ``qubit_map[q]``.

        This does not apply any SWAP gate, nor it changes the internal structure of the
        state. It simply changes the label of the physical bonds of the tensor network.

        Args:
            qubit_map: Dictionary mapping each qubit to its new label.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            ValueError: If any of the keys in ``qubit_map`` are not qubits in the state.
        """
        self._aux_mps.apply_qubit_relabelling(qubit_map)
        super().apply_qubit_relabelling(qubit_map)
        return self

    def add_qubit(self, new_qubit: Qubit, position: int, state: int = 0) -> MPS:
        """Adds a qubit at the specified position.

        Args:
            new_qubit: The identifier of the qubit to be added to the state.
            position: The location the new qubit should be inserted at in the MPS.
                Qubits on this and later indexed have their position shifted by 1.
            state: Choose either ``0`` or ``1`` for the new qubit's state.
                Defaults to ``0``.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            ValueError: If ``new_qubit`` already exists in the state.
            ValueError: If ``position`` is negative or larger than ``len(self)``.
            ValueError: If ``state`` is not ``0`` or ``1``.
        """
        raise NotImplementedError(
            "Creating new qubits is not currently supported in MPSxMPO."
        )

    def _apply_1q_unitary(self, unitary: cp.ndarray, qubit: Qubit) -> MPSxMPO:
        """Applies the 1-qubit unitary to the MPS.

        This does not increase the dimension of any bond.

        Args:
            unitary: The unitary to be applied.
            qubit: The qubit the unitary acts on.

        Returns:
            ``self``, to allow for method chaining.
        """
        position = self.qubit_position[qubit]

        # Apply the gate to the MPS with eager approximation
        self._aux_mps._apply_1q_unitary(unitary, qubit)

        if len(self) == 1:
            # If there is only one qubit, there is no benefit in using an MPO, so
            # simply copy from the standard MPS
            self.tensors[0] = self._aux_mps.tensors[0].copy()
            return self

        # Glossary of bond IDs
        # i -> input to the MPO tensor
        # o -> output of the MPO tensor
        # l -> left virtual bond of the MPO tensor
        # r -> right virtual bond of the MPO tensor
        # g -> output bond of the gate tensor

        # Identify the tensor to contract the gate with
        if self._mpo[position]:  # Not empty
            last_tensor = self._mpo[position][-1]
            last_bonds = "ilro"
            new_bonds = "ilrg"
        else:  # Use the MPS tensor
            last_tensor = self.tensors[position]
            last_bonds = "lro"
            new_bonds = "lrg"

        # Contract
        new_tensor = cq.contract(
            "go," + last_bonds + "->" + new_bonds,
            unitary,
            last_tensor,
            options={"handle": self._lib.handle, "device_id": self._lib.device_id},
            optimize={"path": [(0, 1)]},
        )

        # Update the tensor
        if self._mpo[position]:  # Not empty
            self._mpo[position][-1] = new_tensor
        else:  # Update the MPS tensor
            self.tensors[position] = new_tensor

        return self

    def _apply_2q_unitary(self, unitary: cp.ndarray, q0: Qubit, q1: Qubit) -> MPSxMPO:
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

        # Check whether the MPO is large enough to flush it
        if any(len(self._mpo[pos]) >= self._cfg.k for pos in [l_pos, r_pos]):
            self._flush()

        # Apply the gate to the MPS with eager approximation
        self._aux_mps._apply_2q_unitary(unitary, q0, q1)

        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(unitary, (2, 2, 2, 2))

        # Glossary of bond IDs
        # l -> gate's left input bond
        # r -> gate's right input bond
        # L -> gate's left output bond
        # R -> gate's right output bond
        # s -> virtual bond after QR decomposition

        # Assign the bond IDs for the gate
        if l_pos == positions[0]:
            gate_bonds = "LRlr"
        else:  # Implicit swap
            gate_bonds = "RLrl"

        # Apply SVD on the gate tensor to remove any zero singular values ASAP
        svd_method = tensor.SVDMethod(
            abs_cutoff=self._cfg.zero,
            partition="U",  # Contract S directly into U (named L in this case)
        )
        # Apply the SVD decomposition using the configuration defined above
        L, S, R = tensor.decompose(
            f"{gate_bonds}->lsL,rsR", gate_tensor, method=svd_method, options=options
        )
        assert S is None  # Due to "partition" option in SVDMethod
        dim = L.shape[1]  # Dimension of the shared bond `s`

        # Add dummy bonds of dimension 1 to L and R so that they have the right shape
        L = cp.reshape(L, (2, 1, dim, 2))
        R = cp.reshape(R, (2, dim, 1, 2))

        # Store L and R
        self._mpo[l_pos].append(L)
        self._mpo[r_pos].append(R)
        # If `l_pos` and `r_pos` are distant, add identity tensor on all
        # intermediate positions
        if r_pos - l_pos != 1:
            # Identity between input/output at physical bonds
            p_identity = cp.eye(2, dtype=self._cfg._complex_t)
            # Identity between left/right virtual bonds
            v_identity = cp.eye(dim, dtype=self._cfg._complex_t)
            # Create a "crossing" tensor by applying tensor product of these
            crossing = cq.contract(
                "io,lr->ilro",
                p_identity,
                v_identity,
                options=options,
                optimize={"path": [(0, 1)]},
            )
        # Store the intermediate tensors
        for pos in range(l_pos + 1, r_pos):
            self._mpo[pos].append(crossing.copy())

        # And assign their global bonds
        shared_bond_id = self._new_bond_id()
        self._bond_ids[l_pos].append(
            (
                self._get_physical_bond(l_pos),
                self._new_bond_id(),
                shared_bond_id,
                self._new_bond_id(),
            )
        )
        for pos in range(l_pos + 1, r_pos):
            next_shared_bond_id = self._new_bond_id()
            self._bond_ids[pos].append(
                (
                    self._get_physical_bond(pos),
                    shared_bond_id,
                    next_shared_bond_id,
                    self._new_bond_id(),
                )
            )
            shared_bond_id = next_shared_bond_id
        self._bond_ids[r_pos].append(
            (
                self._get_physical_bond(r_pos),
                shared_bond_id,
                self._new_bond_id(),
                self._new_bond_id(),
            )
        )

        return self

    def get_physical_dimension(self, position: int) -> int:
        """Returns the dimension of the physical bond at ``position``.

        Args:
            position: A position in the MPS.

        Returns:
            The dimension of the physical bond.

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
        return int(last_tensor.shape[-1])

    def _get_physical_bond(self, position: int) -> int:
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

        if self._bond_ids[position]:
            return self._bond_ids[position][-1][-1]
        else:
            return self._new_bond_id()

    def _get_column_bonds(self, position: int, direction: DirMPS) -> list[int]:
        """Returns the unique identifier of all the left (right) virtual bonds of
        MPO tensors at ``position`` if ``direction`` is ``LEFT`` (``RIGHT``).

        Notes:
            It does not return the corresponding bonds of the MPS tensors.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
            ValueError: If ``direction`` is not a value in ``DirMPS``.
        """
        if position < 0 or position >= len(self):
            raise RuntimeError(f"Position {position} is out of bounds.")

        if direction == DirMPS.LEFT:
            index = 1  # By convention, left bond at index 1
        elif direction == DirMPS.RIGHT:
            index = 2  # By convention, right bond at index 2
        else:
            raise ValueError("Argument form must be a value in DirMPS.")

        return [b_ids[index] for b_ids in self._bond_ids[position]]

    def _flush(self) -> None:
        """Applies all batched operations within ``self._mpo`` to the MPS.

        The method applies variational optimisation of the MPS until it
        converges. Based on https://arxiv.org/abs/2207.05612.
        """
        if self._mpo_bond_counter == 0:
            # MPO is empty, there is nothing to flush
            return None

        self._logger.info("Applying variational optimisation.")
        self._logger.info(f"Fidelity before optimisation={self._aux_mps.fidelity}")

        l_cached_tensors: list[Tensor] = []
        r_cached_tensors: list[Tensor] = []

        def update_sweep_cache(pos: int, direction: DirMPS) -> None:
            """Given a position in the MPS and a sweeping direction (see
            ``DirMPS``), calculate the tensor of the partial contraction
            of all MPS-MPO-vMPS* columns from ``pos`` towards ``direction``.
            Update the cache accordingly. Applies canonicalisation on the vMPS
            tensor before contracting.
            """
            self._logger.debug("Updating the sweep cache...")

            # Canonicalise the tensor at ``pos``
            if direction == DirMPS.LEFT:
                self._aux_mps.canonicalise_tensor(pos, form=DirMPS.RIGHT)
            elif direction == DirMPS.RIGHT:
                self._aux_mps.canonicalise_tensor(pos, form=DirMPS.LEFT)

            # Glossary of bond IDs
            # p -> the physical bond of the MPS tensor
            # l,r -> the virtual bonds of the MPS tensor
            # L,R -> the virtual bonds of the variational MPS tensor
            # P -> the physical bond of the variational MPS tensor
            # MPO tensors will use ``self._bond_ids``

            # Get the interleaved representation
            interleaved_rep = [
                # The tensor of the MPS
                self.tensors[pos],
                ["l", "r", "p"],
                # The (conjugated) tensor of the variational MPS
                self._aux_mps.tensors[pos].conj(),
                ["L", "R", "P" if self._mpo[pos] else "p"],
            ]
            for i, mpo_tensor in enumerate(self._mpo[pos]):
                # The MPO tensor at this position
                interleaved_rep.append(mpo_tensor)

                mpo_bonds: list[Union[int, str]] = list(self._bond_ids[pos][i])
                if i == 0:
                    # The input bond of the first MPO tensor must connect to the
                    # physical bond of the correspondong ``self.tensors`` tensor
                    mpo_bonds[0] = "p"
                if i == len(self._mpo[pos]) - 1:
                    # The output bond of the last MPO tensor must connect to the
                    # physical bond of the corresponding ``self._aux_mps`` tensor
                    mpo_bonds[-1] = "P"
                interleaved_rep.append(mpo_bonds)

            # Also contract the previous (cached) tensor during the sweep
            if direction == DirMPS.LEFT:
                if pos != len(self) - 1:  # Otherwise, there is nothing cached yet
                    interleaved_rep.append(r_cached_tensors[-1])
                    r_cached_bonds = self._get_column_bonds(pos + 1, DirMPS.LEFT)
                    interleaved_rep.append(["r", "R"] + r_cached_bonds)
            elif direction == DirMPS.RIGHT:
                if pos != 0:  # Otherwise, there is nothing cached yet
                    interleaved_rep.append(l_cached_tensors[-1])
                    l_cached_bonds = self._get_column_bonds(pos - 1, DirMPS.RIGHT)
                    interleaved_rep.append(["l", "L"] + l_cached_bonds)

            # Figure out the ID of the bonds of the contracted tensor
            if direction == DirMPS.LEFT:
                # Take the left bond of each of the MPO tensors
                result_bonds = self._get_column_bonds(pos, DirMPS.LEFT)
                # Take the left virtual bond of both of the MPS
                interleaved_rep.append(["l", "L"] + result_bonds)
            elif direction == DirMPS.RIGHT:
                # Take the right bond of each of the MPO tensors
                result_bonds = self._get_column_bonds(pos, DirMPS.RIGHT)
                # Take the right virtual bond of both of the MPS
                interleaved_rep.append(["r", "R"] + result_bonds)

            # Contract and store
            T = cq.contract(
                *interleaved_rep,
                options={"handle": self._lib.handle, "device_id": self._lib.device_id},
                optimize={"samples": 0},
            )
            if direction == DirMPS.LEFT:
                r_cached_tensors.append(T)
            elif direction == DirMPS.RIGHT:
                l_cached_tensors.append(T)

            self._logger.debug("Completed update of the sweep cache.")

        def update_variational_tensor(
            pos: int, left_tensor: Optional[Tensor], right_tensor: Optional[Tensor]
        ) -> float:
            """Update the tensor at ``pos`` of the variational MPS using ``left_tensor``
            (and ``right_tensor``) which is meant to contain the contraction of all
            the left (and right) columns of the MPS-MPO-vMPS* network from ``pos``.
            Contract these with the MPS-MPO column at ``pos``.
            Return the current fidelity of this sweep.
            """
            self._logger.debug(f"Optimising tensor at position={pos}")

            interleaved_rep = [
                # The tensor of the MPS
                self.tensors[pos],
                ["l", "r", "p"],
            ]
            result_bonds = ["l", "r", "p"]

            # The MPO tensors at position ``pos``
            for i, mpo_tensor in enumerate(self._mpo[pos]):
                # The MPO tensor at this position
                interleaved_rep.append(mpo_tensor)

                mpo_bonds: list[Union[int, str]] = list(self._bond_ids[pos][i])
                if i == 0:
                    # The input bond of the first MPO tensor must connect to the
                    # physical bond of the correspondong ``self.tensors`` tensor
                    mpo_bonds[0] = "p"
                if i == len(self._mpo[pos]) - 1:
                    # The output bond of the last MPO tensor corresponds to the
                    # physical bond of the corresponding ``self._aux_mps`` tensor
                    mpo_bonds[-1] = "P"
                    result_bonds[-1] = "P"
                interleaved_rep.append(mpo_bonds)

            if left_tensor is not None:
                interleaved_rep.append(left_tensor)
                left_tensor_bonds = self._get_column_bonds(pos - 1, DirMPS.RIGHT)
                interleaved_rep.append(["l", "L"] + left_tensor_bonds)
                result_bonds[0] = "L"
            if right_tensor is not None:
                interleaved_rep.append(right_tensor)
                right_tensor_bonds = self._get_column_bonds(pos + 1, DirMPS.LEFT)
                interleaved_rep.append(["r", "R"] + right_tensor_bonds)
                result_bonds[1] = "R"

            # Append the bond IDs of the resulting tensor
            interleaved_rep.append(result_bonds)

            # Contract and store tensor
            F = cq.contract(
                *interleaved_rep,
                options={"handle": self._lib.handle, "device_id": self._lib.device_id},
                optimize={"samples": 0},
            )

            # Get the fidelity
            optim_fidelity = complex(
                cq.contract(
                    "LRP,LRP->",
                    F.conj(),
                    F,
                    options={
                        "handle": self._lib.handle,
                        "device_id": self._lib.device_id,
                    },
                    optimize={"path": [(0, 1)]},
                )
            )
            assert np.isclose(optim_fidelity.imag, 0.0, atol=self._cfg._atol)
            optim_fidelity = float(optim_fidelity.real)

            # Normalise F and update the variational MPS
            self._aux_mps.tensors[pos] = F / cp.sqrt(
                optim_fidelity, dtype=self._cfg._complex_t
            )

            return optim_fidelity

        ##################################
        # Variational sweeping algorithm #
        ##################################

        # Begin by doing a sweep towards the left that does not update
        # the variational tensors, but simply loads up the ``r_cached_tensors``
        for pos in reversed(range(1, len(self))):
            update_sweep_cache(pos, direction=DirMPS.LEFT)

        prev_fidelity = -1.0  # Dummy value
        sweep_fidelity = 0.0  # Dummy value

        # Repeat sweeps until the fidelity converges
        sweep_direction = DirMPS.RIGHT
        while not np.isclose(prev_fidelity, sweep_fidelity, atol=self._cfg.optim_delta):
            self._logger.info(f"Doing another optimisation sweep...")
            prev_fidelity = sweep_fidelity

            if sweep_direction == DirMPS.RIGHT:
                sweep_fidelity = update_variational_tensor(
                    pos=0, left_tensor=None, right_tensor=r_cached_tensors.pop()
                )
                update_sweep_cache(pos=0, direction=DirMPS.RIGHT)

                for pos in range(1, len(self) - 1):
                    sweep_fidelity = update_variational_tensor(
                        pos=pos,
                        left_tensor=l_cached_tensors[-1],
                        right_tensor=r_cached_tensors.pop(),
                    )
                    update_sweep_cache(pos, direction=DirMPS.RIGHT)
                # The last variational tensor is not updated;
                # it'll be the first in the next sweep

                sweep_direction = DirMPS.LEFT

            elif sweep_direction == DirMPS.LEFT:
                sweep_fidelity = update_variational_tensor(
                    pos=len(self) - 1,
                    left_tensor=l_cached_tensors.pop(),
                    right_tensor=None,
                )
                update_sweep_cache(pos=len(self) - 1, direction=DirMPS.LEFT)

                for pos in reversed(range(1, len(self) - 1)):
                    sweep_fidelity = update_variational_tensor(
                        pos=pos,
                        left_tensor=l_cached_tensors.pop(),
                        right_tensor=r_cached_tensors[-1],
                    )
                    update_sweep_cache(pos, direction=DirMPS.LEFT)
                # The last variational tensor is not updated;
                # it'll be the first in the next sweep

                sweep_direction = DirMPS.RIGHT

            self._logger.info(
                "Optimisation sweep completed. "
                f"Current fidelity={self.fidelity*sweep_fidelity}"
            )

        # Clear out the MPO
        self._mpo = [list() for _ in range(len(self))]
        self._bond_ids = [list() for _ in range(len(self))]
        self._mpo_bond_counter = 0

        # Update the MPS tensors
        self.tensors = [t.copy() for t in self._aux_mps.tensors]

        # Update the fidelity estimate
        self.fidelity *= sweep_fidelity
        self._aux_mps.fidelity = self.fidelity

        self._logger.info(f"Final fidelity after optimisation={self.fidelity}")

    def _new_bond_id(self) -> int:
        self._mpo_bond_counter += 1
        return self._mpo_bond_counter
