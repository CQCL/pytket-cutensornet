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
from typing import Any, Optional, Union
from enum import Enum

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

from pytket.circuit import Command, Op, Qubit  # type: ignore

# An alias so that `intptr_t` from CuQuantum's API (which is not available in
# base python) has some meaningful type name.
Handle = int
# An alias for the type of the unique identifiers of each of the bonds
# of the MPS.
Bond = int


class DirectionMPS(Enum):
    """An enum to refer to relative directions within the MPS."""

    LEFT = 0
    RIGHT = 1


class CuTensorNetHandle:
    """Initialise the cuTensorNet library with automatic workspace memory
    management.

    Note:
        Always use as ``with CuTensorNetHandle() as libhandle:`` so that cuTensorNet
        handles are automatically destroyed at the end of execution.

    Attributes:
        handle (int): The cuTensorNet library handle created by this initialisation.
        device_id (int): The ID of the device (GPU) where cuTensorNet is initialised.
            If not provided, defaults to ``cp.cuda.Device()``.
    """

    def __init__(self, device_id: Optional[int] = None):
        self.handle = cutn.create()
        self._is_destroyed = False
        dev = cp.cuda.Device(device_id)
        self.device_id = int(dev)

        if cp.cuda.runtime.runtimeGetVersion() < 11020:
            raise RuntimeError("Requires CUDA 11.2+.")
        if not dev.attributes["MemoryPoolsSupported"]:
            raise RuntimeError("Device does not support CUDA Memory pools")

        # Avoid shrinking the pool
        mempool = cp.cuda.runtime.deviceGetDefaultMemPool(dev.id)
        if int(cp.__version__.split(".")[0]) >= 10:
            # this API is exposed since CuPy v10
            cp.cuda.runtime.memPoolSetAttribute(
                mempool,
                cp.cuda.runtime.cudaMemPoolAttrReleaseThreshold,
                0xFFFFFFFFFFFFFFFF,  # = UINT64_MAX
            )

        # A device memory handler lets CuTensorNet manage its own GPU memory
        def malloc(size, stream):  # type: ignore
            return cp.cuda.runtime.mallocAsync(size, stream)

        def free(ptr, size, stream):  # type: ignore
            cp.cuda.runtime.freeAsync(ptr, stream)

        memhandle = (malloc, free, "memory_handler")
        cutn.set_device_mem_handler(self.handle, memhandle)

    def __enter__(self) -> CuTensorNetHandle:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        cutn.destroy(self.handle)
        self._is_destroyed = True


class Tensor:
    """Class for the management of tensors via CuPy and cuTensorNet.

    It abstracts away some of the low-level API of cuTensorNet.

    Attributes:
        data (cupy.ndarray): The entries of the tensor arranged in a CuPy ndarray.
        bonds (list[Bond]): A list of IDs for each bond, matching the same order
            as in ``self.data.shape`` (which provides the dimension of each).
        canonical_form (Optional[DirectionMPS]): If in canonical form, indicate
            which one; else None.
    """

    def __init__(self, data: cp.ndarray, bonds: list[Bond]):
        """
        Args:
            data: The entries of the tensor arranged in a CuPy ndarray.
            bonds: A list of IDs for each bond, matching the same order
                as in ``self.data.shape`` (which provides the dimension of each).
        """
        self.data = data
        self.bonds = bonds
        self.canonical_form: Optional[DirectionMPS] = None

    def get_tensor_descriptor(self, libhandle: CuTensorNetHandle) -> Handle:
        """Return the cuTensorNet tensor descriptor.

        Note:
            The user is responsible of destroying the descriptor once
            not in use (see ``cuquantum.cutensornet.destroy_tensor_descriptor``).

        Args:
            libhandle: The cuTensorNet library handle.

        Returns:
            The handle to the tensor descriptor.

        Raises:
            RuntimeError: If ``libhandle`` is no longer in scope.
            TypeError: If the type of the tensor is not supported. Supported types are
                ``np.float32``, ``np.float64``, ``np.complex64`` and ``np.complex128``.
        """
        if libhandle._is_destroyed:
            raise RuntimeError("The library handle you passed is no longer in scope.")
        if self.data.dtype == np.float32:
            cq_dtype = cq.cudaDataType.CUDA_R_32F
        elif self.data.dtype == np.float64:
            cq_dtype = cq.cudaDataType.CUDA_R_64F
        elif self.data.dtype == np.complex64:
            cq_dtype = cq.cudaDataType.CUDA_C_32F
        elif self.data.dtype == np.complex128:
            cq_dtype = cq.cudaDataType.CUDA_C_64F
        else:
            raise TypeError(
                f"The data type {self.data.dtype} of the tensor is not supported."
            )

        return cutn.create_tensor_descriptor(  # type: ignore
            handle=libhandle.handle,
            n_modes=len(self.data.shape),
            extents=self.data.shape,
            strides=self._get_cuquantum_strides(),
            modes=self.bonds,
            data_type=cq_dtype,
        )

    def _get_cuquantum_strides(self) -> list[int]:
        """Return a list of the strides for each of the bonds.

        Returns them in the same order as in ``self.bonds``.

        Returns:
            List of strides in cuQuantum format (#entries).
        """
        return [stride // self.data.itemsize for stride in self.data.strides]

    def get_bond_dimension(self, bond: Bond) -> int:
        """Given a bond ID, return that bond's dimension.

        Args:
            bond: The ID of the bond to be queried. If not in the tensor,
                an exception is raised.

        Returns:
            The dimension of the bond.

        Raises:
            RuntimeError: If ``bond`` is not in a Tensor.
        """
        if bond not in self.bonds:
            raise RuntimeError(f"Bond {bond} not in tensor with bonds: {self.bonds}.")
        return int(self.data.shape[self.bonds.index(bond)])

    def copy(self) -> Tensor:
        """
        Returns:
            A deep copy of the Tensor.
        """
        other = Tensor(self.data.copy(), self.bonds.copy())
        other.canonical_form = self.canonical_form
        return other


class MPS:
    """Represents a state as a Matrix Product State.

    Attributes:
        chi (int): The maximum allowed dimension of a virtual bond.
        truncation_fidelity (float): The target fidelity of SVD truncation.
        tensors (list[Tensor]): A list of tensors in the MPS; ``tensors[0]`` is
            the leftmost and ``tensors[len(self)-1]`` is the rightmost; ``tensors[i]``
            and ``tensors[i+1]`` are connected in the MPS via a bond.
        qubit_position (dict[pytket.circuit.Qubit, int]): A dictionary mapping circuit
            qubits to the position its tensor is at in the MPS.
        fidelity (float): A lower bound of the fidelity, obtained by multiplying
            the fidelities after each contraction. The fidelity of a contraction
            corresponds to ``|<psi|phi>|^2`` where ``|psi>`` and ``|phi>`` are the
            states before and after truncation (assuming both are normalised).
    """

    # Some (non-doc) comments on how bond identifiers are numbered:
    # - The left virtual bond of the tensor `i` of the MPS has ID `i`.
    # - The right virtual bond of the tensor `i` of the MPS has ID `i+1`.
    # - The physical bond of the tensor `i` has ID `i+len(tensors)`.
    def __init__(
        self,
        libhandle: CuTensorNetHandle,
        qubits: list[Qubit],
        chi: Optional[int] = None,
        truncation_fidelity: Optional[float] = None,
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
            float_precision: The floating point precision used in tensor calculations;
                choose from ``numpy`` types: ``np.float64`` or ``np.float32``.
                Complex numbers are represented using two of such
                ``float`` numbers. Default is ``np.float64``.

        Raises:
            ValueError: If less than two qubits are provided.
            ValueError: If both ``chi`` and ``truncation_fidelity`` are fixed.
            ValueError: If the value of ``chi`` is set below 2.
            ValueError: If the value of ``truncation_fidelity`` is not in [0,1].
        """
        if chi is not None and truncation_fidelity is not None:
            raise ValueError("Cannot fix both chi and truncation_fidelity.")
        if chi is None:
            chi = max(2 ** (len(qubits) // 2), 2)
        if truncation_fidelity is None:
            truncation_fidelity = 1

        if chi < 2:
            raise ValueError("The max virtual bond dim (chi) must be >= 2.")
        if truncation_fidelity < 0 or truncation_fidelity > 1:
            raise ValueError("Provide a value of truncation_fidelity in [0,1].")

        if float_precision is None or float_precision == np.float64:  # Double precision
            self._real_t = np.float64  # type: ignore
            self._complex_t = np.complex128  # type: ignore
            self._atol = 1e-12
        elif float_precision == np.float32:  # Single precision
            self._real_t = np.float32  # type: ignore
            self._complex_t = np.complex64  # type: ignore
            self._atol = 1e-4
        else:
            allowed_precisions = [np.float64, np.float32]
            raise TypeError(
                f"Value of float_precision must be in {allowed_precisions}."
            )

        self._stream: cp.cuda.Stream = cp.cuda.get_current_stream()
        self._lib = libhandle
        # Make sure CuPy uses the specified device
        cp.cuda.Device(libhandle.device_id).use()

        #######################################
        # Initialise the MPS with a |0> state #
        #######################################

        self.chi = chi
        self.truncation_fidelity = truncation_fidelity
        self.fidelity = 1.0

        n_tensors = len(qubits)
        if n_tensors == 0:  # There's no initialisation to be done
            return None
        elif n_tensors == 1:
            raise ValueError("Please, provide at least two qubits.")

        self.qubit_position = {q: i for i, q in enumerate(qubits)}

        # Create the first and last tensors (these have one fewer bond)
        lr_shape = (1, 2)  # Initial virtual bond dim is 1; physical is 2
        l_tensor = cp.empty(lr_shape, dtype=self._complex_t)
        r_tensor = cp.empty(lr_shape, dtype=self._complex_t)
        # Initialise each tensor to ket 0
        l_tensor[0][0] = 1
        l_tensor[0][1] = 0
        r_tensor[0][0] = 1
        r_tensor[0][1] = 0

        # Create the list of tensors
        self.tensors = []
        # Append the leftmost tensor
        self.tensors.append(Tensor(l_tensor, [1, n_tensors]))

        # Append each of the tensors in between
        m_shape = (1, 1, 2)  # Two virtual bonds (dim=1) and one physical
        for i in range(1, n_tensors - 1):
            m_tensor = cp.empty(m_shape, dtype=self._complex_t)
            # Initialise the tensor to ket 0
            m_tensor[0][0][0] = 1
            m_tensor[0][0][1] = 0
            self.tensors.append(Tensor(m_tensor, [i, i + 1, i + n_tensors]))

        # Append the rightmost tensor
        self.tensors.append(Tensor(r_tensor, [n_tensors - 1, 2 * n_tensors - 1]))

    def is_valid(self) -> bool:
        """Verify that the MPS object is valid.

        Specifically, verify that the MPS does not exceed the dimension limit ``chi`` of
        the virtual bonds, that physical bonds have dimension 2 and that
        the virtual bonds are connected in a line.

        Returns:
            False if a violation was detected or True otherwise.
        """
        self._flush()

        chi_ok = all(
            all(dim <= self.chi for dim in self.get_virtual_dimensions(pos))
            for pos in range(len(self))
        )
        phys_ok = all(self.get_physical_dimension(pos) == 2 for pos in range(len(self)))
        shape_ok = all(
            len(tensor.data.shape) == len(tensor.bonds) and len(tensor.bonds) <= 3
            for tensor in self.tensors
        )

        # Check the leftmost tensor
        v_bonds_ok = self.get_virtual_bonds(0)[0] == 1
        # Check the middle tensors
        for i in range(1, len(self) - 1):
            v_bonds_ok = v_bonds_ok and (
                self.get_virtual_bonds(i)[0] == i
                and self.get_virtual_bonds(i)[1] == i + 1
            )
        # Check the rightmost tensor
        i = len(self) - 1
        v_bonds_ok = v_bonds_ok and self.get_virtual_bonds(i)[0] == i

        return chi_ok and phys_ok and shape_ok and v_bonds_ok

    def apply_gate(self, gate: Command) -> MPS:
        """Apply the gate to the MPS.

        Note:
            Only one-qubit gates and two-qubit gates are supported. Two-qubit
            gates must act on adjacent qubits.

        Args:
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
            RuntimeError: If gate acts on more than 2 qubits or acts on non-adjacent
                qubits.
            RuntimeError: If physical bond dimension where gate is applied is not 2.
        """
        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        positions = [self.qubit_position[q] for q in gate.qubits]
        if any(self.get_physical_dimension(pos) != 2 for pos in positions):
            raise RuntimeError(
                "Gates can only be applied to tensors with physical"
                + " bond dimension of 2."
            )

        if len(positions) == 1:
            self._apply_1q_gate(positions[0], gate.op)
            # NOTE: if the tensor was in canonical form, it remains being so,
            #   since it is guaranteed that the gate is unitary.

        elif len(positions) == 2:
            dist = positions[1] - positions[0]
            # We explicitly allow both dist==1 or dist==-1 so that non-symmetric
            # gates such as CX can use the same Op for the two ways it can be in.
            if dist not in [1, -1]:
                raise RuntimeError(
                    "Gates must be applied to adjacent qubits! "
                    + f"This is not satisfied by {gate}."
                )
            self._apply_2q_gate((positions[0], positions[1]), gate.op)
            # The tensors will in general no longer be in canonical form.
            self.tensors[positions[0]].canonical_form = None
            self.tensors[positions[1]].canonical_form = None

        else:
            raise RuntimeError(
                "Gates must act on only 1 or 2 qubits! "
                + f"This is not satisfied by {gate}."
            )
        return self

    def canonicalise(self, l_pos: int, r_pos: int) -> None:
        """Canonicalises the MPS object.

        Applies the necessary gauge transformations so that all MPS tensors
        to the left of position ``l_pos`` are in left orthogonal form and
        all MPS tensors to the right of ``r_pos`` in right orthogonal form.

        Args:
            l_pos: The position of the leftmost tensor that is not to be
                canonicalised.
            r_pos: The position of the rightmost tensor that is not to be
                canonicalised.
        """
        for pos in range(l_pos):
            self.canonicalise_tensor(pos, form=DirectionMPS.LEFT)
        for pos in reversed(range(r_pos + 1, len(self))):
            self.canonicalise_tensor(pos, form=DirectionMPS.RIGHT)

    def canonicalise_tensor(self, pos: int, form: DirectionMPS) -> None:
        """Canonicalises a tensor from an MPS object.

        Applies the necessary gauge transformations so that the tensor at
        position ``pos`` in the MPS is in the orthogonal form dictated by
        ``form``.

        Args:
            position: The position of the tensor to be canonicalised.
            form: LEFT form means that its conjugate transpose is its inverse if
                connected to its left bond and physical bond. Similarly for RIGHT.

        Raises:
            ValueError: If ``form`` is not a value in ``DirectionMPS``.
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
            RuntimeError: If position and form don't match.
        """
        if form == self.tensors[pos].canonical_form:
            # Tensor already in canonical form, nothing needs to be done
            return None

        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        if form == DirectionMPS.LEFT:
            next_pos = pos + 1
            gauge_bond = pos + 1
            gauge_T_index = 0
            gauge_Q_index = -2
        elif form == DirectionMPS.RIGHT:
            next_pos = pos - 1
            gauge_bond = pos
            gauge_T_index = -2
            gauge_Q_index = 0
        else:
            raise ValueError("Argument form must be a value in DirectionMPS.")

        # Gather the details from the MPS tensor at this position
        T = self.tensors[pos]
        T_d = T.data
        p_bond = self.get_physical_bond(pos)
        p_dim = self.get_physical_dimension(pos)
        v_bonds = self.get_virtual_bonds(pos)
        v_dims = self.get_virtual_dimensions(pos)

        # Decide the shape of the Q and R tensors
        if pos == 0:
            if form == DirectionMPS.RIGHT:
                raise RuntimeError(
                    "The leftmost tensor cannot be in right orthogonal form."
                )
            new_dim = min(p_dim, v_dims[0])
            Q_bonds = [-1, p_bond]
            Q_shape = [new_dim, p_dim]
            R_bonds = [-1, v_bonds[0]]
            R_shape = [new_dim, v_dims[0]]

        elif pos == len(self) - 1:
            if form == DirectionMPS.LEFT:
                raise RuntimeError(
                    "The rightmost tensor cannot be in left orthogonal form."
                )
            new_dim = min(p_dim, v_dims[0])
            Q_bonds = [-1, p_bond]
            Q_shape = [new_dim, p_dim]
            R_bonds = [v_bonds[0], -1]
            R_shape = [v_dims[0], new_dim]

        else:
            if form == DirectionMPS.LEFT:
                new_dim = min(v_dims[0] * p_dim, v_dims[1])
                Q_bonds = [v_bonds[0], -1, p_bond]
                Q_shape = [v_dims[0], new_dim, p_dim]
                R_bonds = [-1, v_bonds[1]]
                R_shape = [new_dim, v_dims[1]]
            elif form == DirectionMPS.RIGHT:
                new_dim = min(v_dims[1] * p_dim, v_dims[0])
                Q_bonds = [-1, v_bonds[1], p_bond]
                Q_shape = [new_dim, v_dims[1], p_dim]
                R_bonds = [v_bonds[0], -1]
                R_shape = [v_dims[0], new_dim]

        # Create template for the Q and R tensors
        Q_d = cp.empty(Q_shape, dtype=self._complex_t)
        Q = Tensor(Q_d, Q_bonds)
        R_d = cp.empty(R_shape, dtype=self._complex_t)
        R = Tensor(R_d, R_bonds)

        # Create tensor descriptors
        T_desc = T.get_tensor_descriptor(self._lib)
        Q_desc = Q.get_tensor_descriptor(self._lib)
        R_desc = R.get_tensor_descriptor(self._lib)

        # Apply QR decomposition
        cutn.tensor_qr(
            self._lib.handle,
            T_desc,
            T_d.data.ptr,
            Q_desc,
            Q_d.data.ptr,
            R_desc,
            R_d.data.ptr,
            0,  # 0 means let cuQuantum manage mem itself
            self._stream.ptr,  # type: ignore
        )
        self._stream.synchronize()  # type: ignore

        # Contract R into the tensor of the next position
        Tnext_bonds = list(self.tensors[next_pos].bonds)
        Tnext_bonds[gauge_T_index] = -1
        Tnext_d = cq.contract(
            R_d,
            R.bonds,
            self.tensors[next_pos].data,
            self.tensors[next_pos].bonds,
            Tnext_bonds,
        )
        # Reassign virtual bond ID
        Tnext_bonds[gauge_T_index] = gauge_bond
        Q_bonds[gauge_Q_index] = gauge_bond
        # Update self.tensors
        self.tensors[pos].data = Q_d
        self.tensors[pos].bonds = Q_bonds
        self.tensors[pos].canonical_form = form
        self.tensors[next_pos].data = Tnext_d
        self.tensors[next_pos].bonds = Tnext_bonds
        self.tensors[next_pos].canonical_form = None

        # Destroy descriptors
        cutn.destroy_tensor_descriptor(T_desc)
        cutn.destroy_tensor_descriptor(Q_desc)
        cutn.destroy_tensor_descriptor(R_desc)

    def vdot(self, other: MPS) -> complex:
        """Obtain the inner product of the two MPS: ``<self|other>``.

        It can be used to compute the squared norm of an MPS ``mps`` as
        ``mps.vdot(mps)``. The tensors within the MPS are not modified.

        Note:
            The state that is conjugated is ``self``.

        Args:
            other: The other MPS to compare against.

        Raises:
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
            RuntimeError: If number of tensors, dimensions or positions do not match.

        Return:
            The resulting complex number.
        """
        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        if len(self) != len(other):
            raise RuntimeError("Number of tensors do not match.")
        for i in range(len(self)):
            if self.get_physical_dimension(i) != other.get_physical_dimension(i):
                raise RuntimeError(
                    f"Physical bond dimension at position {i} do not match."
                )
        if self.qubit_position != other.qubit_position:
            raise RuntimeError(
                "The qubit labels or their position on the MPS do not match."
            )

        self._flush()
        other._flush()

        # The two MPS will be contracted from left to right, storing the
        # ``partial_result`` tensor.
        partial_result = cq.contract(
            self.tensors[0].data.conj(), [-1, 0], other.tensors[0].data, [1, 0], [-1, 1]
        )
        # Contract all tensors in the middle
        for pos in range(1, len(self) - 1):
            partial_result = cq.contract(
                partial_result,
                [-pos, pos],
                self.tensors[pos].data.conj(),
                [-pos, -(pos + 1), 0],
                other.tensors[pos].data,
                [pos, pos + 1, 0],
                [-(pos + 1), pos + 1],
            )
        # Finally, contract the last tensor
        result = cq.contract(
            partial_result,
            [-(len(self) - 1), len(self) - 1],
            self.tensors[-1].data.conj(),
            [-(len(self) - 1), 0],
            other.tensors[-1].data,
            [len(self) - 1, 0],
            [],  # No open bonds remain; this is just a scalar
        )

        return complex(result)

    def get_virtual_bonds(self, position: int) -> list[Bond]:
        """Returns the virtual bonds unique identifiers of tensor ``tensors[position]``.

        Args:
            position: A position in the MPS.

        Returns:
            A list with the ID of the virtual bonds of the tensor
            in order from left to right.
            If ``position`` is the first or last in the MPS, then the list
            will only contain the corresponding virtual bond.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """
        if position < 0 or position >= len(self):
            raise RuntimeError(f"Position {position} is out of bounds.")
        elif position == 0:
            v_bonds = [position + 1]
        elif position == len(self) - 1:
            v_bonds = [position]
        else:
            v_bonds = [position, position + 1]

        assert all(vb in self.tensors[position].bonds for vb in v_bonds)
        return v_bonds

    def get_virtual_dimensions(self, position: int) -> list[int]:
        """Returns the virtual bonds dimension of the tensor ``tensors[position]``.

        Args:
            position: A position in the MPS.

        Returns:
            A list with the dimensions of the virtual bonds of the
            tensor in order from left to right.
            If ``position`` is the first or last in the MPS, then the list
            will only contain the corresponding virtual bond.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """
        return [
            self.tensors[position].get_bond_dimension(bond)
            for bond in self.get_virtual_bonds(position)
        ]

    def get_physical_bond(self, position: int) -> Bond:
        """Returns the physical bond unique identifier of tensor ``tensors[position]``.

        Args
            position: A position in the MPS.

        Returns:
            The identifier of the physical bond.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """
        if position < 0 or position >= len(self):
            raise RuntimeError(f"Position {position} is out of bounds.")

        # By construction, the largest identifier is the physical one
        return max(self.tensors[position].bonds)

    def get_physical_dimension(self, position: int) -> int:
        """Returns the physical bond dimension of the tensor ``tensors[position]``.

        Args:
            position: A position in the MPS.

        Returns:
            The dimension of the physical bond.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """
        return self.tensors[position].get_bond_dimension(
            self.get_physical_bond(position)
        )

    def get_device_id(self) -> int:
        """
        Returns:
            The identifier of the device (GPU) where the tensors are stored.
        """
        return int(self.tensors[0].data.device)

    def update_libhandle(self, libhandle: CuTensorNetHandle) -> None:
        """Update the ``CuTensorNetHandle`` used by this ``MPS`` object. Multiple
        objects may use the same handle.

        Args:
            libhandle: The new cuTensorNet library handle.

        Raises:
            RuntimeError: If the device (GPU) where ``libhandle`` was initialised
                does not match the one where the tensors of the MPS are stored.
        """
        if libhandle.device_id != self.get_device_id():
            raise RuntimeError(
                "Device of libhandle is not the one where the MPS is stored.",
                f"{libhandle.device_id} != {self.get_device_id()}",
            )
        self._lib = libhandle

    def copy(self) -> MPS:
        """
        Returns:
            A deep copy of the MPS on the same device.
        """
        self._flush()

        # Create a dummy object
        new_mps = MPS(self._lib, qubits=[])
        # Copy all data
        new_mps.chi = self.chi
        new_mps.truncation_fidelity = self.truncation_fidelity
        new_mps.fidelity = self.fidelity
        new_mps.tensors = [t.copy() for t in self.tensors]
        new_mps.qubit_position = self.qubit_position.copy()
        new_mps._complex_t = self._complex_t
        new_mps._real_t = self._real_t

        return new_mps

    def __len__(self) -> int:
        """
        Returns:
            The number of tensors in the MPS.
        """
        return len(self.tensors)

    def _apply_1q_gate(self, position: int, gate: Op) -> MPS:
        raise NotImplementedError(
            "MPS is a base class with no contraction algorithm implemented."
            + " You must use a subclass of MPS, such as MPSxGate or MPSxMPO."
        )

    def _apply_2q_gate(self, positions: tuple[int, int], gate: Op) -> MPS:
        raise NotImplementedError(
            "MPS is a base class with no contraction algorithm implemented."
            + " You must use a subclass of MPS, such as MPSxGate or MPSxMPO."
        )

    def _flush(self) -> None:
        # Does nothing in the general MPS case; but children classes with batched
        # gate contraction will redefine this method so that the last batch of
        # gates is applied.
        return None
