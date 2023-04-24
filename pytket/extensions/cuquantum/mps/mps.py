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
from __future__ import annotations

import cupy as cp
import numpy as np

import cuquantum as cq
import cuquantum.cutensornet as cutn
from pytket.circuit import Op

# An alias so that `intptr_t` from CuQuantum's API (which is not available in
# base python) has some meaningful type name.
Handle = int
# An alias for the type of the unique identifiers of each of the bonds
# of the MPS.
Bond = int


class Tensor:
    """Class for the management of tensors via CuPy and cuQuantum.
    It abstracts away some of the low-level API of cuQuantum.

    Attributes:
        data (cupy.ndarray): The entries of the tensor arranged in a CuPy ndarray.
        bonds (list[Bond]): A list of IDs for each bond, matching the same order
            as in ``self.data.shape`` (which provides the dimension of each).
    """

    def __init__(self, data: cp.ndarray, bonds: list[Bond]):
        """Standard initialisation.

        Args:
            data: The entries of the tensor arranged in a CuPy ndarray.
            bonds: A list of IDs for each bond, matching the same order
                as in ``self.data.shape`` (which provides the dimension of each).
        """
        self.data = data
        self.bonds = bonds

    def get_tensor_descriptor(self, libhandle: Handle) -> Handle:
        """Return the cuQuantum tensor descriptor.

        Note:
            The user is responsible of destroying the descriptor once
            not in use (see ``cuquantum.cutensornet.destroy_tensor_descriptor``).

        Args:
            libhandle: The cuQuantum library handle.

        Returns:
            The handle to the tensor descriptor.
        """
        if self.data.dtype == np.float32:
            cq_dtype = cq.cudaDataType.CUDA_R_32F
        elif self.data.dtype == np.float64:
            cq_dtype = cq.cudaDataType.CUDA_R_64F
        elif self.data.dtype == np.complex64:
            cq_dtype = cq.cudaDataType.CUDA_C_32F
        elif self.data.dtype == np.complex128:
            cq_dtype = cq.cudaDataType.CUDA_C_64F

        return cutn.create_tensor_descriptor(
            handle=libhandle,
            n_modes=len(self.data.shape),
            extents=self.data.shape,
            strides=self.get_cuquantum_strides(),
            modes=self.bonds,
            data_type=cq_dtype,
        )

    def get_cupy_strides(self) -> list[int]:
        """Return a list of the strides for each of the bonds; in the same
        order as in ``self.bonds``. Strides are in CuPy format (#bytes).

        Returns:
            List of strides. Strides are in CuPy format (#bytes).
        """
        return self.data.strides

    def get_cuquantum_strides(self) -> list[int]:
        """Return a list of the strides for each of the bonds; in the same
        order as in ``self.bonds``. Strides are cuQuantum format (#entries).

        Returns:
            List of strides. Strides are cuQuantum format (#entries).
        """
        return [stride // self.data.itemsize for stride in self.data.strides]

    def get_dimension_of(self, bond: Bond) -> int:
        """Given a bond ID, return that bond's dimension.

        Args:
            bond: The ID of the bond to be queried. If not in the tensor,
                an exception is raised.

        Returns:
            The dimension of the bond.
        """
        if bond not in self.bonds:
            raise RuntimeError(f"Bond {bond} not in tensor with bonds: {self.bonds}.")
        return self.data.shape[self.bonds.index(bond)]

    def copy(self) -> Tensor:
        """
        Returns:
            A deep copy of the Tensor.
        """
        return Tensor(self.data.copy(), self.bonds.copy())


class MPS:
    """Parent class for state-based simulation using Matrix Product State
    representation.

    Attributes:
        chi (int): The maximum allowed dimension of a virtual bond.
        tensors (list[Tensor]): A list of tensors in the MPS; tensors[0] is
            the leftmost and tensors[len(self)-1] is the rightmost; tensors[i]
            and tensors[i+1] are connected in the MPS via a bond.
        fidelity (float): An estimate of the fidelity, obtained by multiplying
            the fidelities after each contraction.
    """

    # Some (non-doc) comments on how bond identifiers are numbered:
    # - The left virtual bond of the tensor `i` of the MPS has ID `i`
    # - The right virtual bond of the tensor `i` of the MPS has ID `i+1`
    # - Every physical bond will have an ID >= len(tensors), so it is
    #   always going to be the largest bond ID of the bond list of a tensor.
    #   The ID of the physical bonds will change as we contract gates/MPO
    #   into the MPS, but the ID of the virtual bonds will remain unchanged.
    def __init__(self, n_tensors: int, chi: int, float_precision: str = "float64"):
        """Initialise an MPS on the computational state 0.

        Note:
            Use as ``with MPSxGate(..) as mps:`` so that cuQuantum
            handles are automatically destroyed at the end of execution.

        Args:
            n_tensors: The number of tensors in the MPS.
            chi: The maximum value the dimension of the virtual bonds
                is allowed to take. Higher implies better approximation but
                more computational resources.
            float_precision: Either 'float32' for single precision (32 bits per
                real number) or 'float64' for double precision (64 bits per real).
                Each complex number is represented using two of these real numbers.
                Default is 'float64'.
        """
        if n_tensors < 2 and n_tensors != -1:
            raise Exception("The n_tensors of the MPS must be >= 2.")
        if chi < 2:
            raise Exception("The max virtual bond dim (chi) must be >= 2.")

        allowed_precisions = ["float32", "float64"]
        if float_precision not in allowed_precisions:
            raise Exception(f"Value of float_precision must be in {allowed_precisions}")

        if float_precision == "float32":  # Single precision
            self._real_t = np.float32
            self._complex_t = np.complex64
        elif float_precision == "float64":  # Double precision
            self._real_t = np.float64
            self._complex_t = np.complex128

        #################################################
        # Create CuTensorNet library and memory handles #
        #################################################
        self._libhandle = cutn.create()
        self._stream = cp.cuda.Stream()
        dev = cp.cuda.Device()  # get current device

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
        def malloc(size, stream):
            return cp.cuda.runtime.mallocAsync(size, stream)

        def free(ptr, size, stream):
            cp.cuda.runtime.freeAsync(ptr, stream)

        memhandle = (malloc, free, "memory_handler")
        cutn.set_device_mem_handler(self._libhandle, memhandle)

        #######################################
        # Initialise the MPS with a |0> state #
        #######################################

        self.chi = chi
        self.fidelity = 1

        if n_tensors == -1:  # Special input to avoid initialisation
            return None

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

        # An internal counter of the Bond IDs already used
        self._largest_bond_id = 2 * n_tensors - 1

    def is_valid(self) -> bool:
        """Verify that the MPS does not exceed the dimension limit (chi) of
        the virtual bonds, that physical bonds have dimension <=2 and that
        the virtual bonds are connected in a line.

        Returns:
            False if a violation was detected.
        """
        chi_ok = all(
            all(dim <= self.chi for dim in self.get_virtual_dimensions(pos))
            for pos in range(len(self))
        )
        phys_ok = all(self.get_physical_dimension(pos) <= 2 for pos in range(len(self)))
        shape_ok = all(
            len(tensor.data.shape) == len(tensor.bonds) and len(tensor.bonds) <= 3
            for tensor in self.tensors
        )

        v_bonds_ok = True
        # Check the leftmost tensor
        v_bonds_ok = v_bonds_ok and self.get_virtual_bonds(0)[0] == 1
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

    def contract(self) -> complex:
        """Fully contract the MPS and return the scalar result.

        Note:
            Can only be applied on an MPS whose physical bonds all have
            dimension 1. If this is not the case, an exception is raised.

        Returns:
            The scalar result.
        """
        if not all(self.get_physical_dimension(pos) == 1 for pos in range(len(self))):
            raise RuntimeError(
                "The MPS still has some open physical bonds, so it cannot"
                + " be contracted to a scalar. Use ``apply_postselection``"
                + "where appropriate."
            )

        # The MPS will be contracted from left to right, storing the
        # ``partial_result`` tensor.
        partial_result = self.tensors[0].data.flatten()  # Shape now (2,)
        # Contract all tensors in the middle
        for pos in range(1, len(self) - 1):
            partial_result = cq.contract(
                partial_result,
                [pos],
                self.tensors[pos].data,
                self.tensors[pos].bonds,
                [pos + 1],  # The only open bond is the right one of tensors[pos]
            )
        # Finally, contract the last tensor
        result = cq.contract(
            partial_result,
            [len(self) - 1],
            self.tensors[-1].data,
            self.tensors[-1].bonds,
            [],  # No open bonds remain; this is just a scalar
        )

        return complex(result)

    def vdot(self, mps: MPS) -> complex:
        """Obtain the inner product of the two MPS.

        Note:
            If any of the MPS uses a lazy implementation, remember to call
            ``self.flush()`` and ``mps.flush()`` before ``self.vdot(mps)``
            so that all pending operations are taken into account.

        Args:
            mps: The other MPS to compare against.

        Return:
            The resulting complex number.

        Raise:
            RuntimeError: If number of tensors or dimensions do not match.
        """
        if len(self) != len(mps):
            raise RuntimeError("Number of tensors do not match.")
        for i in range(len(self)):
            if self.get_physical_dimension(i) != mps.get_physical_dimension(i):
                raise RuntimeError(
                    f"Physical bond dimension at position {i} do not match."
                )

        # The two MPS will be contracted from left to right, storing the
        # ``partial_result`` tensor.
        partial_result = cq.contract(
            self.tensors[0].data.conj(), [-1, 0], mps.tensors[0].data, [1, 0], [-1, 1]
        )
        # Contract all tensors in the middle
        for pos in range(1, len(self) - 1):
            partial_result = cq.contract(
                partial_result,
                [-pos, pos],
                self.tensors[pos].data.conj(),
                [-pos, -(pos + 1), 0],
                mps.tensors[pos].data,
                [pos, pos + 1, 0],
                [-(pos + 1), pos + 1],
            )
        # Finally, contract the last tensor
        result = cq.contract(
            partial_result,
            [-(len(self) - 1), len(self) - 1],
            self.tensors[-1].data.conj(),
            [-(len(self) - 1), 0],
            mps.tensors[-1].data,
            [len(self) - 1, 0],
            [],  # No open bonds remain; this is just a scalar
        )

        return complex(result)

    def canonicalise(self, l_pos: int, r_pos: int):
        """Apply the necessary gauge transformations so that all MPS tensors
        to the left of position ``l_pos`` are in left orthogonal form and
        all MPS tensors to the right of ``r_pos`` in right orthogonal form.

        Args:
            l_pos: The position of the leftmost tensor that is not to be
                canonicalised.
            r_pos: The position of the rightmost tensor that is not to be
            canonicalised.
        """
        for pos in range(l_pos):
            self.canonicalise_tensor(pos, form="left")
        for pos in reversed(range(r_pos + 1, len(self))):
            self.canonicalise_tensor(pos, form="right")

    def canonicalise_tensor(self, pos: int, form: str):
        """Apply the necessary gauge transformations so that the tensor at
        position ``pos`` in the MPS has is in the orthogonal form dictated by
        ``form``.

        Args:
            position: The position of the tensor to canonicalise.
            form: Either ``'left'`` or ``'right'``.
        """

        if form == "left":
            next_pos = pos + 1
            gauge_bond = pos + 1
            gauge_T_index = 0
            gauge_Q_index = -2
        elif form == "right":
            next_pos = pos - 1
            gauge_bond = pos
            gauge_T_index = -2
            gauge_Q_index = 0
        else:
            raise RuntimeError(
                f"Form {form} not recognised. Use either 'left' or 'right'."
            )

        # Gather the details from the MPS tensor at this position
        T = self.tensors[pos]
        T_d = T.data
        p_bond = self.get_physical_bond(pos)
        p_dim = self.get_physical_dimension(pos)
        v_bonds = self.get_virtual_bonds(pos)
        v_dims = self.get_virtual_dimensions(pos)

        # Decide the shape of the Q and R tensors
        if pos == 0:
            if form == "right":
                raise RuntimeError(
                    "The leftmost tensor cannot be in right orthogonal form."
                )
            new_dim = min(p_dim, v_dims[0])
            Q_bonds = [-1, p_bond]
            Q_shape = (new_dim, p_dim)
            R_bonds = [-1, v_bonds[0]]
            R_shape = (new_dim, v_dims[0])

        elif pos == len(self) - 1:
            if form == "left":
                raise RuntimeError(
                    "The rightmost tensor cannot be in left orthogonal form."
                )
            new_dim = min(p_dim, v_dims[0])
            Q_bonds = [-1, p_bond]
            Q_shape = (new_dim, p_dim)
            R_bonds = [v_bonds[0], -1]
            R_shape = (v_dims[0], new_dim)

        else:
            if form == "left":
                new_dim = min(v_dims[0] * p_dim, v_dims[1])
                Q_bonds = [v_bonds[0], -1, p_bond]
                Q_shape = (v_dims[0], new_dim, p_dim)
                R_bonds = [-1, v_bonds[1]]
                R_shape = (new_dim, v_dims[1])
            elif form == "right":
                new_dim = min(v_dims[1] * p_dim, v_dims[0])
                Q_bonds = [-1, v_bonds[1], p_bond]
                Q_shape = (new_dim, v_dims[1], p_dim)
                R_bonds = [v_bonds[0], -1]
                R_shape = (v_dims[0], new_dim)

        # Create template for the Q and R tensors
        Q_d = cp.empty(Q_shape, dtype=self._complex_t)
        Q = Tensor(Q_d, Q_bonds)
        R_d = cp.empty(R_shape, dtype=self._complex_t)
        R = Tensor(R_d, R_bonds)

        # Create tensor descriptors
        T_desc = T.get_tensor_descriptor(self._libhandle)
        Q_desc = Q.get_tensor_descriptor(self._libhandle)
        R_desc = R.get_tensor_descriptor(self._libhandle)

        # Apply QR decomposition
        cutn.tensor_qr(
            self._libhandle,
            T_desc,
            T_d.data.ptr,
            Q_desc,
            Q_d.data.ptr,
            R_desc,
            R_d.data.ptr,
            0,
            self._stream.ptr,  # 0 means let cuQuantum manage mem itself
        )
        self._stream.synchronize()

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
        self.tensors[next_pos].data = Tnext_d
        self.tensors[next_pos].bonds = Tnext_bonds

        # Destroy descriptors
        cutn.destroy_tensor_descriptor(T_desc)
        cutn.destroy_tensor_descriptor(Q_desc)
        cutn.destroy_tensor_descriptor(R_desc)

    def get_virtual_bonds(self, position: int) -> [Bond]:
        """Return the unique identifiers of the virtual bonds
        of the tensor ``tensors[position]``.

        Args:
            position: A position in the MPS.

        Returns:
            A list with the ID of the virtual bonds of the tensor
            in order from left to right.
            If ``position`` is the first or last in the MPS, then the list
            will only contain the corresponding virtual bond.
        """
        if position < 0 or position >= len(self):
            raise Exception(f"Position {position} is out of bounds.")
        elif position == 0:
            v_bonds = [position + 1]
        elif position == len(self) - 1:
            v_bonds = [position]
        else:
            v_bonds = [position, position + 1]

        assert all(vb in self.tensors[position].bonds for vb in v_bonds)
        return v_bonds

    def get_virtual_dimensions(self, position: int) -> [int]:
        """Return the dimension of the virtual bonds of the tensor
        ``tensors[position]``.

        Args:
            position: A position in the MPS.

        Returns:
            A list with the dimensions of the virtual bonds of the
            tensor in order from left to right.
            If ``position`` is the first or last in the MPS, then the list
            will only contain the corresponding virtual bond.
        """
        return [
            self.tensors[position].get_dimension_of(bond)
            for bond in self.get_virtual_bonds(position)
        ]

    def get_physical_bond(self, position: int) -> Bond:
        """Return the unique identifier of the physical bond of the tensor
        ``tensors[position]``.

        Args
            position: A position in the MPS.

        Returns:
            The identifier of the physical bond.
        """
        if position < 0 or position >= len(self):
            raise Exception(f"Position {position} is out of bounds.")

        # By construction, the largest identifier is the physical one
        return max(self.tensors[position].bonds)

    def get_physical_dimension(self, position: int) -> int:
        """Return the dimension of the physical bond of the tensor
        ``tensors[position]``.

        Args:
            position: A position in the MPS.

        Returns:
            The dimension of the physical bond.
        """
        return self.tensors[position].get_dimension_of(self.get_physical_bond(position))

    def copy(self) -> MPS:
        """
        Returns:
            A deep copy of the MPS
        """
        # Create object without initialising to |0> state
        new_mps = MPS(n_tensors=-1, chi=self.chi)
        # Copy all data
        new_mps.fidelity = self.fidelity
        new_mps.tensors = [t.copy() for t in self.tensors]
        new_mps._largest_bond_id = self._largest_bond_id
        new_mps._complex_t = self._complex_t
        new_mps._real_t = self._real_t

        return new_mps

    def __len__(self) -> int:
        """
        Returns:
            The number of tensors in the MPS.
        """
        return len(self.tensors)

    def __del__(self):
        cutn.destroy(self._libhandle)

    def __enter__(self) -> MPS:
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        del self

    def _new_bond_id(self) -> Bond:
        self._largest_bond_id += 1
        return self._largest_bond_id

    def apply_1q_gate(self, position: int, gate: Op):
        raise NotImplementedError(
            "MPS is a base class with no contraction algorithm implemented."
            + " You must use a subclass of MPS, such as MPSxGate or MPSxMPO."
        )

    def apply_2q_gate(self, positions: tuple[int, int], gate: Op):
        raise NotImplementedError(
            "MPS is a base class with no contraction algorithm implemented."
            + " You must use a subclass of MPS, such as MPSxGate or MPSxMPO."
        )

    def apply_postselection(self, position: int):
        raise NotImplementedError(
            "MPS is a base class with no contraction algorithm implemented."
            + " You must use a subclass of MPS, such as MPSxGate or MPSxMPO."
        )

    def flush(self):
        raise NotImplementedError(
            "Only implemented for MPS methods that use lazy contraction."
        )
