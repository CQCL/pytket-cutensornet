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

import cupy as cp  # type: ignore
import numpy as np  # type: ignore
import cuquantum as cq  # type: ignore
import cuquantum.cutensornet as cutn  # type: ignore

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

        return cutn.create_tensor_descriptor(  # type: ignore
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
        return self.data.strides  # type: ignore

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
        return int(self.data.shape[self.bonds.index(bond)])

    def copy(self) -> Tensor:
        """
        Returns:
            A deep copy of the Tensor.
        """
        return Tensor(self.data.copy(), self.bonds.copy())
