from __future__ import annotations  # type: ignore
from ast import Dict

from typing import Any, Optional

import cupy as cp  # type: ignore
import numpy as np  # type: ignore
import cuquantum as cq  # type: ignore
import cuquantum.cutensornet as cutn  # type: ignore

from pytket.circuit import Qubit  # type: ignore

from pytket.utils import QubitPauliOperator  # type: ignore
from pytket.pauli import Pauli  # type: ignore

from typing import Dict, List, Optional, Tuple, Union

from numpy.typing import NDArray

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


class QubitPauliMPO:

    """A Class for generating an matrix product operator (MPO) from a QubitPauliOperator.

    Attributes:
        qpo: The QubitPauliOperator to be converted to an MPO.
        physical_indices: A dictionary mapping each qubit to a tuple of integers (i,-i)
            which are the open indices in the bra and ket networks respectively.
        cuquantum_interleaved: The input of the cuquantum.contract function
            in the interleaved format [NDArray, [bonds], NDArray, [bonds],...].
    """

    def __init__(
        self,
        qpo: QubitPauliOperator,
        physical_indices: Dict[Qubit, tuple[int, int]],
        dummy_ind_start: int,
        float_precision=None,
    ) -> None:
        """Initialise the QubitPauliMPO.

        Args:
            qpo: The QubitPauliOperator to be converted to an MPO.
            physical_indices: A dictionary mapping each qubit to a tuple of integers (i,-i)
                which are the open indices in the bra and ket networks respectively.
            dummy_ind_start: The index of the first dummy index in the MPO.
            float_precision: The precision of the floating point numbers in the MPO.
        """

        self.qpo = qpo
        self.physical_indices_qubit = physical_indices
        self.dummy_ind_start = dummy_ind_start

        self._qubit_to_ind = {
            q: i for i, q in enumerate(self.physical_indices_qubit.keys())
        }

        self._n_sites = len(self.physical_indices_qubit)

        self.virtual_bonds = [
            i
            for i in range(
                self.dummy_ind_start, self.dummy_ind_start + self._n_sites - 1
            )
        ]

        self._complex_t = np.complex128

        # allowed_precisions = ["float32", "float64"]
        # if float_precision is None:
        #     float_precision = "float64"
        # elif float_precision not in allowed_precisions:
        #     raise Exception(f"Value of float_precision must be in {allowed_precisions}")

        # if float_precision == "float32":  # Single precision
        #     self._real_t = np.float32  # type: ignore
        #     self._complex_t = np.complex64  # type: ignore
        # elif float_precision == "float64":  # Double precision
        #     self._real_t = np.float64  # type: ignore
        #     self._complex_t = np.complex128  # type: ignore

        # #################################################
        # # Create CuTensorNet library and memory handles #
        # #################################################
        # self._libhandle = cutn.create()
        # self._stream = cp.cuda.Stream()
        # dev = cp.cuda.Device()  # get current device

        # if cp.cuda.runtime.runtimeGetVersion() < 11020:
        #     raise RuntimeError("Requires CUDA 11.2+.")
        # if not dev.attributes["MemoryPoolsSupported"]:
        #     raise RuntimeError("Device does not support CUDA Memory pools")

        # # Avoid shrinking the pool
        # mempool = cp.cuda.runtime.deviceGetDefaultMemPool(dev.id)
        # if int(cp.__version__.split(".")[0]) >= 10:
        #     # this API is exposed since CuPy v10
        #     cp.cuda.runtime.memPoolSetAttribute(
        #         mempool,
        #         cp.cuda.runtime.cudaMemPoolAttrReleaseThreshold,
        #         0xFFFFFFFFFFFFFFFF,  # = UINT64_MAX
        #     )

        # # A device memory handler lets CuTensorNet manage its own GPU memory
        # def malloc(size, stream):  # type: ignore
        #     return cp.cuda.runtime.mallocAsync(size, stream)

        # def free(ptr, size, stream):  # type: ignore
        #     cp.cuda.runtime.freeAsync(ptr, stream)

        # memhandle = (malloc, free, "memory_handler")
        # cutn.set_device_mem_handler(self._libhandle, memhandle)

        #######################################
        # Build MPO
        #######################################

        site_arrays = self.site_arrays(qpo)

        self.tensors = self.build_tensors(site_arrays)

        self._cuquantum_interleaved = self._get_cuquantum_interleaved()

    @property
    def cuquantum_interleaved(self) -> list:
        """Returns an interleaved format of the circuit tensor network."""
        return self._cuquantum_interleaved

    def build_tensors(self, site_arrays: NDArray) -> list[Tensor]:
        """Builds the tensors for the MPO
        Arranging the site array with its physical and virtual bonds
        for each qubit in the QubitPauliOperator

        Args:
            site_arrays: A list of site arrays for each
            qubit in the QubitPauliOperator

        Returns:
            A list of tensors for the MPO"""

        # map the physical indices on a  qubit to a new index to be used in list
        physical_indices_ind = {
            i: phys_ind
            for i, phys_ind in enumerate(self.physical_indices_qubit.values())
        }

        # Left tensor
        tensors = [
            Tensor(site_arrays[0], [*physical_indices_ind[0], self.virtual_bonds[0]])
        ]

        # Middle tensors
        for i in range(1, self._n_sites - 1):
            tensors.append(
                Tensor(
                    site_arrays[i],
                    [
                        *physical_indices_ind[i],
                        self.virtual_bonds[i - 1],
                        self.virtual_bonds[i],
                    ],
                )
            )

        # Right tensor
        n_virtual_bonds = self._n_sites - 1
        tensors.append(
            Tensor(
                site_arrays[self._n_sites - 1],
                [
                    *physical_indices_ind[self._n_sites - 1],
                    self.virtual_bonds[n_virtual_bonds - 1],
                ],
            )
        )

        return tensors

    def terms_matrices(
        self, qpo: QubitPauliOperator
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Returns the matrices for each qubit in the QubitPauliString
        in the QubitPauliOperator in a list of lists.
        The coefficients for each QubitPauliString in the
        QubitPauliOperator are returned in a list

        Args:
            qpo: QubitPauliOperator

        Returns:
            A list of lists of matrices for each QubitPauliString in the QubitPauliOperator
            and a list of coefficients for each QubitPauliString in the QubitPauliOperator
        """

        pauli_op_matrix = {
            Pauli.X: np.array([[0, 1], [1, 0]], dtype=self._complex_t),
            Pauli.Y: np.array([[0, -1j], [1j, 0]], dtype=self._complex_t),
            Pauli.Z: np.array([[1, 0], [0, -1]], dtype=self._complex_t),
            Pauli.I: np.array([[1, 0], [0, 1]], dtype=self._complex_t),
        }

        terms_site_matrices = []
        coeffs = []
        for term, coeff in qpo._dict.items():
            identity_sites = [np.identity(2) for i in range(self._n_sites)]
            coeffs.append(complex(coeff))
            term_site_matrices = identity_sites
            for qubit, pauli in term.map.items():
                identity_sites[self._qubit_to_ind[qubit]] = pauli_op_matrix[pauli]
            terms_site_matrices.append(term_site_matrices)

        return terms_site_matrices, coeffs

    def site_arrays(self, qpo: QubitPauliOperator) -> list[NDArray]:
        """Builds the site arrays for the MPO operator for each Qubit (site) in
        the QubitPauliOperator. Each qubit is represented by
        a 2 x 2 x nterms x nterms rank 4 array.
        The slice nterms x nterms is diagonal and represents the pauli or identity
        for that term in the direct product represention acting on that qubit.
        The first element in then list is multiplied by the coefficent of the term.

        Args:
            qpo: The QubitPauliOperator to build the site arrays for

        Returns:
            A list of site arrays for each qubit in the QubitPauliOperator"""

        n_terms = len(qpo.to_list())
        term_matrices, coeffs = self.terms_matrices(qpo)

        def extract_site(term_list, site):
            return [term[site] for term in term_list]

        site_matrices_list = [
            extract_site(term_matrices, site) for site in range(self._n_sites)
        ]

        site_tensors = []

        for n_site, matrix_list in enumerate(site_matrices_list):
            if n_site == 0 or n_site == self._n_sites - 1:
                shape = (2, 2, n_terms)
                site_tensor = np.zeros(shape, dtype=self._complex_t)
                for n_term, matrix in enumerate(matrix_list):
                    site_tensor[:, :, n_term] = matrix
            else:
                shape = (2, 2, n_terms, n_terms)
                site_tensor = np.zeros(shape, dtype=self._complex_t)
                for n_term, matrix in enumerate(matrix_list):
                    site_tensor[:, :, n_term, n_term] = matrix
            site_tensors.append(site_tensor)

        # multiply hamiltonian coefficents with the first site using broadcasting
        site_tensors[0] = site_tensors[0] * np.array([[coeffs]])
        return site_tensors

    def _get_cuquantum_interleaved(self) -> list:
        """Returns an cuquantum interleaved format
        of the circuit tensor network."""

        interleaved = []
        for tensor in self.tensors:
            interleaved += [tensor.data, tensor.bonds]
        return interleaved
