from __future__ import annotations
import logging
from typing import Union, Optional
import warnings

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
import numpy as np
from sympy import Expr  # type: ignore
from numpy.typing import NDArray
from pytket.circuit import Circuit
from pytket.extensions.cutensornet.general import CuTensorNetHandle, set_logger
from pytket.utils.operators import QubitPauliOperator

try:
    import cuquantum as cq  # type: ignore
    from cuquantum import cutensornet as cutn  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cuquantum", ImportWarning)


class GeneralState:
    """Wraps a cuTensorNet TN object for exact simulations via path optimisation"""

    def __init__(
        self,
        circuit: Circuit,
        libhandle: CuTensorNetHandle,
        loglevel: int = logging.INFO,
    ) -> None:
        """Constructs a tensor network state representation from a pytket circuit.

        Note:
            The tensor network is *not* contracted until the appropriate methods
            from this class are called.

        Note:
            Circuit should not contain boxes - only explicit gates with specific unitary
            matrix representation available in pytket.

        Note:
            If present, implicit wire swaps are replaced with explicit SWAP gates.

        Args:
            circuit: A pytket circuit to be converted to a tensor network.
            libhandle: cuTensorNet handle.
            loglevel: Internal logger output level.
        """
        self._logger = set_logger("GeneralState", loglevel)
        self._circuit = circuit.copy()
        # TODO: This is not strictly necessary; implicit SWAPs could be resolved by
        # qubit relabelling, but it's only worth doing so if there are clear signs
        # of inefficiency due to this.
        self._circuit.replace_implicit_wire_swaps()
        self._lib = libhandle

        libhandle.print_device_properties(self._logger)

        num_qubits = self._circuit.n_qubits
        dim = 2  # We are always dealing with qubits, not qudits
        qubits_dims = (dim,) * num_qubits  # qubit size
        self._logger.debug(f"Converting a quantum circuit with {num_qubits} qubits.")
        data_type = cq.cudaDataType.CUDA_C_64F  # for now let that be hard-coded

        self._state = cutn.create_state(
            self._lib.handle, cutn.StatePurity.PURE, num_qubits, qubits_dims, data_type
        )
        self._gate_tensors = []

        # Append all gates to the TN
        # TODO: we should add a check to verify that the commands are unitaries
        # (e.g. don't accept measurements). Potentially, measurements at the end of
        # the circuit can be ignored at the user's request.
        for com in self._circuit.get_commands():
            gate_unitary = com.op.get_unitary()
            self._gate_tensors.append(_formatted_tensor(gate_unitary, com.op.n_qubits))
            gate_qubit_indices = tuple(
                self._circuit.qubits.index(qb) for qb in com.qubits
            )

            cutn.state_apply_tensor_operator(
                handle=self._lib.handle,
                tensor_network_state=self._state,
                num_state_modes=com.op.n_qubits,
                state_modes=gate_qubit_indices,
                tensor_data=self._gate_tensors[-1].data.ptr,
                tensor_mode_strides=0,
                immutable=1,
                adjoint=0,
                unitary=1,
            )

    def get_statevector(
        self,
        attributes: Optional[dict] = None,
        scratch_fraction: float = 0.5,
        on_host: bool = True,
    ) -> Union[cp.ndarray, np.ndarray]:
        """Contracts the circuit and returns the final statevector.

        Args:
            attributes: A dict of cuTensorNet State attributes and their values.
            scratch_fraction: Fraction of free memory on GPU to allocate as scratch
                space. Defaults to `0.5`.
            on_host: If :code:`True`, converts cupy :code:`ndarray` to numpy
                :code:`ndarray`, copying it to host device (CPU).
        Raises:
            MemoryError: If there is insufficient workspace on GPU.
        Returns:
            Either a :code:`cupy.ndarray` on a GPU, or a :code:`numpy.ndarray` on a
            host device (CPU). Arrays are returned in a 1D shape.
        """

        ####################################
        # Configure the TN for contraction #
        ####################################
        if attributes is None:
            attributes = dict()
        if "NUM_HYPER_SAMPLES" not in attributes:
            attributes["NUM_HYPER_SAMPLES"] = 8
        attribute_pairs = [
            (getattr(cutn.StateAttribute, k), v) for k, v in attributes.items()
        ]

        for attr, val in attribute_pairs:
            attr_dtype = cutn.state_get_attribute_dtype(attr)
            attr_arr = np.asarray(val, dtype=attr_dtype)
            cutn.state_configure(
                self._lib.handle,
                self._state,
                attr,
                attr_arr.ctypes.data,
                attr_arr.dtype.itemsize,
            )

        try:
            ######################################
            # Allocate workspace for contraction #
            ######################################
            stream = cp.cuda.Stream()
            free_mem = self._lib.dev.mem_info[0]
            scratch_size = int(scratch_fraction * free_mem)
            scratch_space = cp.cuda.alloc(scratch_size)
            self._logger.debug(
                f"Allocated {scratch_size} bytes of scratch memory on GPU"
            )
            work_desc = cutn.create_workspace_descriptor(self._lib.handle)

            cutn.state_prepare(
                self._lib.handle,
                self._state,
                scratch_size,
                work_desc,
                stream.ptr,
            )
            workspace_size_d = cutn.workspace_get_memory_size(
                self._lib.handle,
                work_desc,
                cutn.WorksizePref.RECOMMENDED,
                cutn.Memspace.DEVICE,
                cutn.WorkspaceKind.SCRATCH,
            )

            if workspace_size_d <= scratch_size:
                cutn.workspace_set_memory(
                    self._lib.handle,
                    work_desc,
                    cutn.Memspace.DEVICE,
                    cutn.WorkspaceKind.SCRATCH,
                    scratch_space.ptr,
                    workspace_size_d,
                )
                self._logger.debug(
                    f"Set {workspace_size_d} bytes of workspace memory out of the"
                    f" allocated scratch space."
                )

            else:
                raise MemoryError(
                    f"Insufficient workspace size on the GPU device {self._lib.dev.id}"
                )

            ###################
            # Contract the TN #
            ###################
            state_vector = cp.empty(
                (2,) * self._circuit.n_qubits, dtype="complex128", order="F"
            )
            cutn.state_compute(
                self._lib.handle,
                self._state,
                work_desc,
                (state_vector.data.ptr,),
                stream.ptr,
            )
            stream.synchronize()
            sv = state_vector.flatten()
            if on_host:
                sv = cp.asnumpy(sv)
            # Apply the phase from the circuit
            sv *= np.exp(1j * np.pi * self._circuit.phase)
            return sv

        finally:
            cutn.destroy_workspace_descriptor(work_desc)  # type: ignore
            del scratch_space

    def expectation_value(
        self,
        operator: QubitPauliOperator,
        attributes: Optional[dict] = None,
        scratch_fraction: float = 0.5,
    ) -> complex:
        """Calculates the expectation value of the given operator.

        Args:
            operator: The operator whose expectation value is to be measured.
            attributes: A dict of cuTensorNet Expectation attributes and their values.
            scratch_fraction: Fraction of free memory on GPU to allocate as scratch
                space. Defaults to `0.5`.

        Raises:
            ValueError: If the operator acts on qubits not present in the circuit.

        Returns:
            The expectation value.
        """

        ############################################
        # Generate the cuTensorNet operator object #
        ############################################
        pauli_tensors = {
            "X": _formatted_tensor(np.asarray([[0, 1], [1, 0]]), 1),
            "Y": _formatted_tensor(np.asarray([[0, -1j], [1j, 0]]), 1),
            "Z": _formatted_tensor(np.asarray([[1, 0], [0, -1]]), 1),
            "I": _formatted_tensor(np.asarray([[1, 0], [0, 1]]), 1),
        }
        num_qubits = self._circuit.n_qubits
        qubits_dims = (2,) * num_qubits
        data_type = cq.cudaDataType.CUDA_C_64F

        tn_operator = cutn.create_network_operator(
            self._lib.handle, num_qubits, qubits_dims, data_type
        )

        self._logger.debug("Adding operator terms:")
        for pauli_string, coeff in operator._dict.items():
            if isinstance(coeff, Expr):
                numeric_coeff = complex(coeff.evalf())  # type: ignore
            else:
                numeric_coeff = complex(coeff)  # type: ignore
            self._logger.debug(f"   {numeric_coeff}, {pauli_string}")

            # Raise an error if the operator acts on qubits that are not in the circuit
            if any(q not in self._circuit.qubits for q in pauli_string.map.keys()):
                raise ValueError(
                    f"The operator is acting on qubits {pauli_string.map.keys()}, "
                    "but some of these are not present in the circuit, whose set of "
                    f"qubits is: {self._circuit.qubits}."
                )

            # Obtain the tensors corresponding to this operator
            qubit_pauli_map = {
                q: pauli_tensors[pauli.name] for q, pauli in pauli_string.map.items()
            }

            num_pauli = len(qubit_pauli_map)
            num_modes = (1,) * num_pauli
            state_modes = tuple(
                (self._circuit.qubits.index(qb),) for qb in qubit_pauli_map.keys()
            )
            gate_data = tuple(tensor.data.ptr for tensor in qubit_pauli_map.values())

            cutn.network_operator_append_product(
                handle=self._lib.handle,
                tensor_network_operator=tn_operator,
                coefficient=numeric_coeff,
                num_tensors=num_pauli,
                num_state_modes=num_modes,
                state_modes=state_modes,
                tensor_mode_strides=0,
                tensor_data=gate_data,
            )

        ######################################################
        # Configure the cuTensorNet expectation value object #
        ######################################################
        expectation = cutn.create_expectation(
            self._lib.handle, self._state, tn_operator
        )

        if attributes is None:
            attributes = dict()
        if "OPT_NUM_HYPER_SAMPLES" not in attributes:
            attributes["OPT_NUM_HYPER_SAMPLES"] = 8
        attribute_pairs = [
            (getattr(cutn.ExpectationAttribute, k), v) for k, v in attributes.items()
        ]

        for attr, val in attribute_pairs:
            attr_dtype = cutn.expectation_get_attribute_dtype(attr)
            attr_arr = np.asarray(val, dtype=attr_dtype)
            cutn.expectation_configure(
                self._lib.handle,
                expectation,
                attr,
                attr_arr.ctypes.data,
                attr_arr.dtype.itemsize,
            )

        try:
            ######################################
            # Allocate workspace for contraction #
            ######################################
            stream = cp.cuda.Stream()
            free_mem = self._lib.dev.mem_info[0]
            scratch_size = int(scratch_fraction * free_mem)
            scratch_space = cp.cuda.alloc(scratch_size)

            self._logger.debug(
                f"Allocated {scratch_size} bytes of scratch memory on GPU"
            )
            work_desc = cutn.create_workspace_descriptor(self._lib.handle)
            cutn.expectation_prepare(
                self._lib.handle,
                expectation,
                scratch_size,
                work_desc,
                stream.ptr,
            )
            workspace_size_d = cutn.workspace_get_memory_size(
                self._lib.handle,
                work_desc,
                cutn.WorksizePref.RECOMMENDED,
                cutn.Memspace.DEVICE,
                cutn.WorkspaceKind.SCRATCH,
            )

            if workspace_size_d <= scratch_size:
                cutn.workspace_set_memory(
                    self._lib.handle,
                    work_desc,
                    cutn.Memspace.DEVICE,
                    cutn.WorkspaceKind.SCRATCH,
                    scratch_space.ptr,
                    workspace_size_d,
                )
                self._logger.debug(
                    f"Set {workspace_size_d} bytes of workspace memory out of the"
                    f" allocated scratch space."
                )
            else:
                raise MemoryError(
                    f"Insufficient workspace size on the GPU device {self._lib.dev.id}"
                )

            #################################
            # Compute the expectation value #
            #################################
            expectation_value = np.empty(1, dtype="complex128")
            state_norm = np.empty(1, dtype="complex128")
            cutn.expectation_compute(
                self._lib.handle,
                expectation,
                work_desc,
                expectation_value.ctypes.data,
                state_norm.ctypes.data,
                stream.ptr,
            )
            stream.synchronize()

            # Note: we can also return `state_norm.item()`, but this should be 1 since
            # we are always running unitary circuits
            assert np.isclose(state_norm.item(), 1.0)

            return expectation_value.item()  # type: ignore

        finally:
            #####################################################
            # Destroy the Operator and ExpectationValue objects #
            #####################################################
            cutn.destroy_workspace_descriptor(work_desc)  # type: ignore
            cutn.destroy_expectation(expectation)
            cutn.destroy_network_operator(tn_operator)
            del scratch_space

    def destroy(self) -> None:
        """Destroys tensor network state."""
        cutn.destroy_state(self._state)


def _formatted_tensor(matrix: NDArray, n_qubits: int) -> cp.ndarray:
    """Convert a matrix to the tensor format accepted by NVIDIA's API."""

    # Transpose is needed because of the way cuTN stores tensors.
    # See https://github.com/NVIDIA/cuQuantum/discussions/124
    # #discussioncomment-8683146 for details.
    cupy_matrix = cp.asarray(matrix).T.astype(dtype="complex128", order="F")
    # We also need to reshape since a matrix only has 2 bonds, but for an
    # n-qubit gate we want 2^n bonds for input and another 2^n for output
    return cupy_matrix.reshape([2] * (2 * n_qubits), order="F")
