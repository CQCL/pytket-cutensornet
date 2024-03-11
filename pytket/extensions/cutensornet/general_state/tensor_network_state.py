from __future__ import annotations
import logging
import math
from typing import Union, Optional, Any
import warnings

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
import numpy as np
from sympy import Expr  # type: ignore
from numpy.typing import NDArray
from pytket.circuit import Circuit  # type: ignore
from pytket.extensions.cutensornet.general import set_logger
from pytket.extensions.cutensornet.structured_state import CuTensorNetHandle
from pytket.utils.operators import QubitPauliOperator

try:
    import cuquantum as cq  # type: ignore
    from cuquantum import cutensornet as cutn  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cuquantum", ImportWarning)


class GeneralState:
    """Handles cuTensorNet tensor network state object."""

    def __init__(
        self,
        circuit: Circuit,
        libhandle: CuTensorNetHandle,
        loglevel: int = logging.INFO,
    ) -> None:
        """Constructs a tensor network state representation from a pytket circuit.

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
        self._circuit = circuit
        self._circuit.replace_implicit_wire_swaps()
        self._handle = libhandle.handle
        self._dev = libhandle.dev

        libhandle.print_device_properties(self._logger)

        num_qubits = circuit.n_qubits
        dim = 2  # We are always dealing with qubits, not qudits
        qubits_dims = (dim,) * num_qubits  # qubit size
        self._logger.debug(f"Converting a quantum circuit with {num_qubits} qubits.")
        data_type = cq.cudaDataType.CUDA_C_64F  # for now let that be hard-coded

        # These are only required when doing preparation and evaluation.
        self._stream = None
        self._scratch_space = None
        self._work_desc = None

        self._state = cutn.create_state(
            self._handle, cutn.StatePurity.PURE, num_qubits, qubits_dims, data_type
        )
        self._mutable_gates_map = {}
        self._gate_tensors = []
        for com in circuit.get_commands():
            gate_unitary = com.op.get_unitary().astype("complex128", copy=False)
            # Transpose is needed because of the way cuTN stores tensors.
            # See https://github.com/NVIDIA/cuQuantum/discussions/124
            # #discussioncomment-8683146 for details.
            self._gate_tensors.append(
                cp.asarray(gate_unitary)
                .T.astype(dtype="complex128", order="F")
                .reshape([2] * (2 * com.op.n_qubits), order="F")
            )
            gate_strides = 0  # Always 0?
            gate_qubit_indices = tuple(
                self._circuit.qubits.index(qb) for qb in com.qubits
            )
            gate_id = cutn.state_apply_tensor(
                self._handle,
                self._state,
                com.op.n_qubits,
                gate_qubit_indices,
                self._gate_tensors[-1].data.ptr,
                gate_strides,
                1,
                0,
                1,
            )
            if com.opgroup is not None:
                self._mutable_gates_map[com.opgroup] = gate_id

    @property
    def state(self) -> Any:
        """Returns tensor network state handle as Python :code:`int`."""
        return self._state

    def update_gates(self, gates_update_map: dict[str, NDArray]) -> None:
        """Updates gate unitaries in the tensor network state.

        Args:
            gates_update_map: Map from gate (Command) opgroup name to a corresponding
             gate unitary.

        Raises:
            ValueError: If a gate's unitary is of a wrong size.
        """
        for gate_label, unitary in gates_update_map.items():
            gate_id = self._mutable_gates_map[gate_label]
            gate_n_qubits = math.log2(unitary.shape[0])
            if not gate_n_qubits.is_integer():
                raise ValueError(
                    f"Gate {gate_label} unitary's number of rows is not a power of two."
                )
            gate_tensor = cp.asarray(unitary, dtype="complex128").reshape(
                [2] * (2 * int(gate_n_qubits)), order="F"
            )
            cutn.state_update_tensor(
                self._handle, self._state, gate_id, gate_tensor.data.ptr, 1
            )

    def configure(self, attributes: Optional[dict] = None) -> GeneralState:
        """Configures tensor network state for future contraction.

        Args:
            attributes: A dict of cuTensorNet State attributes and their values.

        Returns:
            Self (to allow for chaining with other methods).
        """
        if attributes is None:
            attributes = {"NUM_HYPER_SAMPLES": 8}
        attribute_values = [val for val in attributes.values()]
        attribute_lst = [
            getattr(cutn.StateAttribute, attr) for attr in attributes.keys()
        ]
        for attr, val in zip(attribute_lst, attribute_values):
            attr_dtype = cutn.state_get_attribute_dtype(attr)
            attr_arr = np.asarray(val, dtype=attr_dtype)
            cutn.state_configure(
                self._handle,
                self._state,
                attr,
                attr_arr.ctypes.data,
                attr_arr.dtype.itemsize,
            )
        return self

    def prepare(self, scratch_fraction: float = 0.5) -> GeneralState:
        """Prepare tensor network state for future contraction.

        Allocates workspace memory necessary for contraction.

        Raises:
            MemoryError: If there is insufficient workspace on GPU.

        Args:
            scratch_fraction: Fraction of free memory on GPU to allocate as scratch
             space.

        Returns:
            Self (to allow for chaining with other methods).
        """
        self._stream = (
            cp.cuda.Stream()
        )  # In current cuTN release it is unused (could be 0x0)
        free_mem = self._dev.mem_info[0]
        scratch_size = int(scratch_fraction * free_mem)
        self._scratch_space = cp.cuda.alloc(scratch_size)
        self._logger.debug(f"Allocated {scratch_size} bytes of scratch memory on GPU")
        self._work_desc = cutn.create_workspace_descriptor(self._handle)
        cutn.state_prepare(
            self._handle,
            self._state,
            scratch_size,
            self._work_desc,
            self._stream.ptr,
        )
        workspace_size_d = cutn.workspace_get_memory_size(
            self._handle,
            self._work_desc,
            cutn.WorksizePref.RECOMMENDED,
            cutn.Memspace.DEVICE,
            cutn.WorkspaceKind.SCRATCH,
        )

        if workspace_size_d <= scratch_size:
            cutn.workspace_set_memory(
                self._handle,
                self._work_desc,
                cutn.Memspace.DEVICE,
                cutn.WorkspaceKind.SCRATCH,
                self._scratch_space.ptr,
                workspace_size_d,
            )
            self._logger.debug(
                f"Set {workspace_size_d} bytes of workspace memory out of the allocated"
                f" scratch space."
            )
            return self
        else:
            self.destroy()
            raise MemoryError(
                f"Insufficient workspace size on the GPU device {self._dev.id}"
            )

    def compute(self, on_host: bool = True) -> Union[cp.ndarray, np.ndarray]:
        """Evaluates state vector.

        Args:
            on_host: If :code:`True`, converts cupy :code:`ndarray` to numpy
                :code:`ndarray`, copying it to host device (CPU).

        Returns:
            Either a :code:`cupy.ndarray` on a GPU, or a :code:`numpy.ndarray` on a
            host device (CPU). Arrays are returned in a 1D shape.
        """
        state_vector = cp.empty(
            (2,) * self._circuit.n_qubits, dtype="complex128", order="F"
        )
        cutn.state_compute(  # type: ignore
            self._handle,
            self._state,
            self._work_desc,
            (state_vector.data.ptr,),
            self._stream.ptr,
        )
        self._stream.synchronize()  # type: ignore
        if on_host:
            return cp.asnumpy(state_vector.flatten())
        return state_vector.flatten()

    def destroy(self) -> None:
        """Destroys tensor network state."""
        if self._work_desc is not None:  # type: ignore
            cutn.destroy_workspace_descriptor(self._work_desc)
        cutn.destroy_state(self._state)
        del self._scratch_space


class GeneralOperator:
    """Handles tensor network operator."""

    def __init__(
        self,
        operator: QubitPauliOperator,
        num_qubits: int,
        libhandle: CuTensorNetHandle,
        loglevel: int = logging.INFO,
    ) -> None:
        """Constructs a tensor network operator.

        From a list of Pauli strings and corresponding coefficients.

        Args:
            operator: The Pauli operator.
            num_qubits: Number of qubits in a circuit for which operator is to be
             defined.
            libhandle: cuTensorNet handle.
            loglevel: Internal logger output level.
        """
        # Mind the transpose for Y (same argument as in GeneralState)
        self._pauli = {
            "X": cp.asarray([[0, 1], [1, 0]]).astype("complex128", order="F"),
            "Y": cp.asarray([[0, -1j], [1j, 0]]).T.astype("complex128", order="F"),
            "Z": cp.asarray([[1, 0], [0, -1]]).astype("complex128", order="F"),
            "I": cp.asarray([[1, 0], [0, 1]]).astype("complex128", order="F"),
        }
        self._logger = set_logger("GeneralOperator", loglevel)
        self._handle = libhandle.handle
        qubits_dims = (2,) * num_qubits
        data_type = cq.cudaDataType.CUDA_C_64F  # TODO: implement a config class?
        self._operator = cutn.create_network_operator(
            self._handle, num_qubits, qubits_dims, data_type
        )
        self._logger.debug("Adding operator terms:")
        for pauli_string, coeff in operator._dict.items():
            if isinstance(coeff, Expr):
                numeric_coeff = complex(coeff.evalf())  # type: ignore
            else:
                numeric_coeff = complex(coeff)  # type: ignore
            self._logger.debug(f"   {numeric_coeff}, {pauli_string}")
            num_pauli = len(pauli_string.map)
            num_modes = (1,) * num_pauli
            state_modes = tuple((qubit.index[0],) for qubit in pauli_string.map.keys())
            gate_data = tuple(
                self._pauli[pauli.name].data.ptr for pauli in pauli_string.map.values()
            )
            cutn.network_operator_append_product(
                self._handle,
                self._operator,
                numeric_coeff,
                num_pauli,
                num_modes,
                state_modes,
                0,
                gate_data,
            )

    @property
    def operator(self) -> Any:
        """Returns tensor network operator handle as Python :code:`int`."""
        return self._operator

    def destroy(self) -> None:
        """Destroys tensor network operator."""
        cutn.destroy_network_operator(self._operator)


class GeneralExpectationValue:
    """Handles a general tensor network operator expectation value."""

    def __init__(
        self,
        state: GeneralState,
        operator: GeneralOperator,
        libhandle: CuTensorNetHandle,
        loglevel: int = logging.INFO,
    ) -> None:
        """Initialises expectation value object and corresponding work space.

        Notes:
            State and Operator must have the same handle as ExpectationValue.
            State (and Operator?) need to exist during the whole lifetime of
             ExpectationValue.

        Args:
            state: General tensor network state.
            operator: General tensor network operator.
            libhandle: cuTensorNet handle.
            loglevel: Internal logger output level.

        Raises:
            MemoryError: If there is insufficient workspace size on a GPU device.
        """
        self._handle = libhandle.handle
        self._dev = libhandle.dev
        self._logger = set_logger("GeneralExpectationValue", loglevel)

        self._stream = None
        self._scratch_space = None
        self._work_desc = None

        self._expectation = cutn.create_expectation(
            self._handle, state._state, operator._operator
        )

    def configure(self, attributes: Optional[dict] = None) -> GeneralExpectationValue:
        """Configures expectation value for future contraction.

        Args:
            attributes: A map of cuTensorNet :code:`ExpectationAttribute` objects to
                their values.

        Note:
            Currently :code:`ExpectationAttribute` has only one attribute.

        Returns:
            Self (to allow for chaining with other methods).
        """
        if attributes is None:
            attributes = {"OPT_NUM_HYPER_SAMPLES": 8}
        attribute_values = [val for val in attributes.values()]
        attribute_lst = [
            getattr(cutn.ExpectationAttribute, attr) for attr in attributes.keys()
        ]
        for attr, val in zip(attribute_lst, attribute_values):
            attr_dtype = cutn.expectation_get_attribute_dtype(attr)
            attr_arr = np.asarray(val, dtype=attr_dtype)
            cutn.expectation_configure(
                self._handle,
                self._expectation,
                attr,
                attr_arr.ctypes.data,
                attr_arr.dtype.itemsize,
            )
        return self

    def prepare(self, scratch_fraction: float = 0.5) -> GeneralExpectationValue:
        """Prepare tensor network state for future contraction.

        Allocates workspace memory necessary for contraction.

        Raises:
            MemoryError: If there is insufficient space on the GPU device.

        Args:
            scratch_fraction: Fraction of free memory on GPU to allocate as scratch
             space.

        Returns:
            Self (to allow for chaining with other methods).
        """
        # TODO: need to figure out if this needs to be done explicitly at all
        self._stream = (
            cp.cuda.Stream()
        )  # In current cuTN release it is unused (could be 0x0)
        free_mem = self._dev.mem_info[0]
        scratch_size = int(scratch_fraction * free_mem)
        self._scratch_space = cp.cuda.alloc(scratch_size)
        self._logger.debug(f"Allocated {scratch_size} bytes of scratch memory on GPU")
        self._work_desc = cutn.create_workspace_descriptor(self._handle)
        cutn.expectation_prepare(
            self._handle,
            self._expectation,
            scratch_size,
            self._work_desc,
            self._stream.ptr,
        )
        workspace_size_d = cutn.workspace_get_memory_size(
            self._handle,
            self._work_desc,
            cutn.WorksizePref.RECOMMENDED,
            cutn.Memspace.DEVICE,
            cutn.WorkspaceKind.SCRATCH,
        )

        if workspace_size_d <= scratch_size:
            cutn.workspace_set_memory(
                self._handle,
                self._work_desc,
                cutn.Memspace.DEVICE,
                cutn.WorkspaceKind.SCRATCH,
                self._scratch_space.ptr,
                workspace_size_d,
            )
            self._logger.debug(
                f"Set {workspace_size_d} bytes of workspace memory out of the allocated"
                f" scratch space."
            )
            return self
        else:
            self.destroy()
            raise MemoryError(
                f"Insufficient workspace size on the GPU device {self._dev.id}"
            )

    def compute(self) -> tuple[complex, complex]:
        """Computes expectation value."""
        expectation_value = np.empty(1, dtype="complex128")
        state_norm = np.empty(1, dtype="complex128")
        cutn.expectation_compute(  # type: ignore
            self._handle,
            self._expectation,
            self._work_desc,
            expectation_value.ctypes.data,
            state_norm.ctypes.data,
            self._stream.ptr,
        )
        self._stream.synchronize()  # type: ignore
        return expectation_value.item(), state_norm.item()

    def destroy(self) -> None:
        """Destroys tensor network expectation value and workspace descriptor."""
        if self._work_desc is not None:  # type: ignore
            cutn.destroy_workspace_descriptor(self._work_desc)
        cutn.destroy_expectation(self._expectation)
        del self._scratch_space  # TODO is this the correct way?
