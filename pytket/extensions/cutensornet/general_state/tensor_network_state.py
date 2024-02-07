import logging
import math
from typing import Optional
import warnings

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
import numpy as np
from numpy.typing import NDArray
from pytket.circuit import Circuit  # type: ignore
from pytket.pauli import QubitPauliString  # type: ignore
from pytket.extensions.cutensornet.general import set_logger
from pytket.extensions.cutensornet.structured_state import CuTensorNetHandle

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

        Args:
            circuit: A pytket circuit to be converted to a tensor network.
            libhandle: cuTensorNet handle.
            loglevel: Internal logger output level.
        """
        self._logger = set_logger("GeneralState", loglevel)
        self._circuit = circuit
        self._handle = libhandle.handle

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
        for com in circuit.get_commands():
            gate_unitary = com.op.get_unitary().astype("complex128", copy=False)
            gate_tensor = cp.asarray(gate_unitary, dtype="complex128").reshape(
                [2] * (2 * com.op.n_qubits), order="F"
            )  # TODO: why column-major order?
            gate_strides = 0  # Always 0?
            gate_qubit_indices = [self._circuit.qubits.index(qb) for qb in com.qubits]
            gate_n_qubits = len(gate_qubit_indices)
            gate_qubit_indices_reversed = tuple(reversed(gate_qubit_indices))
            gate_id = cutn.state_apply_tensor(
                self._handle,
                self._state,
                gate_n_qubits,
                gate_qubit_indices_reversed,
                gate_tensor.data.ptr,
                gate_strides,
                1,
                0,
                1,
            )
            if com.opgroup is not None:
                self._mutable_gates_map[com.opgroup] = gate_id

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
                self._handle, self._state, gate_id, gate_tensor.data.ptr, 0, 1
            )

    def configure(self, attributes: Optional[dict] = None) -> None:
        """Configures tensor network state for future contraction.

        Args:
            attributes: A dict of cuTensorNet State attributes and their values.
        """
        if attributes is None:
            attributes = {"OPT_NUM_HYPER_SAMPLES": 8}
        attribute_values = [val for val in attributes.values()]
        attributes = [getattr(cutn.StateAttribute, attr) for attr in attributes.keys()]
        for attr, val in zip(attributes, attribute_values):
            attr_dtype = cutn.state_get_attribute_dtype(attr)
            attr_arr = np.asarray(val, dtype=attr_dtype)
            cutn.state_configure(
                self._handle,
                self._state,
                attr_dtype,
                attr_arr.ctypes.data,
                attr_arr.dtype.itemsize,
            )

    def prepare(self, scratch_fraction: float = 0.5) -> None:
        """Prepare tensor network state for future contraction.

        Allocates workspace memory necessary for contraction.

        Args:
            scratch_fraction: Fraction of free memory on GPU to allocate as scratch
             space.
        """
        self._stream = (
            cp.cuda.Stream()
        )  # In current cuTN release it is unused (could be 0x0)
        free_mem = self._handle.dev.mem_info[0]
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
        else:
            cutn.destroy_workspace_descriptor(self._work_desc)
            del self._scratch_space  # TODO: is it OK to do so?

    def compute(self) -> tuple:
        """Evaluates state vector."""
        state_vector = cp.asarray(pow(self._circuit.n_qubits, 2), dtype="complex128")
        cutn.state_compute(
            self._handle,
            self._state,
            self._work_desc,
            0,
            0,
            state_vector,
            self._stream.ptr,
        )
        self._stream.synchronize()
        return state_vector

    def destroy(self) -> None:
        """Destroys tensor network state."""
        cutn.destroy_state(self._state)


class GeneralOperator:
    """Handles tensor network operator."""

    PAULI = {
        "X": cp.array([[0, 1], [1, 0]], dtype="complex128", order="F"),
        "Y": cp.array([[0, -1j], [1j, 0]], dtype="complex128", order="F"),
        "Z": cp.array([[1, 0], [0, -1]], dtype="complex128", order="F"),
        "I": cp.array([[1, 0], [0, 1]], dtype="complex128", order="F"),
    }

    def __init__(
        self,
        operator: list[tuple[complex, QubitPauliString]],
        num_qubits: int,
        libhandle: CuTensorNetHandle,
        loglevel: int = logging.INFO,
    ) -> None:
        """Constructs a tensor network operator.

        From a list of Pauli strings and corresponding coefficients.

        Args:
            operator: List of tuples, containing a Pauli string and a corresponding
             numeric coefficient.
            num_qubits: Number of qubits in a circuit for which operator is to be
             defined.
            libhandle: cuTensorNet handle.
            loglevel: Internal logger output level.
        """
        self._logger = set_logger("GeneralOperator", loglevel)
        self._handle = libhandle.handle
        qubits_dims = (2,) * num_qubits
        data_type = cq.cudaDataType.CUDA_C_64F  # TODO: implement a config class?
        self._operator = cutn.create_network_operator(
            self._handle, num_qubits, qubits_dims, data_type
        )
        self._logger.debug("Adding operator terms:")
        for coeff, pauli_string in operator:
            self.append_pauli_string(pauli_string, coeff)

    def append_pauli_string(
        self, pauli_string: QubitPauliString, coeff: complex
    ) -> None:
        """Appends a Pauli string to a tensor network operator.

        Args:
            pauli_string: A Pauli string.
            coeff: Numeric coefficient.
        """
        self._logger.debug(f"   {coeff}, {pauli_string}")
        num_pauli = len(pauli_string.map)
        num_modes = (1,) * num_pauli
        state_modes = tuple((qubit.index[0],) for qubit in pauli_string.map.keys())
        gate_data = tuple(
            self.PAULI[pauli.name].data.ptr for pauli in pauli_string.map.values()
        )
        cutn.network_operator_append_product(
            self._handle,
            self._operator,
            coeff,
            num_pauli,
            num_modes,
            state_modes,
            0,
            gate_data,
        )

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
        num_hyper_samples: int = 8,
        scratch_fraction: float = 0.5,
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
            num_hyper_samples: Number of hyper samples to use at contraction.
            scratch_fraction: Fraction of free memory on GPU to allocate as scratch
             space.

        Raises:
            MemoryError: If there is insufficient workspace size on a GPU device.
        """
        self._handle = libhandle.handle
        self._logger = set_logger("GeneralExpectationValue", loglevel)

        self._expectation = cutn.create_expectation(self._handle, state, operator)

        # Configure expectation value contraction.
        # TODO: factor into a separate method, if order-independent with workspace
        #  allocation
        num_hyper_samples_dtype = cutn.expectation_get_attribute_dtype(
            cutn.ExpectationAttribute.OPT_NUM_HYPER_SAMPLES
        )
        num_hyper_samples = np.asarray(num_hyper_samples, dtype=num_hyper_samples_dtype)
        cutn.expectation_configure(
            self._handle,
            self._expectation,
            cutn.ExpectationAttribute.OPT_NUM_HYPER_SAMPLES,
            num_hyper_samples.ctypes.data,
            num_hyper_samples.dtype.itemsize,
        )

        # Set a workspace. One may consider doing this somewhere else outside of the
        # class, but it seems to be really only needed for expectation value.
        # TODO: need to figure out if this needs to be done explicitly at all
        self._stream = (
            cp.cuda.Stream()
        )  # In current cuTN release it is unused (could be 0x0)
        free_mem = libhandle.dev.mem_info[0]
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
        else:
            self.destroy()
            raise MemoryError(
                f"Insufficient workspace size on the GPU device {self._handle.dev.id}"
            )

    def compute(self) -> tuple[complex, complex]:
        """Computes expectation value."""
        expectation_value = np.empty(1, dtype="complex128")
        state_norm = np.empty(1, dtype="complex128")
        cutn.expectation_compute(
            self._handle,
            self._expectation,
            self._work_desc,
            expectation_value.ctypes.data,
            state_norm.ctypes.data,
            self._stream.ptr,
        )
        self._stream.synchronize()
        return expectation_value.item(), state_norm.item()

    def destroy(self) -> None:
        """Destroys tensor network expectation value and workspace descriptor."""
        cutn.destroy_workspace_descriptor(self._work_desc)
        cutn.destroy_expectation(self._expectation)
        del self._scratch_space  # TODO is this the correct way?
