import logging
import math
import warnings

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
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

        # This is only required (if at all?) when doing evaluation
        # free_mem = libhandle.dev.mem_info[0]
        # use half of the total free size
        # scratch_size = free_mem // 2
        # scratch_space = cp.cuda.alloc(scratch_size)

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
        cutn.destroy_network_operator()
