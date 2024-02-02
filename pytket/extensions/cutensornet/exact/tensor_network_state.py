import logging
import math
import warnings

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
from pytket.circuit import Circuit  # type: ignore
from pytket.extensions.cutensornet.general import set_logger
from pytket.extensions.cutensornet.mps import CuTensorNetHandle

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
            gate_tensor = (
                com.op.get_unitary()
                .astype("complex128")
                .reshape([2] * (2 * com.op.n_qubits), order="F")
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

    def update_gates(self, gates_update_map: dict) -> None:
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
            gate_tensor = unitary.astype("complex128").reshape(
                [2] * (2 * int(gate_n_qubits)), order="F"
            )
            cutn.state_update_tensor(
                self._handle, self._state, gate_id, gate_tensor.data.ptr, 0, 1
            )
