import logging
import warnings
from typing import Optional

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


class TensorNetworkState:
    """Handles cuTensorNet tensor network state object."""

    def __init__(
        self,
        circuit: Circuit,
        libhandle: Optional[CuTensorNetHandle] = None,
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
        if libhandle is None:
            libhandle = CuTensorNetHandle()
        self._logger = set_logger("TensorNetwork", loglevel)
        self._circuit = circuit

        device_props = cp.cuda.runtime.getDeviceProperties(libhandle.device_id)
        self._logger.debug("===== device info ======")
        self._logger.debug("GPU-name:", device_props["name"].decode())
        self._logger.debug("GPU-clock:", device_props["clockRate"])
        self._logger.debug("GPU-memoryClock:", device_props["memoryClockRate"])
        self._logger.debug("GPU-nSM:", device_props["multiProcessorCount"])
        self._logger.debug("GPU-major:", device_props["major"])
        self._logger.debug("GPU-minor:", device_props["minor"])
        self._logger.debug("========================")

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
            libhandle.handle, cutn.StatePurity.PURE, num_qubits, qubits_dims, data_type
        )
        mutable_gates_map = {}
        for com in circuit.get_commands():
            gate_unitary = (
                com.op.get_unitary()
                .astype("complex128")
                .reshape([2] * (2 * com.op.n_qubits), order="F")
            )  # TODO: why column-major order?
            gate_strides = 0  # Always 0?
            gate_qubit_indices = [self._circuit.qubits.index(qb) for qb in com.qubits]
            gate_n_qubits = len(gate_qubit_indices)
            gate_qubit_indices_reversed = tuple(reversed(gate_qubit_indices))
            gate_id = cutn.state_apply_tensor(
                libhandle.handle,
                self._state,
                gate_n_qubits,
                gate_qubit_indices_reversed,
                gate_unitary.data.ptr,
                gate_strides,
                1,
                0,
                1,
            )
            if com.opgroup is not None:
                mutable_gates_map[com.opgroup] = gate_id
