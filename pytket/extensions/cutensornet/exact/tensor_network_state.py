import logging
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


class TensorNetworkState:
    """Handles cuTensorNet tensor network state object."""

    def __init__(
            self,
            circuit: Circuit,
            libhandle: CuTensorNetHandle,
            loglevel: int = logging.INFO
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

        # Is this necessary?
        free_mem = libhandle.dev.mem_info[0]
        # use half of the totol free size
        scratch_size = free_mem // 2
        scratch_space = cp.cuda.alloc(scratch_size)

        for com in circuit.get_commands():
            pass