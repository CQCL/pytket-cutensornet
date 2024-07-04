# Copyright 2019-2024 Quantinuum
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
import logging
from typing import Union, Optional, Tuple, Dict
import warnings

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
import numpy as np
from sympy import Expr  # type: ignore
from numpy.typing import NDArray
from pytket.circuit import Circuit, Qubit, Bit, OpType
from pytket.extensions.cutensornet.general import CuTensorNetHandle, set_logger
from pytket.utils import OutcomeArray
from pytket.utils.operators import QubitPauliOperator
from pytket.backends.backendresult import BackendResult

try:
    import cuquantum as cq  # type: ignore
    from cuquantum import cutensornet as cutn  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cuquantum", ImportWarning)


class GeneralState:
    """Wrapper of cuTensorNet object for exact simulations via path optimisation."""

    def __init__(
        self,
        circuit: Circuit,
        libhandle: CuTensorNetHandle,
        loglevel: int = logging.INFO,
    ) -> None:
        """Constructs a tensor network for the output state of a pytket circuit.

        The qubits are assumed to be initialised in the ``|0>`` state.
        The resulting object stores the *uncontracted* tensor network.

        Note:
            A ``libhandle`` is created via a ``with CuTensorNetHandle() as libhandle:``
            statement. The device where the ``GeneralState`` is stored will match the
            one specified by the library handle.

        Note:
            The ``circuit`` must not contain any ``CircBox`` or non-unitary command.

        Args:
            circuit: A pytket circuit to be converted to a tensor network.
            libhandle: An instance of a ``CuTensorNetHandle``.
            loglevel: Internal logger output level.
        """
        self._logger = set_logger("GeneralState", loglevel)
        self._lib = libhandle
        libhandle.print_device_properties(self._logger)

        # Remove end-of-circuit measurements and keep track of them separately
        # It also resolves implicit swaps
        self._circuit, self._measurements = _remove_meas_and_implicit_swaps(circuit)
        # Identify each qubit with the index of  the bond that represents it in the
        # tensor network stored in this GeneralState. Qubits are sorted in increasing
        # lexicographical order, which is the TKET standard.
        self._qubit_idx_map = {q: i for i, q in enumerate(sorted(self._circuit.qubits))}

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
        for com in self._circuit.get_commands():
            try:
                gate_unitary = com.op.get_unitary()
            except:
                raise ValueError(
                    "All commands in the circuit must be unitary gates. The circuit "
                    f"contains {com}; no unitary matrix could be retrived for it."
                )
            self._gate_tensors.append(_formatted_tensor(gate_unitary, com.op.n_qubits))
            gate_qubit_indices = tuple(self._qubit_idx_map[qb] for qb in com.qubits)

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
            attributes: Optional. A dict of cuTensorNet `StateAttribute` keys and
                their values.
            scratch_fraction: Optional. Fraction of free memory on GPU to allocate as
                scratch space.
            on_host: Optional. If ``True``, converts cupy ``ndarray`` to numpy
                ``ndarray``, copying it to host device (CPU).
        Raises:
            MemoryError: If there is insufficient workspace on GPU.
        Returns:
            Either a ``cupy.ndarray`` on a GPU, or a ``numpy.ndarray`` on a
            host device (CPU). Arrays are returned in a 1D shape.
        """

        ####################################
        # Configure the TN for contraction #
        ####################################
        if attributes is None:
            attributes = dict()
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
            attributes: Optional. A dict of cuTensorNet `ExpectationAttribute` keys
                and their values.
            scratch_fraction: Optional. Fraction of free memory on GPU to allocate as
                 scratch space.

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
                (self._qubit_idx_map[qb],) for qb in qubit_pauli_map.keys()
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

    def sample(
        self,
        n_shots: int,
        attributes: Optional[dict] = None,
        scratch_fraction: float = 0.5,
    ) -> BackendResult:
        """Obtains samples from the measurements at the end of the circuit.

        Args:
            n_shots: The number of samples to obtain.
            attributes: Optional. A dict of cuTensorNet `SamplerAttribute` keys and
                their values.
            scratch_fraction: Optional. Fraction of free memory on GPU to allocate as
                scratch space.
        Raises:
            MemoryError: If there is insufficient workspace on GPU.
        Returns:
            A pytket ``BackendResult`` with the data from the shots.
        """

        num_measurements = len(self._measurements)
        # We will need both a list of the qubits and a list of the classical bits
        # and it is essential that the elements in the same index of either list
        # match according to the self._measurements map. We guarantee this here.
        qbit_list, cbit_list = zip(*self._measurements.items())
        measured_modes = tuple(self._qubit_idx_map[qb] for qb in qbit_list)

        ############################################
        # Configure the cuTensorNet sampler object #
        ############################################

        sampler = cutn.create_sampler(
            handle=self._lib.handle,
            tensor_network_state=self._state,
            num_modes_to_sample=num_measurements,
            modes_to_sample=measured_modes,
        )

        if attributes is None:
            attributes = dict()
        attribute_pairs = [
            (getattr(cutn.SamplerAttribute, k), v) for k, v in attributes.items()
        ]

        for attr, val in attribute_pairs:
            attr_dtype = cutn.sampler_get_attribute_dtype(attr)
            attr_arr = np.asarray(val, dtype=attr_dtype)
            cutn.sampler_configure(
                self._lib.handle,
                sampler,
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
            cutn.sampler_prepare(
                self._lib.handle,
                sampler,
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

            ###########################
            # Sample from the circuit #
            ###########################
            samples = np.empty((num_measurements, n_shots), dtype="int64", order="F")
            cutn.sampler_sample(
                self._lib.handle,
                sampler,
                n_shots,
                work_desc,
                samples.ctypes.data,
                stream.ptr,
            )
            stream.synchronize()

            # Convert the data in `samples` to an `OutcomeArray`
            # `samples` is a 2D numpy array `samples[SampleId][QubitId]`, which is
            # the transpose of what `OutcomeArray.from_readouts` expects
            shots = OutcomeArray.from_readouts(samples.T)
            # We need to specify which bits correspond to which columns in the shots
            # table. Since cuTensorNet promises that the ordering of outcomes is
            # determined by the ordering we provided as `measured_modes`, which in
            # turn corresponds to the ordering of qubits in `qbit_list`, the fact that
            # `cbit_list` has the appropriate order in relation to `self._measurements`
            # determines this defines the ordering of classical bits we intend.
            return BackendResult(c_bits=cbit_list, shots=shots)

        finally:
            ##############################
            # Destroy the Sampler object #
            ##############################
            cutn.destroy_workspace_descriptor(work_desc)  # type: ignore
            cutn.destroy_sampler(sampler)
            del scratch_space

    def destroy(self) -> None:
        """Destroy the tensor network and free up GPU memory.

        Note:
            Users are required to call `destroy()` when done using a
            `GeneralState` object. GPU memory deallocation is not
            guaranteed otherwise.
        """
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


def _remove_meas_and_implicit_swaps(circ: Circuit) -> Tuple[Circuit, Dict[Qubit, Bit]]:
    """Convert a pytket Circuit to an equivalent circuit with no measurements or
    implicit swaps. The measurements are returned as a map between qubits and bits.

    Only supports end-of-circuit measurements, which are removed from the returned
    circuit and added to the dictionary.
    """
    pure_circ = Circuit()
    for q in circ.qubits:
        pure_circ.add_qubit(q)
    q_perm = circ.implicit_qubit_permutation()

    measure_map = dict()
    # Track measured Qubits to identify mid-circuit measurement
    measured_qubits = set()

    for command in circ:
        cmd_qubits = [q_perm[q] for q in command.qubits]

        for q in cmd_qubits:
            if q in measured_qubits:
                raise ValueError("Circuit contains a mid-circuit measurement")

        if command.op.type == OpType.Measure:
            measure_map[cmd_qubits[0]] = command.bits[0]
            measured_qubits.add(cmd_qubits[0])
        else:
            if command.bits:
                raise ValueError("Circuit contains an operation on a bit")
            pure_circ.add_gate(command.op, cmd_qubits)

    pure_circ.add_phase(circ.phase)
    return pure_circ, measure_map  # type: ignore
