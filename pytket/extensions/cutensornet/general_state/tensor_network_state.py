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
    from cuquantum.cutensornet.experimental import NetworkState, NetworkOperator  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cuquantum", ImportWarning)

# TODO: Add the options as argument to be passed to NetworkState
class GeneralState:  # TODO: Write it as a context manager so that I can call free()
    """Wrapper of cuTensorNet object for exact simulations via path optimisation."""

    def __init__(
        self,
        circuit: Circuit,
        loglevel: int = logging.WARNING,
    ) -> None:
        """Constructs a tensor network for the output state of a pytket circuit.

        The qubits are assumed to be initialised in the ``|0>`` state.
        The resulting object stores the *uncontracted* tensor network.

        Note:
            The ``circuit`` must not contain any ``CircBox`` or non-unitary command.

        Args:
            circuit: A pytket circuit to be converted to a tensor network.
            loglevel: Internal logger output level.
        """
        self._logger = set_logger("GeneralState", loglevel)
        # TODO: Consider supporting scratch_fraction of some form of memory limit

        # Remove end-of-circuit measurements and keep track of them separately
        # It also resolves implicit swaps
        # TODO: Is there any point in keeping the circuit around?
        self._circuit, self._measurements = _remove_meas_and_implicit_swaps(circuit)
        # Identify each qubit with the index of the bond that represents it in the
        # tensor network stored in this GeneralState. Qubits are sorted in increasing
        # lexicographical order, which is the TKET standard.
        self._qubit_idx_map = {q: i for i, q in enumerate(sorted(self._circuit.qubits))}

        num_qubits = self._circuit.n_qubits
        dim = 2  # We are always dealing with qubits, not qudits
        qubits_dims = (dim,) * num_qubits  # qubit size
        self._logger.debug(f"Converting a quantum circuit with {num_qubits} qubits.")
        data_type = "complex128"  # for now let that be hard-coded

        self._state = NetworkState(qubits_dims, dtype=data_type)

        self._gate_tensors = []  # TODO: Do I still need to keep these myself?
        commands = self._circuit.get_commands()

        # Append all gates to the NetworkState
        for com in commands:
            try:
                gate_unitary = com.op.get_unitary()
            except:
                raise ValueError(
                    "All commands in the circuit must be unitary gates. The circuit "
                    f"contains {com}; no unitary matrix could be retrived from it."
                )
            self._gate_tensors.append(_formatted_tensor(gate_unitary, com.op.n_qubits))
            gate_qubit_indices = tuple(self._qubit_idx_map[qb] for qb in com.qubits)

            tensor_id = self._state.apply_tensor_operator(
                gate_qubit_indices,
                self._gate_tensors[-1],
                immutable=True,  # TODO: Change for parameterised gates
                unitary=True,
            )

        # If the circuit has no gates, apply one identity gate so that CuTensorNet does not panic
        # due to no tensor operator in the NetworkState
        if len(commands) == 0:
            identity_tensor = _formatted_tensor(np.identity(2, dtype="complex128"), 1)
            tensor_id = self._state.apply_tensor_operator(
                (0, ),
                identity_tensor,
                immutable=True,
                unitary=True,
            )

    def get_statevector(
        self,
        attributes: Optional[dict] = None,
        scratch_fraction: float = 0.75,
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

        self._logger.debug("(Statevector) contracting the TN")
        state_vector = self._state.compute_state_vector()
        sv = state_vector.flatten()  # Convert to 1D
        if on_host:
            sv = cp.asnumpy(sv)
        # Apply the phase from the circuit
        sv *= np.exp(1j * np.pi * self._circuit.phase)
        return sv

    def expectation_value(
        self,
        operator: QubitPauliOperator,
    ) -> complex:
        """Calculates the expectation value of the given operator.

        Args:
            operator: The operator whose expectation value is to be measured.

        Raises:
            ValueError: If the operator acts on qubits not present in the circuit.

        Returns:
            The expectation value.
        """

        self._logger.debug("(Expectation value) converting operator to NetworkOperator")

        paulis = ["I", "X", "Y", "Z"]
        pauli_strs = dict()
        for pstr, coeff in operator._dict.items():

            # Raise an error if the operator acts on qubits that are not in the circuit
            if any(q not in self._circuit.qubits for q in pstr.map.keys()):
                raise ValueError(
                    f"The operator is acting on qubits {pstr.map.keys()}, "
                    "but some of these are not present in the circuit, whose set of "
                    f"qubits is: {self._circuit.qubits}."
                )

            pauli_list = [pstr[q] for q in self._qubit_idx_map.keys()]
            this_pauli_string = "".join(map(lambda x: paulis[x], pauli_list))
            pauli_strs[this_pauli_string] = complex(coeff)

        tn_operator = NetworkOperator.from_pauli_strings(pauli_strs, dtype="complex128")

        self._logger.debug("(Expectation value) contracting the TN")
        return self._state.compute_expectation(tn_operator)

    def sample(  # TODO: Support seeds (and test)
        self,
        n_shots: int,
    ) -> BackendResult:
        """Obtains samples from the measurements at the end of the circuit.

        Args:
            n_shots: The number of samples to obtain.
        Raises:
            ValueError: If the circuit contains no measurements.
        Returns:
            A pytket ``BackendResult`` with the data from the shots.
        """

        num_measurements = len(self._measurements)
        if num_measurements == 0:
            raise ValueError(
                "Cannot sample from the circuit, it contains no measurements."
            )
        # We will need both a list of the qubits and a list of the classical bits
        # and it is essential that the elements in the same index of either list
        # match according to the self._measurements map. We guarantee this here.
        qbit_list, cbit_list = zip(*self._measurements.items())
        measured_modes = tuple(self._qubit_idx_map[qb] for qb in qbit_list)

        self._logger.debug("(Sampling) contracting the TN")
        samples = self._state.compute_sampling(
            nshots=n_shots,
            modes=measured_modes,
        )

        # Convert the data in `samples` to an `OutcomeArray` using `from_readouts`
        # which expects a 2D array `samples[SampleId][QubitId]` of 0s and 1s.
        self._logger.debug("(Sampling) converting samples to pytket Backend")
        readouts = np.empty(shape=(n_shots, num_measurements), dtype=int)
        sample_id = 0
        for bitstring, count in samples.items():
            outcome = [int(b) for b in bitstring]

            for _ in range(count):
                readouts[sample_id] = outcome  # TODO: test endian-ness
                sample_id += 1

        shots = OutcomeArray.from_readouts(readouts)

        # We need to specify which bits correspond to which columns in the shots
        # table. Since cuTensorNet promises that the ordering of outcomes is
        # determined by the ordering we provided as `measured_modes`, which in
        # turn corresponds to the ordering of qubits in `qbit_list`, the fact that
        # `cbit_list` has the appropriate order in relation to `self._measurements`
        # implies this defines the ordering of classical bits we intend.
        return BackendResult(c_bits=cbit_list, shots=shots)

    def destroy(self) -> None:
        """Destroy the tensor network and free up GPU memory.

        Note:
            Users are required to call `destroy()` when done using a
            `GeneralState` object. GPU memory deallocation is not
            guaranteed otherwise.
        """
        self._logger.debug("Freeing memory of NetworkState")
        self._state.free()


def _formatted_tensor(matrix: NDArray, n_qubits: int) -> cp.ndarray:
    """Convert a matrix to the tensor format accepted by NVIDIA's API."""

    cupy_matrix = cp.asarray(matrix).astype(dtype="complex128")
    # We also need to reshape since a matrix only has 2 bonds, but for an
    # n-qubit gate we want 2^n bonds for input and another 2^n for output
    return cupy_matrix.reshape([2] * (2 * n_qubits))


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
