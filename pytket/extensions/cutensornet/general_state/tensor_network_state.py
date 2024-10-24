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
from typing import Union, Optional, Any
import warnings

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
import numpy as np
from sympy import Expr, Symbol  # type: ignore
from numpy.typing import NDArray
from pytket.circuit import Circuit, Qubit, Bit, OpType, Op
from pytket.extensions.cutensornet.general import set_logger
from pytket.utils import OutcomeArray
from pytket.utils.operators import QubitPauliOperator
from pytket.backends.backendresult import BackendResult

try:
    from cuquantum.cutensornet.experimental import NetworkState, NetworkOperator  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cuquantum", ImportWarning)


class GeneralState:
    """Wrapper of cuTensorNet's NetworkState for exact simulation of states.

    Constructs a tensor network for the output state of a pytket circuit.
    The qubits are assumed to be initialised in the ``|0>`` state.
    The object stores the *uncontracted* tensor network.

    Note:
        Preferably used as ``with GeneralState(...) as state:`` so that GPU memory is
        automatically released after execution.

        The ``circuit`` must not contain any ``CircBox`` or non-unitary command.

    Args:
        circuit: A pytket circuit to be converted into a tensor network.
        attributes: Optional. A dict of cuTensorNet ``TNConfig`` keys and
            their values.
        scratch_fraction: Optional. Fraction of free memory on GPU to allocate as
            scratch space; value between 0 and 1. Defaults to ``0.8``.
        loglevel: Internal logger output level. Use 30 for warnings only, 20 for
            verbose and 10 for debug mode.
    """

    def __init__(
        self,
        circuit: Circuit,
        attributes: Optional[dict] = None,
        scratch_fraction: float = 0.8,
        loglevel: int = logging.WARNING,
    ) -> None:
        self._logger = set_logger("GeneralState", loglevel)

        # Remove end-of-circuit measurements and keep track of them separately
        # It also resolves implicit swaps
        circuit, self._measurements = _remove_meas_and_implicit_swaps(circuit)
        # Identify each qubit with the index of the bond that represents it in the
        # tensor network stored in this GeneralState. Qubits are sorted in increasing
        # lexicographical order, which is the TKET standard.
        self._qubit_idx_map = {q: i for i, q in enumerate(sorted(circuit.qubits))}

        self._phase = circuit.phase
        self.n_qubits = circuit.n_qubits
        dim = 2  # We are always dealing with qubits, not qudits
        qubits_dims = (dim,) * len(circuit.qubits)  # qubit size
        data_type = "complex128"  # for now let that be hard-coded

        self.tn_state = NetworkState(
            qubits_dims,
            dtype=data_type,
            config=attributes,
            options={
                "memory_limit": f"{int(scratch_fraction*100)}%",
                "logger": self._logger,
            },
        )

        # Maintain a dict of tensor_id->Op for symbolic Ops to be update when the user
        # calls any evaluation function with the symbols specified.
        self._symbolic_ops: dict[int, Op] = dict()
        # Append all gates to the NetworkState
        commands = circuit.get_commands()
        for com in commands:
            is_fixed = len(com.op.free_symbols()) == 0
            if is_fixed:
                try:
                    gate_unitary = com.op.get_unitary()
                except Exception as e:
                    raise ValueError(
                        "All commands in the circuit must be unitary gates. The circuit"
                        f" contains {com}; no unitary matrix could be retrived from it."
                    ) from e
            else:
                # Dummy unitary to be updated later with user specified paramaters
                gate_unitary = np.identity(2**com.op.n_qubits, dtype="complex128")

            gate_qubit_indices = tuple(self._qubit_idx_map[qb] for qb in com.qubits)
            tensor_id = self.tn_state.apply_tensor_operator(
                gate_qubit_indices,
                _formatted_tensor(gate_unitary, com.op.n_qubits),
                immutable=is_fixed,
                unitary=True,
            )

            if not is_fixed:
                self._symbolic_ops[tensor_id] = com.op

        # If the circuit has no gates, apply one identity gate so that CuTensorNet does
        # not panic due to no tensor operator in the NetworkState
        if len(commands) == 0:
            tensor_id = self.tn_state.apply_tensor_operator(
                (0,),
                _formatted_tensor(np.identity(2), 1),
                immutable=True,
                unitary=True,
            )

    def get_statevector(
        self,
        symbol_map: Optional[dict[Symbol, float]] = None,
        on_host: bool = True,
    ) -> Union[cp.ndarray, np.ndarray]:
        """Contracts the circuit and returns the final statevector.

        Args:
            symbol_map: A dictionary where each element of the pytket circuit's
                ``.free_symbols()`` is assigned a real number.
            on_host: Optional. If ``True``, converts cupy ``ndarray`` to numpy
                ``ndarray``, copying it to host device (CPU). Defaults to ``True``.
        Returns:
            Either a ``cupy.ndarray`` on a GPU, or a ``numpy.ndarray`` on a
            host device (CPU). Arrays are returned in a 1D shape.

        Raises:
            ValueError: If not every free symbol in the circuit is assigned a
                value in ``symbol_map``.
        """
        _update_tensors(self.tn_state, self._symbolic_ops, symbol_map)

        self._logger.debug("(Statevector) contracting the TN")
        state_vector = self.tn_state.compute_state_vector()
        sv = state_vector.flatten()  # Convert to 1D
        if on_host:
            sv = cp.asnumpy(sv)
        # Apply the phase from the circuit
        sv *= np.exp(1j * np.pi * self._phase)
        return sv

    def get_amplitude(
        self,
        state: int,
        symbol_map: Optional[dict[Symbol, float]] = None,
    ) -> complex:
        """Returns the amplitude of the chosen computational state.

        Note:
            The result is equivalent to ``state.get_statevector[b]``, but this method
            is faster when querying a single amplitude (or just a few).

        Args:
            state: The integer whose bitstring describes the computational state.
                The qubits in the bitstring are in increasing lexicographic order.
            symbol_map: A dictionary where each element of the pytket circuit's
                ``.free_symbols()`` is assigned a real number.

        Returns:
            The amplitude of the computational state in ``self``.

        Raises:
            ValueError: If not every free symbol in the circuit is assigned a
                value in ``symbol_map``.
        """
        _update_tensors(self.tn_state, self._symbolic_ops, symbol_map)

        self._logger.debug("(Amplitude) contracting the TN")
        bitstring = format(state, "b").zfill(self.n_qubits)
        amplitude = self.tn_state.compute_amplitude(bitstring)
        # Apply the phase from the circuit
        amplitude *= np.exp(1j * np.pi * self._phase)
        return complex(amplitude)

    def expectation_value(
        self,
        operator: QubitPauliOperator,
        symbol_map: Optional[dict[Symbol, float]] = None,
    ) -> complex:
        """Calculates the expectation value of the given operator.

        Args:
            operator: The operator whose expectation value is to be calculated.
            symbol_map: A dictionary where each element of the pytket circuit's
                ``.free_symbols()`` is assigned a real number.

        Returns:
            The expectation value.

        Raises:
            ValueError: If the operator acts on qubits not present in the circuit.
            ValueError: If not every free symbol in the circuit is assigned a
                value in ``symbol_map``.
        """
        _update_tensors(self.tn_state, self._symbolic_ops, symbol_map)

        self._logger.debug("(Expectation value) converting operator to NetworkOperator")

        paulis = ["I", "X", "Y", "Z"]
        pauli_strs = dict()
        for pstr, coeff in operator._dict.items():

            # Raise an error if the operator acts on qubits that are not in the circuit
            if any(q not in self._qubit_idx_map.keys() for q in pstr.map.keys()):
                raise ValueError(
                    f"The operator is acting on qubits {pstr.map.keys()}, "
                    "but some of these are not present in the circuit, whose set of "
                    f"qubits is: {self._qubit_idx_map.keys()}."
                )

            pauli_list = [pstr[q] for q in self._qubit_idx_map.keys()]
            this_pauli_string = "".join(map(lambda x: paulis[x], pauli_list))
            pauli_strs[this_pauli_string] = complex(coeff)

        tn_operator = NetworkOperator.from_pauli_strings(pauli_strs, dtype="complex128")

        self._logger.debug("(Expectation value) contracting the TN")
        return complex(self.tn_state.compute_expectation(tn_operator))

    def sample(
        self,
        n_shots: int,
        symbol_map: Optional[dict[Symbol, float]] = None,
        seed: Optional[int] = None,
    ) -> BackendResult:
        """Obtains samples from the measurements at the end of the circuit.

        Args:
            n_shots: The number of samples to obtain.
            symbol_map: A dictionary where each element of the pytket circuit's
                ``.free_symbols()`` is assigned a real number.
            seed: An optional RNG seed. Different calls to ``sample`` with the same
                seed will generate the same list of shot outcomes.

        Returns:
            A pytket ``BackendResult`` with the data from the shots.

        Raises:
            ValueError: If the circuit contains no measurements.
            ValueError: If not every free symbol in the circuit is assigned a
                value in ``symbol_map``.
        """
        _update_tensors(self.tn_state, self._symbolic_ops, symbol_map)

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
        if seed is not None:
            seed = abs(seed)  # Must be a positive integer
        samples = self.tn_state.compute_sampling(
            nshots=n_shots,
            modes=measured_modes,
            seed=seed,
        )

        # Convert the data in `samples` to an `OutcomeArray` using `from_readouts`
        # which expects a 2D array `samples[SampleId][QubitId]` of 0s and 1s.
        self._logger.debug("(Sampling) converting samples to pytket Backend")
        readouts = np.empty(shape=(n_shots, num_measurements), dtype=int)
        sample_id = 0
        for bitstring, count in samples.items():
            outcome = [int(b) for b in bitstring]

            for _ in range(count):
                readouts[sample_id] = outcome
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

        The preferred approach is to use a context manager as in
        ``with GeneralState(...) as state:``. Otherwise, the user must release
        memory explicitly by calling ``destroy()``.
        """
        self._logger.debug("Freeing memory of GeneralState")
        self.tn_state.free()

    def __enter__(self) -> GeneralState:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.destroy()


class GeneralBraOpKet:
    """Constructs a tensor network for ``<bra|operator|ket>``.

    The qubits in ``ket`` and ``bra`` are assumed to be initialised in the ``|0>``
    state. The object stores the *uncontracted* tensor network.

    Note:
        Preferably used as ``with GeneralBraOpKet(...) as braket:`` so that GPU memory
        is automatically released after execution.

        The circuits must not contain any ``CircBox`` or non-unitary command.

        The operator is provided when ``contract`` is called.

    Args:
        bra: A pytket circuit describing the |bra> state.
        ket: A pytket circuit describing the |ket> state.
        attributes: Optional. A dict of cuTensorNet ``TNConfig`` keys and
            their values.
        scratch_fraction: Optional. Fraction of free memory on GPU to allocate as
            scratch space; value between 0 and 1. Defaults to ``0.8``.
        loglevel: Internal logger output level. Use 30 for warnings only, 20 for
            verbose and 10 for debug mode.

    Raises:
        ValueError: If the circuits for ``ket`` or ``bra`` contain measurements.
        ValueError: If the set of qubits of ``ket`` and ``bra`` do not match.
    """

    def __init__(
        self,
        bra: Circuit,
        ket: Circuit,
        attributes: Optional[dict] = None,
        scratch_fraction: float = 0.8,
        loglevel: int = logging.WARNING,
    ) -> None:
        self._logger = set_logger("GeneralBraOpKet", loglevel)

        # Check that the circuits have the same qubits
        if set(ket.qubits) != set(bra.qubits):
            raise ValueError(
                "The circuits given to GeneralBraOpKet must act on the same qubits."
            )
        # Remove end-of-circuit measurements and keep track of them separately
        # It also resolves implicit swaps
        ket, meas = _remove_meas_and_implicit_swaps(ket)
        if len(meas) != 0:
            raise ValueError(
                "The circuits given to a GeneralBraOpKet cannot have measurements."
            )
        bra, meas = _remove_meas_and_implicit_swaps(bra)
        if len(meas) != 0:
            raise ValueError(
                "The circuits given to a GeneralBraOpKet cannot have measurements."
            )
        # Identify each qubit with the index of the bond that represents it in the
        # tensor network stored in this GeneralState. Qubits are sorted in increasing
        # lexicographical order, which is the TKET standard.
        self._qubit_idx_map = {q: i for i, q in enumerate(sorted(ket.qubits))}
        self.n_qubits = ket.n_qubits

        dim = 2  # We are always dealing with qubits, not qudits
        qubits_dims = (dim,) * self.n_qubits  # qubit size
        data_type = "complex128"  # for now let that be hard-coded

        self.tn = NetworkState(
            qubits_dims,
            dtype=data_type,
            config=attributes,
            options={
                "memory_limit": f"{int(scratch_fraction*100)}%",
                "logger": self._logger,
            },
        )

        # Maintain a dict of tensor_id->Op for symbolic Ops to be update when the user
        # calls any evaluation function with the symbols specified.
        self._symbolic_ops: dict[int, Op] = dict()
        # Apply all commands from the ket circuit
        self._logger.debug("Converting the ket circuit to a NetworkState")
        commands = ket.get_commands()
        for com in commands:
            is_fixed = len(com.op.free_symbols()) == 0
            if is_fixed:
                try:
                    gate_unitary = com.op.get_unitary()
                except Exception as e:
                    raise ValueError(
                        "All commands in the circuit must be unitary gates. The circuit"
                        f" contains {com}; no unitary matrix could be retrived from it."
                    ) from e
            else:
                # Dummy unitary to be updated later with user specified paramaters
                gate_unitary = np.identity(2**com.op.n_qubits, dtype="complex128")

            gate_qubit_indices = tuple(self._qubit_idx_map[qb] for qb in com.qubits)
            tensor_id = self.tn.apply_tensor_operator(
                gate_qubit_indices,
                _formatted_tensor(gate_unitary, com.op.n_qubits),
                immutable=is_fixed,
                unitary=True,
            )

            if not is_fixed:
                self._symbolic_ops[tensor_id] = com.op

        # If the circuit has no gates, apply one identity gate so that CuTensorNet does
        # not panic due to no tensor operator in the NetworkState
        if len(commands) == 0:
            tensor_id = self.tn.apply_tensor_operator(
                (0,),
                _formatted_tensor(np.identity(2), 1),
                immutable=True,
                unitary=True,
            )
        self.ket_phase = ket.phase

        # Create a placeholder Pauli identity operator, to be replaced when calling
        # contract.
        self._logger.debug("Creating a placeholder operator and appending it")
        self._pauli_op_ids = []
        for mode in range(self.n_qubits):
            tensor_id = self.tn.apply_tensor_operator(
                (mode,),
                _formatted_tensor(np.identity(2), 1),
                immutable=False,
                unitary=True,
            )
            self._pauli_op_ids.append(tensor_id)

        # Apply all commands from the adjoint of the bra circuit
        self._logger.debug("Applying the dagger of the bra circuit to the NetworkState")
        commands = bra.dagger().get_commands()
        for com in commands:
            is_fixed = len(com.op.free_symbols()) == 0
            if is_fixed:
                try:
                    gate_unitary = com.op.get_unitary()
                except Exception as e:
                    raise ValueError(
                        "All commands in the circuit must be unitary gates. The circuit"
                        f" contains {com}; no unitary matrix could be retrived from it."
                    ) from e
            else:
                # Dummy unitary to be updated later with user specified paramaters
                gate_unitary = np.identity(2**com.op.n_qubits, dtype="complex128")

            gate_qubit_indices = tuple(self._qubit_idx_map[qb] for qb in com.qubits)
            tensor_id = self.tn.apply_tensor_operator(
                gate_qubit_indices,
                _formatted_tensor(gate_unitary, com.op.n_qubits),
                immutable=is_fixed,
                unitary=True,
            )

            if not is_fixed:
                self._symbolic_ops[tensor_id] = com.op

        # If the circuit has no gates, apply one identity gate so that CuTensorNet does
        # not panic due to no tensor operator in the NetworkState
        if len(commands) == 0:
            tensor_id = self.tn.apply_tensor_operator(
                (0,),
                _formatted_tensor(np.identity(2), 1),
                immutable=True,
                unitary=True,
            )
        self.bra_phase = bra.phase

    def contract(
        self,
        operator: Optional[QubitPauliOperator] = None,
        symbol_map: Optional[dict[Symbol, float]] = None,
    ) -> complex:
        """Contract the tensor network to obtain the value of ``<bra|operator|ket>``.

        Args:
            operator: A pytket ``QubitPauliOperator`` describing the operator. If not
                given, then the identity operator is used, so it computes inner product.
            symbol_map: A dictionary where each element of both pytket circuits'
                ``.free_symbols()`` is assigned a real number.

        Returns:
            The value of ``<bra|operator|ket>``.

        Raises:
            ValueError: If ``operator`` acts on qubits that are not in the circuits.
        """
        _update_tensors(self.tn, self._symbolic_ops, symbol_map)

        paulis = ["I", "X", "Y", "Z"]
        pauli_matrix = {
            "I": np.identity(2),
            "X": Op.create(OpType.X).get_unitary(),
            "Y": Op.create(OpType.Y).get_unitary(),
            "Z": Op.create(OpType.Z).get_unitary(),
        }
        pauli_strs: dict[str, complex] = dict()

        # Some care has to be taken when handling QubitPauliOperators, since identity
        # Paulis may be omitted from the dictionary.
        if operator is None:
            pauli_strs = {"".join("I" for _ in range(self.n_qubits)): complex(1.0)}
        else:
            for tk_pstr, coeff in operator._dict.items():
                # Raise an error if the operator acts on qubits missing from the circuit
                if any(q not in self._qubit_idx_map.keys() for q in tk_pstr.map.keys()):
                    raise ValueError(
                        f"The operator is acting on qubits {tk_pstr.map.keys()}, some "
                        "of these are missing from the set of qubits present in the "
                        f"circuits: {self._qubit_idx_map.keys()}."
                    )
                pauli_list = [tk_pstr[q] for q in self._qubit_idx_map.keys()]
                this_pauli_string = "".join(map(lambda x: paulis[x], pauli_list))
                pauli_strs[this_pauli_string] = complex(coeff)

        # Calculate the value by iterating over all components of the QubitPauliOperator
        value = 0.0
        zero_bitstring = "".join("0" for _ in range(self.n_qubits))
        for pstr, coeff in pauli_strs.items():
            # Update the NetworkState with this Pauli
            self._logger.debug(f"Updating the tensors of the Pauli operator {pstr}")
            for mode, pauli in enumerate(pstr):
                self.tn.update_tensor_operator(
                    self._pauli_op_ids[mode],
                    _formatted_tensor(pauli_matrix[pauli], 1),
                    unitary=True,
                )

            if isinstance(coeff, Expr):
                numeric_coeff = complex(coeff.evalf())  # type: ignore
            else:
                numeric_coeff = complex(coeff)

            # Compute the amplitude of the |0> state. Since NetworkState holds the
            # circuit bra.dagger()*operator*ket|0>, the value of the amplitude at <0|
            # will be <bra|operator|ket>.
            self._logger.debug(f"Computing the contribution of Pauli operator {pstr}")
            value += numeric_coeff * self.tn.compute_amplitude(zero_bitstring)

        # Apply the phases from the circuits
        value *= np.exp(1j * np.pi * self.ket_phase)
        value *= np.exp(-1j * np.pi * self.bra_phase)

        return complex(value)

    def destroy(self) -> None:
        """Destroy the tensor network and free up GPU memory.

        The preferred approach is to use a context manager as in
        ``with GeneralBraOpKet(...) as braket:``. Otherwise, the user must release
        memory explicitly by calling ``destroy()``.
        """
        self._logger.debug("Freeing memory of GeneralBraOpKet")
        self.tn.free()

    def __enter__(self) -> GeneralBraOpKet:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.destroy()


def _formatted_tensor(matrix: NDArray, n_qubits: int) -> cp.ndarray:
    """Convert a matrix to the tensor format accepted by NVIDIA's API."""

    cupy_matrix = cp.asarray(matrix, order="C").astype(dtype="complex128")
    # We also need to reshape since a matrix only has 2 bonds, but for an
    # n-qubit gate we want 2^n bonds for input and another 2^n for output
    return cupy_matrix.reshape([2] * (2 * n_qubits))


def _remove_meas_and_implicit_swaps(circ: Circuit) -> tuple[Circuit, dict[Qubit, Bit]]:
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


def _update_tensors(
    tn: NetworkState,
    symbolic_ops: dict[int, Op],
    symbol_map: Optional[dict[Symbol, float]],
) -> None:
    """Updates the tensors with the specified values for symbols.

    Args:
        tn: The NetworkState that we intend to update.
        symbolic_ops: A dictionary mapping ``tensor_id`` to the parameterised Op.
        symbol_map: A dictionary mapping symbols to real values.

    Raises:
        ValueError: If not every free symbol in the circuits is assigned a
            value in ``symbol_map``.
    """
    if symbol_map is None:
        symbol_map = dict()

    for tensor_id, op in symbolic_ops.items():
        subs_params = op.params.copy()
        if any(symb not in symbol_map.keys() for symb in op.free_symbols()):
            raise ValueError(
                f"Missing values for some of the free symbols {op.free_symbols()}. "
                f"Symbols given: {symbol_map}."
            )

        for symb in op.free_symbols():
            subs_params = [
                p.subs(symb, symbol_map[symb]) if isinstance(p, Expr) else p
                for p in subs_params
            ]
        gate_unitary = Op.create(op.type, subs_params).get_unitary()

        tn.update_tensor_operator(
            tensor_id,
            _formatted_tensor(gate_unitary, op.n_qubits),
            unitary=True,
        )
