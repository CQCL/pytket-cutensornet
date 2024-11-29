# Copyright 2019-2024 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
#     http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations  # type: ignore
from abc import ABC, abstractmethod
import warnings
import logging
from typing import Any, Optional, Type

import numpy as np  # type: ignore

from pytket.circuit import (
    Command,
    Op,
    OpType,
    Qubit,
    Bit,
    Conditional,
)
from pytket.pauli import QubitPauliString

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)

from pytket.extensions.cutensornet import CuTensorNetHandle
from .classical import apply_classical_command, from_little_endian

# An alias for the CuPy type used for tensors
try:
    Tensor = cp.ndarray
except NameError:
    Tensor = Any


class Config:
    """Configuration class for simulation using ``StructuredState``."""

    def __init__(
        self,
        chi: Optional[int] = None,
        truncation_fidelity: Optional[float] = None,
        seed: Optional[int] = None,
        float_precision: Type[Any] = np.float64,
        value_of_zero: float = 1e-16,
        leaf_size: int = 8,
        k: int = 4,
        optim_delta: float = 1e-5,
        loglevel: int = logging.WARNING,
    ):
        """Instantiate a configuration object for ``StructuredState`` simulation.

        Note:
            Providing both a custom ``chi`` and ``truncation_fidelity`` will raise an
            exception. Choose one or the other (or neither, for exact simulation).

        Args:
            chi: The maximum value allowed for the dimension of the virtual
                bonds. Higher implies better approximation but more
                computational resources. If not provided, ``chi`` will be unbounded.
            truncation_fidelity: Every time a two-qubit gate is applied, the virtual
                bond will be truncated to the minimum dimension that satisfies
                ``|<psi|phi>|^2 >= trucantion_fidelity``, where ``|psi>`` and ``|phi>``
                are the states before and after truncation (both normalised).
                If not provided, it will default to its maximum value 1.
            seed: Seed for the random number generator. Setting a seed provides
                reproducibility across simulations using ``StructuredState``, in the
                sense that they will produce the same sequence of measurement outcomes.
                Crucially, consecutive samples taken from the same ``StructuredState``
                can still be different from each other.
            float_precision: The floating point precision used in tensor calculations;
                choose from ``numpy`` types: ``np.float64`` or ``np.float32``.
                Complex numbers are represented using two of such
                ``float`` numbers. Default is ``np.float64``.
            value_of_zero: Any number below this value will be considered equal to zero.
                Even when no ``chi`` or ``truncation_fidelity`` is provided, singular
                values below this number will be truncated.
                We suggest to use a value slightly below what your chosen
                ``float_precision`` can reasonably achieve. For instance, ``1e-16`` for
                ``np.float64`` precision (default) and ``1e-7`` for ``np.float32``.
            leaf_size: For ``TTN`` simulation only. Sets the maximum number of
                qubits in a leaf node when using ``TTN``. Default is 8.
            k: For ``MPSxMPO`` simulation only. Sets the maximum number of layers
                the MPO is allowed to have before being contracted. Increasing this
                might increase fidelity, but it will also increase resource requirements
                exponentially. Default value is 4.
            optim_delta: For ``MPSxMPO`` simulation only. Sets the stopping criteria for
                the optimisation when contracting the ``k`` layers of MPO. Stops when
                the increase of fidelity between iterations is smaller than this value.
                Default value is ``1e-5``.
            loglevel: Internal logger output level. Use 30 for warnings only, 20 for
                verbose and 10 for debug mode.

        Raises:
            ValueError: If both ``chi`` and ``truncation_fidelity`` are fixed.
            ValueError: If the value of ``chi`` is set below 2.
            ValueError: If the value of ``truncation_fidelity`` is not in [0,1].
        """
        _CHI_LIMIT = 2**60
        if (
            chi is not None
            and chi < _CHI_LIMIT
            and truncation_fidelity is not None
            and truncation_fidelity != 1.0
        ):
            raise ValueError("Cannot fix both chi and truncation_fidelity.")
        if chi is None:
            chi = _CHI_LIMIT  # In practice, this is like having it be unbounded
        if truncation_fidelity is None:
            truncation_fidelity = 1

        if chi < 2:
            raise ValueError("The max virtual bond dim (chi) must be >= 2.")
        if truncation_fidelity < 0 or truncation_fidelity > 1:
            raise ValueError("Provide a value of truncation_fidelity in [0,1].")

        self.chi = chi
        self.truncation_fidelity = truncation_fidelity

        if float_precision is None or float_precision == np.float64:  # Double precision
            self._real_t = np.float64  # type: ignore
            self._complex_t = np.complex128  # type: ignore
            self._atol = 1e-12
        elif float_precision == np.float32:  # Single precision
            self._real_t = np.float32  # type: ignore
            self._complex_t = np.complex64  # type: ignore
            self._atol = 1e-4
        else:
            allowed_precisions = [np.float64, np.float32]
            raise TypeError(
                f"Value of float_precision must be in {allowed_precisions}."
            )
        self.zero = value_of_zero

        if value_of_zero > self._atol / 1000:
            warnings.warn(
                "Your chosen value_of_zero is relatively large. "
                "Faithfulness of final fidelity estimate is not guaranteed.",
                UserWarning,
            )

        self.seed = seed

        if leaf_size >= 65:  # Imposed to avoid bond ID collisions
            # More than 20 qubits is already unreasonable for a leaf anyway
            raise ValueError("Maximum allowed leaf_size is 65.")

        self.leaf_size = leaf_size
        self.k = k
        self.optim_delta = 1e-5
        self.loglevel = loglevel

    def copy(self) -> Config:
        """Standard copy of the contents."""
        return Config(
            chi=self.chi,
            truncation_fidelity=self.truncation_fidelity,
            seed=self.seed,
            float_precision=self._real_t,  # type: ignore
            value_of_zero=self.zero,
            leaf_size=self.leaf_size,
            k=self.k,
            optim_delta=self.optim_delta,
            loglevel=self.loglevel,
        )


class StructuredState(ABC):
    """Class representing a Tensor Network state."""

    _lib: CuTensorNetHandle
    _cfg: Config
    _logger: logging.Logger
    _bits_dict: dict[Bit, bool]  # Tracks the state of the classical variables

    def apply_gate(self, gate: Command) -> StructuredState:
        """Apply the command to the `StructuredState`.

        Note:
            Only one-qubit gates and two-qubit gates are supported.

        Args:
            gate: The command to be applied.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
            ValueError: If the command introduced is not a unitary gate.
            ValueError: If the command acts on more than 2 qubits.
        """
        self._logger.debug(f"Applying {gate}.")
        self._apply_command(gate.op, gate.qubits, gate.bits, gate.args)
        return self

    def _apply_command(
        self, op: Op, qubits: list[Qubit], bits: list[Bit], args: list[Any]
    ) -> None:
        """The implementation of `apply_gate`, acting on the unwrapped Command info."""
        if op.type == OpType.Measure:
            q = qubits[0]
            b = bits[0]
            self._bits_dict[b] = self.measure({q}, destructive=False)[q] != 0

        elif op.type == OpType.Reset:
            assert len(qubits)
            q = qubits[0]
            # Measure and correct if outcome is |1>
            outcome_1 = self.measure({q}, destructive=False)[q] != 0
            if outcome_1:
                self._apply_command(Op.create(OpType.X), [q], [], [q])

        elif op.is_gate():  # Either a unitary gate or a not supported "gate"
            try:
                unitary = op.get_unitary()
            except:
                raise ValueError(f"The command {op.type} introduced is not supported.")

            if len(qubits) not in [1, 2]:
                raise ValueError(
                    "Gates must act on only 1 or 2 qubits! "
                    + f"This is not satisfied by {op.type}."
                )

            self.apply_unitary(unitary, qubits)

        elif isinstance(op, Conditional):
            input_bits = args[: op.width]
            tgt_value = op.value
            # The input_bits encode a "value" int in little-endian
            var_value = from_little_endian([self._bits_dict[b] for b in input_bits])  # type: ignore
            # If the condition is apply the command in the body
            if var_value == tgt_value:
                self._apply_command(op.op, qubits, bits, args)

        else:  # A purely classical operation
            apply_classical_command(op, bits, args, self._bits_dict)

    @abstractmethod
    def is_valid(self) -> bool:
        """Verify that the tensor network state is valid.

        Returns:
            False if a violation was detected or True otherwise.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def apply_unitary(
        self, unitary: cp.ndarray, qubits: list[Qubit]
    ) -> StructuredState:
        """Applies the unitary to the specified qubits of the StructuredState.

        Note:
            It is assumed that the matrix provided by the user is unitary. If this is
            not the case, the program will still run, but its behaviour is undefined.

        Args:
            unitary: The matrix to be applied as a CuPy ndarray. It should either be
                a 2x2 matrix if acting on one qubit or a 4x4 matrix if acting on two.
            qubits: The qubits the unitary acts on. Only one qubit and two qubit
                unitaries are supported.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
            ValueError: If the number of qubits provided is not one or two.
            ValueError: If the size of the matrix does not match with the number of
                qubits provided.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def apply_scalar(self, scalar: complex) -> StructuredState:
        """Multiplies the state by a complex number.

        Args:
            scalar: The complex number to be multiplied.

        Returns:
            ``self``, to allow for method chaining.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def apply_qubit_relabelling(self, qubit_map: dict[Qubit, Qubit]) -> StructuredState:
        """Relabels each qubit ``q`` as ``qubit_map[q]``.

        This does not apply any SWAP gate, nor it changes the internal structure of the
        state. It simply changes the label of the physical bonds of the tensor network.

        Args:
            qubit_map: Dictionary mapping each qubit to its new label.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            ValueError: If any of the keys in ``qubit_map`` are not qubits in the state.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def vdot(self, other: StructuredState) -> complex:
        """Obtain the inner product of the two states: ``<self|other>``.

        It can be used to compute the squared norm of a state ``state`` as
        ``state.vdot(state)``. The tensors within the state are not modified.

        Note:
            The state that is conjugated is ``self``.

        Args:
            other: The other ``StructuredState``.

        Returns:
            The resulting complex number.

        Raises:
            RuntimeError: If the two states do not have the same qubits.
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def sample(self) -> dict[Qubit, int]:
        """Returns a sample from a Z measurement applied on every qubit.

        Notes:
            The contents of ``self`` are not updated. This is equivalent to applying
            ``state = self.copy()`` then ``state.measure(state.get_qubits())``.

        Returns:
            A dictionary mapping each qubit in the state to its 0 or 1 outcome.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def measure(self, qubits: set[Qubit], destructive: bool = True) -> dict[Qubit, int]:
        """Applies a Z measurement on each of the ``qubits``.

        Notes:
            After applying this function, ``self`` will contain the normalised
            projected state.

        Args:
            qubits: The subset of qubits to be measured.
            destructive: If ``True``, the resulting state will not contain the
                measured qubits. If ``False``, these qubits will appear on the
                state corresponding to the measurement outcome. Defaults to ``True``.

        Returns:
            A dictionary mapping the given ``qubits`` to their measurement outcome,
            i.e. either ``0`` or ``1``.

        Raises:
            ValueError: If an element in ``qubits`` is not a qubit in the state.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def postselect(self, qubit_outcomes: dict[Qubit, int]) -> float:
        """Applies a postselection, updates the states and returns its probability.

        Notes:
            After applying this function, ``self`` will contain the projected
            state over the non-postselected qubits.

            The resulting state has been normalised.

        Args:
            qubit_outcomes: A dictionary mapping a subset of qubits to their
                desired outcome value (either ``0`` or ``1``).

        Returns:
            The probability of this postselection to occur in a measurement.

        Raises:
            ValueError: If a key in ``qubit_outcomes`` is not a qubit in the state.
            ValueError: If a value in ``qubit_outcomes`` is other than ``0`` or ``1``.
            ValueError: If all of the qubits in the state are being postselected.
                Instead, you may wish to use ``get_amplitude()``.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def expectation_value(self, pauli_string: QubitPauliString) -> float:
        """Obtains the expectation value of the Pauli string observable.

        Args:
            pauli_string: A pytket object representing a tensor product of Paulis.

        Returns:
            The expectation value.

        Raises:
            ValueError: If a key in ``pauli_string`` is not a qubit in the state.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def get_fidelity(self) -> float:
        """Returns the current fidelity of the state."""
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def get_statevector(self) -> np.ndarray:
        """Returns the statevector with qubits in Increasing Lexicographic Order (ILO).

        Raises:
            ValueError: If there are no qubits left in the state.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def get_amplitude(self, state: int) -> complex:
        """Returns the amplitude of the chosen computational state.

        Notes:
            The result is equivalent to ``state.get_statevector[b]``, but this method
            is faster when querying a single amplitude (or just a few).

        Args:
            state: The integer whose bitstring describes the computational state.
                The qubits in the bitstring are in increasing lexicographic order.

        Returns:
            The amplitude of the computational state in ``self``.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    def get_bits(self) -> dict[Bit, bool]:
        """Returns the dictionary of bits and their values.

        A bit with value ``False`` corresponds to ``0``, and ``True`` is ``1``.
        """
        return self._bits_dict.copy()

    @abstractmethod
    def get_qubits(self) -> set[Qubit]:
        """Returns the set of qubits that ``self`` is defined on."""
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def get_byte_size(self) -> int:
        """Returns the number of bytes ``self`` currently occupies in GPU memory."""
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def get_device_id(self) -> int:
        """Returns the identifier of the device (GPU) where the tensors are stored."""
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def update_libhandle(self, libhandle: CuTensorNetHandle) -> None:
        """Update the ``CuTensorNetHandle`` used by ``self``. Multiple
        objects may use the same handle.

        Args:
            libhandle: The new cuTensorNet library handle.

        Raises:
            RuntimeError: If the device (GPU) where ``libhandle`` was initialised
                does not match the one where the tensors of ``self`` are stored.
        """
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def copy(self) -> StructuredState:
        """Returns a deep copy of ``self`` on the same device."""
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")

    @abstractmethod
    def _flush(self) -> None:
        raise NotImplementedError(f"Method not implemented in {type(self).__name__}.")
