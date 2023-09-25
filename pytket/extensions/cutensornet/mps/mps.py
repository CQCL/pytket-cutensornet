# Copyright 2019-2023 Quantinuum
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
import warnings
from typing import Any, Optional, Union
from enum import Enum

from random import random  # type: ignore
import numpy as np  # type: ignore

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
try:
    import cuquantum as cq  # type: ignore
    from cuquantum.cutensornet import tensor  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)

from pytket.circuit import Command, Op, OpType, Qubit
from pytket.pauli import Pauli, QubitPauliString

from pytket.extensions.cutensornet.general import CuTensorNetHandle

# An alias so that `intptr_t` from CuQuantum's API (which is not available in
# base python) has some meaningful type name.
Handle = int
# An alias for the CuPy type used for tensors
try:
    Tensor = cp.ndarray
except NameError:
    Tensor = Any


class DirectionMPS(Enum):
    """An enum to refer to relative directions within the MPS."""

    LEFT = 0
    RIGHT = 1


class MPS:
    """Represents a state as a Matrix Product State.

    Attributes:
        chi (int): The maximum allowed dimension of a virtual bond.
        truncation_fidelity (float): The target fidelity of SVD truncation.
        tensors (list[Tensor]): A list of tensors in the MPS; ``tensors[0]`` is
            the leftmost and ``tensors[len(self)-1]`` is the rightmost; ``tensors[i]``
            and ``tensors[i+1]`` are connected in the MPS via a bond. All of the
            tensors are rank three, with the dimensions listed in ``.shape`` matching
            the left, right and physical bonds, in that order.
        canonical_form (dict[int, Optional[DirectionMPS]]): A dictionary mapping
            positions to the canonical form direction of the corresponding tensor,
            or ``None`` if it the tensor is not canonicalised.
        qubit_position (dict[pytket.circuit.Qubit, int]): A dictionary mapping circuit
            qubits to the position its tensor is at in the MPS.
        fidelity (float): A lower bound of the fidelity, obtained by multiplying
            the fidelities after each contraction. The fidelity of a contraction
            corresponds to ``|<psi|phi>|^2`` where ``|psi>`` and ``|phi>`` are the
            states before and after truncation (assuming both are normalised).
    """

    def __init__(
        self,
        libhandle: CuTensorNetHandle,
        qubits: list[Qubit],
        chi: Optional[int] = None,
        truncation_fidelity: Optional[float] = None,
        float_precision: Optional[Union[np.float32, np.float64]] = None,
    ):
        """Initialise an MPS on the computational state ``|0>``.

        Note:
            A ``libhandle`` should be created via a ``with CuTensorNet() as libhandle:``
            statement. The device where the MPS is stored will match the one specified
            by the library handle.

            Providing both a custom ``chi`` and ``truncation_fidelity`` will raise an
            exception. Choose one or the other (or neither, for exact simulation).

        Args:
            libhandle: The cuTensorNet library handle that will be used to carry out
                tensor operations on the MPS.
            qubits: The list of qubits in the circuit to be simulated.
            chi: The maximum value allowed for the dimension of the virtual
                bonds. Higher implies better approximation but more
                computational resources. If not provided, ``chi`` will be set
                to ``2**(len(qubits) // 2)``, which is enough for exact contraction.
            truncation_fidelity: Every time a two-qubit gate is applied, the virtual
                bond will be truncated to the minimum dimension that satisfies
                ``|<psi|phi>|^2 >= trucantion_fidelity``, where ``|psi>`` and ``|phi>``
                are the states before and after truncation (both normalised).
                If not provided, it will default to its maximum value 1.
            float_precision: The floating point precision used in tensor calculations;
                choose from ``numpy`` types: ``np.float64`` or ``np.float32``.
                Complex numbers are represented using two of such
                ``float`` numbers. Default is ``np.float64``.

        Raises:
            ValueError: If less than two qubits are provided.
            ValueError: If both ``chi`` and ``truncation_fidelity`` are fixed.
            ValueError: If the value of ``chi`` is set below 2.
            ValueError: If the value of ``truncation_fidelity`` is not in [0,1].
        """
        if chi is not None and truncation_fidelity is not None:
            raise ValueError("Cannot fix both chi and truncation_fidelity.")
        if chi is None:
            chi = max(2 ** (len(qubits) // 2), 2)
        if truncation_fidelity is None:
            truncation_fidelity = 1

        if chi < 2:
            raise ValueError("The max virtual bond dim (chi) must be >= 2.")
        if truncation_fidelity < 0 or truncation_fidelity > 1:
            raise ValueError("Provide a value of truncation_fidelity in [0,1].")

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

        self._lib = libhandle

        #######################################
        # Initialise the MPS with a |0> state #
        #######################################

        self.chi = chi
        self.truncation_fidelity = truncation_fidelity
        self.fidelity = 1.0

        n_tensors = len(qubits)
        if n_tensors == 0:  # There's no initialisation to be done
            return None
        elif n_tensors == 1:
            raise ValueError("Please, provide at least two qubits.")

        self.qubit_position = {q: i for i, q in enumerate(qubits)}

        # Create the list of tensors
        self.tensors = []
        self.canonical_form = {i: None for i in range(n_tensors)}

        # Append each of the tensors initialised in state |0>
        m_shape = (1, 1, 2)  # Two virtual bonds (dim=1) and one physical
        for i in range(n_tensors):
            m_tensor = cp.empty(m_shape, dtype=self._complex_t)
            # Initialise the tensor to ket 0
            m_tensor[0][0][0] = 1
            m_tensor[0][0][1] = 0
            self.tensors.append(m_tensor)

    def is_valid(self) -> bool:
        """Verify that the MPS object is valid.

        Specifically, verify that the MPS does not exceed the dimension limit ``chi`` of
        the virtual bonds, that physical bonds have dimension 2, that all tensors
        are rank three and that the data structure sizes are consistent.

        Returns:
            False if a violation was detected or True otherwise.
        """
        self._flush()

        chi_ok = all(
            all(dim <= self.chi for dim in self.get_virtual_dimensions(pos))
            for pos in range(len(self))
        )
        phys_ok = all(self.get_physical_dimension(pos) == 2 for pos in range(len(self)))
        shape_ok = all(len(tensor.shape) == 3 for tensor in self.tensors)

        ds_ok = set(self.canonical_form.keys()) == set(range(len(self)))
        ds_ok = ds_ok and set(self.qubit_position.values()) == set(range(len(self)))

        return chi_ok and phys_ok and shape_ok and ds_ok

    def apply_gate(self, gate: Command) -> MPS:
        """Apply the gate to the MPS.

        Note:
            Only one-qubit gates and two-qubit gates are supported. Two-qubit
            gates must act on adjacent qubits.

        Args:
            gate: The gate to be applied.

        Returns:
            ``self``, to allow for method chaining.

        Raises:
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
            RuntimeError: If gate acts on more than 2 qubits or acts on non-adjacent
                qubits.
            RuntimeError: If physical bond dimension where gate is applied is not 2.
        """
        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        positions = [self.qubit_position[q] for q in gate.qubits]
        if any(self.get_physical_dimension(pos) != 2 for pos in positions):
            raise RuntimeError(
                "Gates can only be applied to tensors with physical"
                + " bond dimension of 2."
            )

        if len(positions) == 1:
            self._apply_1q_gate(positions[0], gate.op)
            # NOTE: if the tensor was in canonical form, it remains being so,
            #   since it is guaranteed that the gate is unitary.

        elif len(positions) == 2:
            dist = positions[1] - positions[0]
            # We explicitly allow both dist==1 or dist==-1 so that non-symmetric
            # gates such as CX can use the same Op for the two ways it can be in.
            if dist not in [1, -1]:
                raise RuntimeError(
                    "Gates must be applied to adjacent qubits! "
                    + f"This is not satisfied by {gate}."
                )
            self._apply_2q_gate((positions[0], positions[1]), gate.op)
            # The tensors will in general no longer be in canonical form.
            self.canonical_form[positions[0]] = None
            self.canonical_form[positions[1]] = None

        else:
            raise RuntimeError(
                "Gates must act on only 1 or 2 qubits! "
                + f"This is not satisfied by {gate}."
            )
        return self

    def canonicalise(self, l_pos: int, r_pos: int) -> None:
        """Canonicalises the MPS object.

        Applies the necessary gauge transformations so that all MPS tensors
        to the left of position ``l_pos`` are in left orthogonal form and
        all MPS tensors to the right of ``r_pos`` in right orthogonal form.

        Args:
            l_pos: The position of the leftmost tensor that is not to be
                canonicalised.
            r_pos: The position of the rightmost tensor that is not to be
                canonicalised.
        """
        for pos in range(l_pos):
            self.canonicalise_tensor(pos, form=DirectionMPS.LEFT)
        for pos in reversed(range(r_pos + 1, len(self))):
            self.canonicalise_tensor(pos, form=DirectionMPS.RIGHT)

    def canonicalise_tensor(self, pos: int, form: DirectionMPS) -> None:
        """Canonicalises a tensor from an MPS object.

        Applies the necessary gauge transformations so that the tensor at
        position ``pos`` in the MPS is in the orthogonal form dictated by
        ``form``.

        Args:
            position: The position of the tensor to be canonicalised.
            form: LEFT form means that its conjugate transpose is its inverse if
                connected to its left bond and physical bond. Similarly for RIGHT.

        Raises:
            ValueError: If ``form`` is not a value in ``DirectionMPS``.
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
        """
        if form == self.canonical_form[pos]:
            # Tensor already in canonical form, nothing needs to be done
            return None

        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        # Glossary of bond IDs used here:
        # s -> shared virtual bond between T and Tnext
        # v -> the other virtual bond of T
        # V -> the other virtual bond of Tnext
        # p -> physical bond of T
        # P -> physical bond of Tnext

        # Gather the details from the MPS tensors at this position
        T = self.tensors[pos]

        # Assign the bond IDs
        if form == DirectionMPS.LEFT:
            next_pos = pos + 1
            Tnext = self.tensors[next_pos]
            T_bonds = "vsp"
            Q_bonds = "vap"
            R_bonds = "as"
            Tnext_bonds = "sVP"
            result_bonds = "aVP"
        elif form == DirectionMPS.RIGHT:
            next_pos = pos - 1
            Tnext = self.tensors[next_pos]
            T_bonds = "svp"
            Q_bonds = "avp"
            R_bonds = "as"
            Tnext_bonds = "VsP"
            result_bonds = "VaP"
        else:
            raise ValueError("Argument form must be a value in DirectionMPS.")

        # Apply QR decomposition
        subscripts = T_bonds + "->" + Q_bonds + "," + R_bonds
        options = {"handle": self._lib.handle, "device_id": self._lib.device_id}
        Q, R = tensor.decompose(
            subscripts, T, method=tensor.QRMethod(), options=options
        )

        # Contract R into Tnext
        subscripts = R_bonds + "," + Tnext_bonds + "->" + result_bonds
        result = cq.contract(subscripts, R, Tnext)

        # Update self.tensors
        self.tensors[pos] = Q
        self.tensors[pos].canonical_form = form
        self.tensors[next_pos] = result
        self.tensors[next_pos].canonical_form = None

    def vdot(self, other: MPS) -> complex:
        """Obtain the inner product of the two MPS: ``<self|other>``.

        It can be used to compute the squared norm of an MPS ``mps`` as
        ``mps.vdot(mps)``. The tensors within the MPS are not modified.

        Note:
            The state that is conjugated is ``self``.

        Args:
            other: The other MPS to compare against.

        Returns:
            The resulting complex number.

        Raises:
            RuntimeError: If number of tensors, dimensions or positions do not match.
            RuntimeError: If there are no tensors in the MPS.
            RuntimeError: If the ``CuTensorNetHandle`` is out of scope.
        """
        if self._lib._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of update_libhandle and CuTensorNetHandle.",
            )

        if len(self) != len(other):
            raise RuntimeError("Number of tensors do not match.")
        for i in range(len(self)):
            if self.get_physical_dimension(i) != other.get_physical_dimension(i):
                raise RuntimeError(
                    f"Physical bond dimension at position {i} do not match."
                )
        if self.qubit_position != other.qubit_position:
            raise RuntimeError(
                "The qubit labels or their position on the MPS do not match."
            )
        if len(self) == 0:
            raise RuntimeError("There are no tensors in the MPS.")

        self._flush()
        other._flush()

        # Special case if only one tensor remains
        if len(self) == 1:
            result = cq.contract("LRp,lrp->", self.tensors[0].conj(), other.tensors[0])

        else:
            # The two MPS will be contracted from left to right, storing the
            # ``partial_result`` tensor.
            partial_result = cq.contract(
                "LRp,lrp->Rr", self.tensors[0].conj(), other.tensors[0]
            )
            # Contract all tensors in the middle
            for pos in range(1, len(self) - 1):
                partial_result = cq.contract(
                    "Ll,LRp,lrp->Rr",
                    partial_result,
                    self.tensors[pos].conj(),
                    other.tensors[pos],
                )
            # Finally, contract the last tensor
            result = cq.contract(
                "Ll,LRp,lrp->",  # R and r are dim 1, so they are omitted; scalar result
                partial_result,
                self.tensors[-1].conj(),
                other.tensors[-1],
            )

        return complex(result)

    def sample(self) -> dict[Qubit, int]:
        """Returns a sample from a Z measurement applied on every qubit.

        Notes:
            The MPS ``self`` is not updated. This is equivalent to applying
            ``mps = self.copy()`` then ``mps.measure(mps.get_qubits())``.

        Returns:
            A dictionary mapping each of the qubits in the MPS to their 0 or 1 outcome.
        """

        # TODO: Copying is not strictly necessary, but to avoid it we would need to
        # modify the algorithm in `measure`. This may be done eventually if `copy`
        # is shown to be a bottleneck when sampling (which is likely).
        mps = self.copy()
        return mps.measure(mps.get_qubits())

    def measure(self, qubits: set[Qubit]) -> dict[Qubit, int]:
        """Applies a Z measurement on ``qubits``, updates the MPS and returns outcome.

        Notes:
            After applying this function, ``self`` will contain the MPS of the projected
            state over the non-measured qubits.

            The resulting state has been normalised.

        Args:
            qubits: The subset of qubits to be measured.

        Returns:
            A dictionary mapping the given ``qubits`` to their measurement outcome,
            i.e. either ``0`` or ``1``.

        Raises:
            ValueError: If an element in ``qubits`` is not a qubit in the MPS.
        """
        result = dict()

        # Obtain the positions that need to be measured and build the reverse dict
        position_qubit_map = dict()
        for q in qubits:
            if q not in self.qubit_position:
                raise ValueError(f"Qubit {q} is not a qubit in the MPS.")
            position_qubit_map[self.qubit_position[q]] = q
        positions = sorted(position_qubit_map.keys())

        # Tensor for postselection to |0>
        zero_tensor = cp.zeros(2, dtype=self._complex_t)
        zero_tensor[0] = 1

        # Measure and postselect each of the positions, one by one
        while positions:
            pos = positions.pop()  # The rightmost position to be measured

            # Convert to canonical form with center at this position
            self.canonicalise(pos, pos)

            # Glossary of bond IDs:
            # l -> left virtual bond of tensor in `pos`
            # r -> right virtual bond of tensor in `pos`
            # p -> physical bond of tensor in `pos`
            # P -> physical bond of tensor in `pos` (copy)

            # Take the tensor in this position and obtain its prob for |0>.
            # Since the MPS is in canonical form, this corresponds to the probability
            # if we were to take all of the other tensors into account.
            prob = cq.contract(
                "lrp,lrP,p,P->",  # No open bonds remain; this is just a scalar
                self.tensors[pos].conj(),
                self.tensors[pos],
                zero_tensor,
                zero_tensor,
            )

            # Throw a coin to decide measurement outcome
            outcome = 0 if prob > random() else 1
            result[position_qubit_map[pos]] = outcome

            # Postselect the MPS for this outcome, renormalising at the same time
            postselection_tensor = cp.zeros(2, dtype=self._complex_t)
            postselection_tensor[outcome] = 1 / np.sqrt(
                abs(outcome - prob)
            )  # Normalise

            self._postselect_qubit(position_qubit_map[pos], postselection_tensor)

        return result

    def postselect(self, qubit_outcomes: dict[Qubit, int]) -> float:
        """Applies a postselection, updates the MPS and returns its probability.

        Notes:
            After applying this function, ``self`` will contain the MPS of the projected
            state over the non-postselected qubits.

            The resulting state has been normalised.

        Args:
            qubit_outcomes: A dictionary mapping a subset of qubits in the MPS to their
                desired outcome value (either ``0`` or ``1``).

        Returns:
            The probability of this postselection to occur in a measurement.

        Raises:
            ValueError: If a key in ``qubit_outcomes`` is not a qubit in the MPS.
            ValueError: If a value in ``qubit_outcomes`` is other than ``0`` or ``1``.
            ValueError: If all of the qubits in the MPS are being postselected. Instead,
                you may wish to use ``get_amplitude()``.
        """
        for q, v in qubit_outcomes.items():
            if q not in self.qubit_position:
                raise ValueError(f"Qubit {q} is not a qubit in the MPS.")
            if v not in {0, 1}:
                raise ValueError(f"Outcome of {q} cannot be {v}. Choose int 0 or 1.")

        if len(qubit_outcomes) == len(self):
            raise ValueError(
                "Cannot postselect all qubits. You may want to use get_amplitude()."
            )

        # Apply a postselection for each of the qubits
        for qubit, outcome in qubit_outcomes.items():
            # Create the rank-1 postselection tensor
            postselection_tensor = cp.zeros(2, dtype=self._complex_t)
            postselection_tensor[outcome] = 1
            # Apply postselection
            self._postselect_qubit(qubit, postselection_tensor)

        # Calculate the squared norm of the postselected state; this is its probability
        prob = self.vdot(self)
        assert np.isclose(prob.imag, 0.0, atol=self._atol)
        prob = prob.real

        # Renormalise; it suffices to update the first tensor
        if len(self) > 0 and not np.isclose(prob, 0.0, atol=self._atol):
            self.tensors[0] = self.tensors[0] / np.sqrt(prob)
            self.canonical_form[0] = None

        return prob

    def _postselect_qubit(self, qubit: Qubit, postselection_tensor: cp.ndarray) -> None:
        """Postselect the qubit with the given tensor."""

        pos = self.qubit_position[qubit]
        self.tensors[pos] = cq.contract(
            "lrp,p->lr",
            self.tensors[pos],
            postselection_tensor,
        )

        # Glossary of bond IDs:
        # s -> shared bond between tensor in `pos` and next
        # v -> the other virtual bond of tensor in `pos`
        # V -> the other virtual bond of tensor in next position
        # p -> physical bond of tensor in `pos`
        # P -> physical bond of tensor in next position

        if len(self) == 1:  # This is the last tensor
            pass

        elif pos != 0:  # Contract with next tensor on the left
            self.tensors[pos - 1] = cq.contract(
                "sv,VsP->VvP",
                self.tensors[pos],
                self.tensors[pos - 1],
            )
            self.canonical_form[pos - 1] = None
        else:  # There are no tensors on the left, contract with the one on the right
            self.tensors[pos + 1] = cq.contract(
                "vs,sVP->vVP",
                self.tensors[pos],
                self.tensors[pos + 1],
            )
            self.canonical_form[pos + 1] = None

        # Shift all entries after `pos` to the left
        for q, p in self.qubit_position.items():
            if pos < p:
                self.qubit_position[q] = p - 1
        for p in range(pos, len(self) - 1):
            self.canonical_form[p] = self.canonical_form[p + 1]

        # Remove the entry from the data structures
        del self.qubit_position[qubit]
        del self.canonical_form[len(self) - 1]
        self.tensors.pop(pos)

    def expectation_value(self, pauli_string: QubitPauliString) -> float:
        """Obtains the expectation value of the Pauli string observable.

        Args:
            pauli_string: A pytket object representing a tensor product of Paulis.

        Returns:
            The expectation value.

        Raises:
            ValueError: If a key in ``pauli_string`` is not a qubit in the MPS.
        """
        for q in pauli_string.map.keys():
            if q not in self.qubit_position:
                raise ValueError(f"Qubit {q} is not a qubit in the MPS.")

        mps_copy = self.copy()
        pauli_optype = {Pauli.Z: OpType.Z, Pauli.X: OpType.X, Pauli.Y: OpType.Y}

        # Apply each of the Pauli operators to the MPS copy
        for qubit, pauli in pauli_string.map.items():
            if pauli != Pauli.I:
                pos = mps_copy.qubit_position[qubit]
                pauli_unitary = Op.create(pauli_optype[pauli]).get_unitary()
                pauli_tensor = cp.asarray(
                    pauli_unitary.astype(dtype=self._complex_t, copy=False),
                    dtype=self._complex_t,
                )

                # Contract the Pauli to the MPS tensor of the corresponding qubit
                mps_copy.tensors[pos] = cq.contract(
                    "lrp,Pp->lrP", mps_copy.tensors[pos], pauli_tensor
                )

        # Obtain the inner product
        value = self.vdot(mps_copy)
        assert np.isclose(value.imag, 0.0, atol=self._atol)

        return value.real

    def get_statevector(self) -> np.ndarray:
        """Returns the statevector with qubits in Increasing Lexicographic Order (ILO).

        Raises:
            ValueError: If there are no qubits left in the MPS.
        """
        if len(self) == 0:
            raise ValueError("There are no qubits left in this MPS.")

        # If there is only one qubit left, it is trivial
        if len(self) == 1:
            result_tensor = self.tensors[0]

        else:
            # Create the interleaved representation with all tensors
            interleaved_rep = []
            for pos in range(len(self)):
                interleaved_rep.append(self.tensors[pos])
                interleaved_rep.append(
                    ["v" + str(pos), "v" + str(pos + 1), "p" + str(pos)]
                )

            # Specify the output bond IDs in ILO order
            output_bonds = []
            for q in sorted(self.qubit_position.keys()):
                output_bonds.append("p" + str(self.qubit_position[q]))
            interleaved_rep.append(output_bonds)

            # Contract
            result_tensor = cq.contract(*interleaved_rep)

        # Convert to numpy vector and flatten
        statevector: np.ndarray = cp.asnumpy(result_tensor).flatten()
        return statevector

    def get_amplitude(self, state: int) -> complex:
        """Returns the amplitude of the chosen computational state.

        Notes:
            The result is equivalent to ``mps.get_statevector[b]``, but this method
            is faster when querying a single amplitude (or just a few).

        Args:
            state: The integer whose bitstring describes the computational state.
                The qubits in the bitstring are in increasing lexicographic order.

        Returns:
            The amplitude of the computational state in the MPS.
        """

        # Find out what the map MPS_position -> bit value is
        ilo_qubits = sorted(self.qubit_position.keys())
        mps_pos_bitvalue = dict()

        for i, q in enumerate(ilo_qubits):
            pos = self.qubit_position[q]
            bitvalue = 1 if state & 2 ** (len(self) - i - 1) else 0
            mps_pos_bitvalue[pos] = bitvalue

        # Carry out the contraction, starting from a dummy tensor
        result_tensor = cp.ones(1, dtype=self._complex_t)  # rank-1, dimension 1

        for pos in range(len(self)):
            postselection_tensor = cp.zeros(2, dtype=self._complex_t)
            postselection_tensor[mps_pos_bitvalue[pos]] = 1
            # Contract postselection with qubit into the result_tensor
            result_tensor = cq.contract(
                "l,lrp,p->r", result_tensor, self.tensors[pos], postselection_tensor
            )

        assert result_tensor.shape == (1,)
        return complex(result_tensor[0])

    def get_qubits(self) -> set[Qubit]:
        """Returns the set of qubits that this MPS is defined on."""
        return set(self.qubit_position.keys())

    def get_virtual_dimensions(self, position: int) -> tuple[int, int]:
        """Returns the virtual bonds dimension of the tensor ``tensors[position]``.

        Args:
            position: A position in the MPS.

        Returns:
            A tuple where the first element is the dimensions of the left virtual bond
            and the second elements is that of the right virtual bond.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """
        if position < 0 or position >= len(self):
            raise RuntimeError(f"Position {position} is out of bounds.")

        virtual_dims: tuple[int, int] = self.tensors[position].shape[:2]
        return virtual_dims

    def get_physical_dimension(self, position: int) -> int:
        """Returns the physical bond dimension of the tensor ``tensors[position]``.

        Args:
            position: A position in the MPS.

        Returns:
            The dimension of the physical bond.

        Raises:
            RuntimeError: If ``position`` is out of bounds.
        """
        if position < 0 or position >= len(self):
            raise RuntimeError(f"Position {position} is out of bounds.")

        physical_dim: int = self.tensors[position].shape[2]
        return physical_dim

    def get_device_id(self) -> int:
        """
        Returns:
            The identifier of the device (GPU) where the tensors are stored.
        """
        return int(self.tensors[0].device)

    def update_libhandle(self, libhandle: CuTensorNetHandle) -> None:
        """Update the ``CuTensorNetHandle`` used by this ``MPS`` object. Multiple
        objects may use the same handle.

        Args:
            libhandle: The new cuTensorNet library handle.

        Raises:
            RuntimeError: If the device (GPU) where ``libhandle`` was initialised
                does not match the one where the tensors of the MPS are stored.
        """
        if libhandle.device_id != self.get_device_id():
            raise RuntimeError(
                "Device of libhandle is not the one where the MPS is stored.",
                f"{libhandle.device_id} != {self.get_device_id()}",
            )
        self._lib = libhandle

    def copy(self) -> MPS:
        """
        Returns:
            A deep copy of the MPS on the same device.
        """
        self._flush()

        # Create a dummy object
        new_mps = MPS(self._lib, qubits=[])
        # Copy all data
        new_mps.chi = self.chi
        new_mps.truncation_fidelity = self.truncation_fidelity
        new_mps.fidelity = self.fidelity
        new_mps.tensors = [t.copy() for t in self.tensors]
        new_mps.canonical_form = self.canonical_form.copy()
        new_mps.qubit_position = self.qubit_position.copy()
        new_mps._complex_t = self._complex_t
        new_mps._real_t = self._real_t

        return new_mps

    def __len__(self) -> int:
        """
        Returns:
            The number of tensors in the MPS.
        """
        return len(self.tensors)

    def _apply_1q_gate(self, position: int, gate: Op) -> MPS:
        raise NotImplementedError(
            "MPS is a base class with no contraction algorithm implemented."
            + " You must use a subclass of MPS, such as MPSxGate or MPSxMPO."
        )

    def _apply_2q_gate(self, positions: tuple[int, int], gate: Op) -> MPS:
        raise NotImplementedError(
            "MPS is a base class with no contraction algorithm implemented."
            + " You must use a subclass of MPS, such as MPSxGate or MPSxMPO."
        )

    def _flush(self) -> None:
        # Does nothing in the general MPS case; but children classes with batched
        # gate contraction will redefine this method so that the last batch of
        # gates is applied.
        return None
