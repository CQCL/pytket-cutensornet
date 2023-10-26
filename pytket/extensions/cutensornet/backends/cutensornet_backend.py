# Copyright 2019-2023 Quantinuum
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

"""Methods to allow tket circuits to be run on the cuTensorNet simulator."""

import warnings

try:
    import cuquantum as cq  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)
from logging import warning
from typing import List, Union, Optional, Sequence
from uuid import uuid4
import numpy as np
from sympy import Expr  # type: ignore
from pytket.circuit import Circuit, OpType, Qubit  # type: ignore
from pytket.backends import ResultHandle, CircuitStatus, StatusEnum, CircuitNotRunError
from pytket.backends.backend import KwargTypes, Backend, BackendResult
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.cutensornet.tensor_network_convert import (
    TensorNetwork,
    ExpectationValueTensorNetwork,
    tk_to_tensor_network,
    measure_qubits_state,
)
from pytket.passes import auto_rebase_pass
from pytket.predicates import (  # type: ignore
    Predicate,
    GateSetPredicate,
    NoClassicalBitsPredicate,
    NoSymbolsPredicate,
)
from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    DecomposeBoxes,
    RemoveRedundancies,
    SynthesiseTket,
    FullPeepholeOptimise,
)
from pytket.utils.operators import QubitPauliOperator


class CuTensorNetBackend(Backend):
    """A pytket Backend wrapping around the cuTensorNet simulator."""

    _supports_state = True
    _supports_expectation = True
    _persistent_handles = False
    _GATE_SET = {
        OpType.X,
        OpType.Y,
        OpType.Z,
        OpType.S,
        OpType.Sdg,
        OpType.T,
        OpType.Tdg,
        OpType.V,
        OpType.Vdg,
        OpType.SX,
        OpType.SXdg,
        OpType.H,
        OpType.Rx,
        OpType.Ry,
        OpType.Rz,
        OpType.U1,
        OpType.U2,
        OpType.U3,
        OpType.TK1,
        OpType.TK2,
        OpType.CX,
        OpType.CY,
        OpType.CZ,
        OpType.CH,
        OpType.CV,
        OpType.CVdg,
        OpType.CSX,
        OpType.CSXdg,
        OpType.CRz,
        OpType.CRy,
        OpType.CRx,
        OpType.CU1,
        OpType.CU3,
        OpType.CCX,
        OpType.ECR,
        OpType.SWAP,
        OpType.CSWAP,
        OpType.ISWAP,
        OpType.XXPhase,
        OpType.YYPhase,
        OpType.ZZPhase,
        OpType.ZZMax,
        OpType.ESWAP,
        OpType.PhasedX,
        OpType.FSim,
        OpType.Sycamore,
    }

    # TODO: add self._backend_info?
    def __init__(self) -> None:
        """Constructs a new cuTensorNet backend object."""
        super().__init__()

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    # TODO: return some info? Should it return self._backend_info instantiated on
    #  construction?
    @property
    def backend_info(self) -> Optional[BackendInfo]:
        """Returns information on the backend."""
        return None

    @property
    def required_predicates(self) -> List[Predicate]:
        """Returns the minimum set of predicates that a circuit must satisfy.

        Predicates need to be satisfied before the circuit can be successfully run on
        this backend.

        Returns:
            List of required predicates.
        """
        preds = [
            NoClassicalBitsPredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(self._GATE_SET),
        ]
        return preds

    def rebase_pass(self) -> BasePass:
        """Defines rebasing method.

        Returns:
            Automatic rebase pass object based on the backend gate set.
        """
        return auto_rebase_pass(self._GATE_SET)

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        """Returns a default compilation pass.

        A suggested compilation pass that will guarantee the resulting circuit
        will be suitable to run on this backend with as few preconditions as
        possible.

        Args:
            optimisation_level: The level of optimisation to perform during
                compilation. Level 0 just solves the device constraints without
                optimising. Level 1 additionally performs some light optimisations.
                Level 2 adds more intensive optimisations that can increase compilation
                time for large circuits. Defaults to 1.
        Returns:
            Compilation pass guaranteeing required predicates.
        """
        assert optimisation_level in range(3)
        seq = [
            DecomposeBoxes(),
            RemoveRedundancies(),
        ]  # Decompose boxes into basic gates
        if optimisation_level == 1:
            seq.append(SynthesiseTket())  # Optional fast optimisation
        elif optimisation_level == 2:
            seq.append(FullPeepholeOptimise())  # Optional heavy optimisation
        seq.append(self.rebase_pass())  # Map to target gate set
        return SequencePass(seq)

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        """Returns circuit status object.

        Returns:
            CircuitStatus object.

        Raises:
            CircuitNotRunError: if there is no handle object in cache.
        """
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Optional[Union[int, Sequence[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """Submits circuits to the backend for running.

        The results will be stored in the backend's result cache to be retrieved by the
        corresponding get_<data> method.

        Args:
            circuits: List of circuits to be submitted.
            n_shots: Number of shots in case of shot-based calculation.
            valid_check: Whether to check for circuit correctness.

        Returns:
            Results handle objects.

        Raises:
            TypeError: If global phase is dependent on a symbolic parameter.
        """
        circuit_list = list(circuits)
        if valid_check:
            self._check_all_circuits(circuit_list)
        handle_list = []
        for circuit in circuit_list:
            state_tnet = tk_to_tensor_network(circuit)
            state = cq.contract(*state_tnet).flatten()
            try:  # This constraint (from pytket-Qulacs) seems reasonable?
                phase = float(circuit.phase)
                coeff = np.exp(phase * np.pi * 1j)
                state *= coeff  # type: ignore
            except TypeError:
                warning(
                    "Global phase is dependent on a symbolic parameter, so cannot "
                    "adjust for phase"
                )
            res_qubits = [qb for qb in sorted(circuit.qubits)]
            handle = ResultHandle(str(uuid4()))
            self._cache[handle] = {
                "result": BackendResult(q_bits=res_qubits, state=state)
            }
            handle_list.append(handle)
        return handle_list

    # TODO: this should be optionally parallelised with MPI
    #  (both wrt Pauli strings and contraction itself).
    def get_operator_expectation_value(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        post_selection: Optional[dict[Qubit, int]] = None,
        valid_check: bool = True,
    ) -> float:
        """Calculates expectation value of an operator using cuTensorNet contraction.

        Has an option to do post selection on an ancilla register.

        Args:
            state_circuit: Circuit representing state.
            operator: Operator which expectation value is to be calculated.
            valid_check: Whether to perform circuit validity check.
            post_selection: Dictionary of qubits to post select where the key is
                qubit and the value is bit outcome.

        Returns:
            Expectation value.
        """
        if valid_check:
            self._check_all_circuits([state_circuit])

        expectation = 0

        ket_network = TensorNetwork(state_circuit)
        bra_network = ket_network.dagger()

        if post_selection is not None:
            post_select_qubits = list(post_selection.keys())
            if set(post_select_qubits).issubset(operator.all_qubits):
                raise ValueError(
                    "Post selection qubit must not be a subset of operator qubits"
                )
            ket_network = measure_qubits_state(ket_network, post_selection)
            bra_network = measure_qubits_state(
                bra_network, post_selection
            )  # This needed because dagger does not work with post selection

        for qos, coeff in operator._dict.items():
            expectation_value_network = ExpectationValueTensorNetwork(
                bra_network, qos, ket_network
            )
            if isinstance(coeff, Expr):
                numeric_coeff = complex(coeff.evalf())  # type: ignore
            else:
                numeric_coeff = complex(coeff)  # type: ignore
            expectation_term = numeric_coeff * cq.contract(
                *expectation_value_network.cuquantum_interleaved
            )
            expectation += expectation_term
        return expectation.real

    def get_circuit_overlap(
        self,
        circuit_ket: Circuit,
        circuit_bra: Optional[Circuit] = None,
        valid_check: bool = True,
    ) -> float:
        """Calculates an overlap of two states represented by two circuits.

        Args:
            circuit_bra: Circuit representing the bra state.
            circuit_ket: Circuit representing the ket state.
            valid_check: Whether to perform circuit validity check.

        Returns:
            Overlap value.
        """
        if circuit_bra is None:
            circuit_bra = circuit_ket
        if valid_check:
            self._check_all_circuits([circuit_bra, circuit_ket])

        ket_net = TensorNetwork(circuit_ket)
        overlap_net_interleaved = ket_net.vdot(TensorNetwork(circuit_bra))
        overlap: float = cq.contract(*overlap_net_interleaved)
        return overlap
