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

"""Methods to allow tket circuits to be run on the cuTensorNet simulator."""

from abc import abstractmethod

from typing import List, Union, Optional, Sequence
from uuid import uuid4
from pytket.circuit import Circuit, OpType
from pytket.backends import ResultHandle, CircuitStatus, StatusEnum, CircuitNotRunError
from pytket.backends.backend import KwargTypes, Backend, BackendResult
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.cutensornet.general_state import (
    GeneralState,
)
from pytket.predicates import (  # type: ignore
    Predicate,
    NoSymbolsPredicate,
    NoClassicalControlPredicate,
    NoMidMeasurePredicate,
    NoBarriersPredicate,
    UserDefinedPredicate,
)
from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    DecomposeBoxes,
    RemoveRedundancies,
    SynthesiseTket,
    FullPeepholeOptimise,
    CustomPass,
)

from .._metadata import __extension_version__, __extension_name__


class _CuTensorNetBaseBackend(Backend):
    """A pytket Backend wrapping around the ``GeneralState`` simulator."""

    _persistent_handles = False

    def __init__(self) -> None:
        """Constructs a new cuTensorNet backend object."""
        super().__init__()

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    @property
    def required_predicates(self) -> List[Predicate]:
        """Returns the minimum set of predicates that a circuit must satisfy.

        Predicates need to be satisfied before the circuit can be successfully run on
        this backend.

        Returns:
            List of required predicates.
        """
        preds = [
            NoSymbolsPredicate(),
            NoClassicalControlPredicate(),
            NoMidMeasurePredicate(),
            NoBarriersPredicate(),
            UserDefinedPredicate(_check_all_unitary_or_measurements),
        ]
        return preds

    def rebase_pass(self) -> BasePass:
        """This method returns a dummy pass that does nothing, since there is
        no need to rebase. It is provided by requirement of a child of Backend,
        but it should not be included in the documentation.
        """
        return CustomPass(lambda circ: circ)  # Do nothing

    def default_compilation_pass(self, optimisation_level: int = 0) -> BasePass:
        """Returns a default compilation pass.

        A suggested compilation pass that will guarantee the resulting circuit
        will be suitable to run on this backend with as few preconditions as
        possible.

        Args:
            optimisation_level: The level of optimisation to perform during
                compilation. Level 0 just solves the device constraints without
                optimising. Level 1 additionally performs some light optimisations.
                Level 2 adds more intensive optimisations that can increase compilation
                time for large circuits. Defaults to 0.
        Returns:
            Compilation pass guaranteeing required predicates.
        """
        assert optimisation_level in range(3)
        seq = [
            DecomposeBoxes(),
            RemoveRedundancies(),
        ]  # Decompose boxes into basic gates

        # NOTE: these are the standard passes used in TKET backends. I haven't
        # benchmarked what's their effect on the simulation time.
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

    @abstractmethod
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
        ...


class CuTensorNetStateBackend(_CuTensorNetBaseBackend):
    """A pytket Backend using ``GeneralState`` to obtain state vectors."""

    _supports_state = True

    def __init__(self) -> None:
        """Constructs a new cuTensorNet backend object."""
        super().__init__()

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        """Returns information on the backend."""
        return BackendInfo(
            name="CuTensorNetStateBackend",
            architecture=None,
            device_name="NVIDIA GPU",
            version=__extension_name__ + "==" + __extension_version__,
            # The only constraint to the gateset is that they must be unitary matrices
            # or end-of-circuit measurements. These constraints are already specified
            # in the required_predicates of the backend. The empty set for gateset is
            # meant to be interpreted as "all gates".
            # TODO: list all gates in a programmatic way?
            gate_set=set(),
            misc={"characterisation": None},
        )

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

        Note:
            Any element from the ``TNConfig`` enum (see NVIDIA's CuTensorNet
            API) can be provided as arguments to this method. For instance:
            ``process_circuits(..., tn_config={"num_hyper_samples": 100})``.

        Args:
            circuits: List of circuits to be submitted.
            n_shots: Number of shots in case of shot-based calculation.
                This should be ``None``, since this backend does not support shots.
            valid_check: Whether to check for circuit correctness.
            tnconfig: Optional. A dict of cuTensorNet ``TNConfig`` keys and
                their values.
            scratch_fraction: Optional. Fraction of free memory on GPU to allocate as
                scratch space; value between 0 and 1. Defaults to ``0.8``.

        Returns:
            Results handle objects.
        """
        scratch_fraction = float(kwargs.get("scratch_fraction", 0.8))  # type: ignore
        tnconfig = kwargs.get("tnconfig", dict())  # type: ignore

        circuit_list = list(circuits)
        if valid_check:
            self._check_all_circuits(circuit_list)
        handle_list = []
        for circuit in circuit_list:
            with GeneralState(
                circuit, attributes=tnconfig, scratch_fraction=scratch_fraction
            ) as tn:
                sv = tn.get_statevector()
            res_qubits = [qb for qb in sorted(circuit.qubits)]
            handle = ResultHandle(str(uuid4()))
            self._cache[handle] = {"result": BackendResult(q_bits=res_qubits, state=sv)}
            handle_list.append(handle)
        return handle_list


class CuTensorNetShotsBackend(_CuTensorNetBaseBackend):
    """A pytket Backend using ``GeneralState`` to obtain shots."""

    _supports_shots = True
    _supports_counts = True

    def __init__(self) -> None:
        """Constructs a new cuTensorNet backend object."""
        super().__init__()

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        """Returns information on the backend."""
        return BackendInfo(
            name="CuTensorNetShotsBackend",
            architecture=None,
            device_name="NVIDIA GPU",
            version=__extension_name__ + "==" + __extension_version__,
            # The only constraint to the gateset is that they must be unitary matrices
            # or end-of-circuit measurements. These constraints are already specified
            # in the required_predicates of the backend. The empty set for gateset is
            # meant to be interpreted as "all gates".
            # TODO: list all gates in a programmatic way?
            gate_set=set(),
            misc={"characterisation": None},
        )

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

        Note:
            Any element from the ``TNConfig`` enum (see NVIDIA's CuTensorNet
            API) can be provided as arguments to this method. For instance:
            ``process_circuits(..., tn_config={"num_hyper_samples": 100})``.

        Args:
            circuits: List of circuits to be submitted.
            n_shots: Number of shots in case of shot-based calculation.
                Optionally, this can be a list of shots specifying the number of shots
                for each circuit separately.
            valid_check: Whether to check for circuit correctness.
            seed: An optional RNG seed. Different calls to ``process_circuits`` with the
                same seed will generate the same list of shot outcomes.
            tnconfig: Optional. A dict of cuTensorNet ``TNConfig`` keys and
                their values.
            scratch_fraction: Optional. Fraction of free memory on GPU to allocate as
                scratch space; value between 0 and 1. Defaults to ``0.8``.

        Returns:
            Results handle objects.
        """
        scratch_fraction = float(kwargs.get("scratch_fraction", 0.8))  # type: ignore
        tnconfig = kwargs.get("tnconfig", dict())  # type: ignore
        seed = kwargs.get("seed", None)

        if n_shots is None:
            raise ValueError(
                "You must specify n_shots when using CuTensorNetShotsBackend."
            )
        if type(n_shots) == int:
            all_shots = [n_shots] * len(circuits)
        else:
            all_shots = n_shots  # type: ignore

        circuit_list = list(circuits)
        if valid_check:
            self._check_all_circuits(circuit_list)
        handle_list = []
        for circuit, circ_shots in zip(circuit_list, all_shots):
            handle = ResultHandle(str(uuid4()))
            with GeneralState(
                circuit, attributes=tnconfig, scratch_fraction=scratch_fraction
            ) as tn:
                self._cache[handle] = {"result": tn.sample(circ_shots, seed=seed)}
            handle_list.append(handle)
        return handle_list


def _check_all_unitary_or_measurements(circuit: Circuit) -> bool:
    """Auxiliary function for custom predicate"""
    try:
        for cmd in circuit:
            if cmd.op.type != OpType.Measure:
                cmd.op.get_unitary()
        return True
    except:
        return False
