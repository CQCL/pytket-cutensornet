from typing import Any
from enum import Enum

from pytket.circuit import Circuit  # type ignore
from pytket.passes import DecomposeBoxes  # type: ignore
from pytket.transform import Transform  # type: ignore
from pytket.architecture import Architecture  # type: ignore
from pytket.passes import DefaultMappingPass  # type: ignore
from pytket.predicates import CompilationUnit  # type: ignore

from .mps import MPS
from .mps_gate import MPSxGate
from .mps_mpo import MPSxMPO

ContractionAlg = Enum("ContractionAlg", ["MPSxGate", "MPSxMPO"])


def simulate(
    circuit: Circuit, algorithm: ContractionAlg, chi: int, **kwargs: Any
) -> MPS:
    """Simulate the given circuit and return the ``MPS`` representing the final state.

    Note:
        If ``circuit`` contains circuit boxes, this method will decompose them.
        Similarly, it will route the circuit as appropriate. If you wish to retrieve
        the circuit after these passes were applied, use ``prepare_circuit()``.

    Args:
        circuit: The pytket circuit to be simulated.
        algorithm: Choose between the values of the ``ContractionAlg`` enum.
        chi: The maximum virtual bond dimension.
        **kwargs: Any extra argument accepted by the initialisers of the chosen
            ``algorithm`` class can be passed as a keyword argument. See the
            documentation of the corresponding class for details.

    Returns:
        An instance of ``MPS`` containing (an approximation of) the final state
        of the circuit.
    """

    prep_circ = prepare_circuit(circuit)

    float_precision = kwargs.get("float_precision", None)
    device_id = kwargs.get("device_id", None)

    if algorithm == ContractionAlg.MPSxGate:
        mps = MPSxGate(prep_circ.qubits, chi, float_precision, device_id)  # type: ignore
    elif algorithm == ContractionAlg.MPSxMPO:
        k = kwargs.get("k", None)
        mps = MPSxMPO(prep_circ.qubits, chi, k, float_precision, device_id)  # type: ignore
    else:
        print(f"Unrecognised algorithm: {algorithm}.")

    with mps.init_cutensornet():
        for g in prep_circ.get_commands():
            mps.apply_gate(g)

    return mps


def prepare_circuit(circuit: Circuit) -> Circuit:
    """Return an equivalent circuit with the appropriate structure to be simulated by
    an ``MPS`` algorithm.

    Args:
        circuit: The circuit to be simulated.

    Returns:
        An equivalent circuit with the appropriate structure.
    """

    prep_circ = circuit.copy()

    # Compile down to 1-qubit and 2-qubit gates with no implicit swaps
    DecomposeBoxes().apply(prep_circ)
    prep_circ.replace_implicit_wire_swaps()

    # Implement it in a line architecture
    cu = CompilationUnit(prep_circ)
    architecture = Architecture([(i, i + 1) for i in range(prep_circ.n_qubits - 1)])
    DefaultMappingPass(architecture).apply(cu)
    prep_circ = cu.circuit
    Transform.DecomposeBRIDGE().apply(prep_circ)

    return prep_circ
