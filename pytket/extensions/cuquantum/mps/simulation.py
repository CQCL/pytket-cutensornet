
from pytket.circuit import Circuit  # type ignore

from .mps import MPS
from .mps_gate import MPSxGate
from .mps_mpo import MPSxMPO

def simulate(circuit: Circuit, algorithm: str, chi: int, **kwargs) -> MPS:
    """Simulate the given circuit and return the ``MPS`` representing the final state.

    Note:
        If ``circuit`` contains circuit boxes, this method will decompose them.
        Similarly, it will route the circuit as appropriate. If you wish to retrieve
        the circuit after these passes were applied, use ``prepare_circuit()``.

        Use as ``with simulate(..) as mps:`` so that cuQuantum
        handles are automatically destroyed at the end of execution.

    Args:
        circuit: The pytket circuit to be simulated.
        algorithm: Choose between "MPSxGate" and "MPSxMPO".
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

    if algorithm == "MPSxGate":
        mps = MPSxGate(prep_circ.qubits, chi, float_precision)
    elif algorithm == "MPSxMPO":
        k = kwargs.get("k", None)
        mps = MPSxMPO(prep_circ.qubits, chi, k, float_precision)
    else:
        print(f"Unrecognised algorithm: {algorithm}.")

    for g in prep_circ.get_commands():
        mps.apply_gate(g)

    return self


def prepare_circuit(circuit: Circuit) -> Circuit
    """Return an equivalent circuit with the appropriate structure to be simulated by
    an ``MPS`` algorithm.

    Args:
        circuit: The circuit to be simulated.

    Returns:
        An equivalent circuit with the appropriate structure.
    """

    prep_circ = circuit.copy()
    DecomposeBoxes().apply(prep_circ)

    # Implement it in a line architecture
    cu = CompilationUnit(prep_circ)
    architecture = Architecture([(i, i+1) for i in range(prep_circ.n_qubits - 1)])
    DefaultMappingPass(architecture).apply(cu)
    prep_circuit = cu.circuit
    Transform.DecomposeBRIDGE().apply(prep_circ)

    return prep_circ
