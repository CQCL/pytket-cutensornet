from pytket.circuit import Circuit  # type: ignore
from pytket.circuit import Qubit  # type: ignore
from pytket.pauli import QubitPauliString  # type: ignore
from pytket.pauli import Pauli  # type: ignore
from pytket.utils import QubitPauliOperator  # type: ignore
from pytket.extensions.cutensornet.backends import CuTensorNetBackend


def hadamard_test(circ: Circuit) -> float:
    """Performs a hadamard test on a circuit.
    The circuit must have a qubit in register a and q.
    The tensor network projects on the state |0> in register a to get p0.
    The tensor network projects on the state |1> in register a to get p1.
    The expectation value is p0 - p1.

    Args:
        circ (Circuit): Circuit to perform hadamard test on

    Raises:
        ValueError: If circuit does not have qubits in registers a and q

    Returns:
        float: Expectation value of hadamard test
    """
    if {qreg.name for qreg in circ.q_registers} != {"a", "q"}:
        raise ValueError("Circuit must only have qubits in registers a and q")

    op = QubitPauliOperator(
        {QubitPauliString({Qubit("q", 0): Pauli.I}): 1.0}
    )  # This adds identities to all qubits

    b = CuTensorNetBackend()
    c = b.get_compiled_circuit(circ.copy())
    # print(op)
    p0_dict = {Qubit("a", 0): 0}
    p0 = b.get_operator_expectation_value(
        c.copy(), op, p0_dict
    )  # These should save contraction paths
    p1_dict = {Qubit("a", 0): 1}
    p1 = b.get_operator_expectation_value(c.copy(), op, p1_dict)
    return p0 - p1


def general_hadamard_test(circ: Circuit, post_select: dict[Qubit, int]) -> float:
    """Performs a general hadamard test on a circuit.
    The circuit must have a qubit in register a, p and q for lcu circuits.
    The circuit must have a qubit in register a, p, q and s for qsp circuits.
    The tensor network projects on the state |0> in register a and
    whatever is defined in the post selection dict in p (and s) to get p0.
    The tensor network projects on the state |1> in register a
    and whatever is defined in the post selection dict  in p (and s)to get p1.
    The expectation value is p0 - p1.

    Args:
        circ (Circuit): Circuit to perform hadamard test on
        post_select (dict): Dictionary of post selection qubit and value

    Raises:
        ValueError: If circuit does not have qubits in
            registers a, p and q for lcu circuits
        ValueError: If circuit does not have qubits in
            registers a, p, q and s for qsp circuits

    Returns:
        float: Expectation value of hadamard test
    """
    if {qreg.name for qreg in circ.q_registers} != {"a", "p", "q"} and {
        "a",
        "p",
        "q",
        "s",
    }:
        raise ValueError(
            "Circuit must only have qubits in registers a , p and q for \
                  lcu and a ,p, q and s for quantum signal processing circuits"
        )

    op = QubitPauliOperator(
        {QubitPauliString({Qubit("q", 0): Pauli.I}): 1.0}
    )  # This adds identities to all qubits
    b = CuTensorNetBackend()
    c = b.get_compiled_circuit(circ.copy())
    p0_dict = post_select
    p0_dict.update({Qubit("a", 0): 0})
    p0 = b.get_operator_expectation_value(
        c.copy(), op, p0_dict
    )  # These should save contraction paths
    p1_dict = post_select
    p1_dict.update({Qubit("a", 0): 1})
    p1 = b.get_operator_expectation_value(c.copy(), op, p1_dict)
    return p0 - p1
