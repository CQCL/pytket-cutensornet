from pytket.circuit import Circuit, Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator
from pytket.extensions.cuquantum.backends import CuTensorNetBackend

def hadamard_test(circ):

    print({qreg.name for qreg in circ.q_registers})
    if {qreg.name for qreg in circ.q_registers} != {'a', 'q'}:
        raise ValueError("Circuit must only have qubits in registers a and q")

    op = QubitPauliOperator({QubitPauliString({Qubit("q", 0): Pauli.I}): 1.0}) #This adds identities to all qubits

    b = CuTensorNetBackend()
    c = b.get_compiled_circuit(circ.copy())
    # print(op)
    p0_dict = {Qubit("a", 0): 0}
    p0 = b.get_operator_expectation_value_postselect(c.copy(), op, p0_dict) #These should save contraction paths
    print('pooo',p0)
    p1_dict = {Qubit("a", 0): 1}
    p1 = b.get_operator_expectation_value_postselect(c.copy(), op, p1_dict)
    return p0 - p1

def general_hadamard_test(circ, post_select):

    if {qreg.name for qreg in circ.q_registers} != [{"a", "p", "q"}, {"a", "p", "s", "q"}] :
        raise ValueError("Circuit must only have qubits in registers a , p and q for lcu and a ,p, q and s for quantum signal processing circuits")

    # op = QubitPauliOperator({QubitPauliString({Qubit("q", i): Pauli.I}): 1.0})
    op = QubitPauliOperator()
    b = CuTensorNetBackend()
    c = b.get_compiled_circuit(circ.copy())
    p0_dict = post_select.extend({Qubit("a", 0): 0})
    p0 = b.get_operator_expectation_value_postselect(c.copy(), op, p0_dict) #These should save contraction paths
    p1_dict =  post_select.extend({Qubit("a", 0): 1})
    p1 = b.get_operator_expectation_value_postselect(c.copy(), op, p1_dict)
    return p0 - p1