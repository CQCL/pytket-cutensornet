import pytest
from pytket.circuit import Circuit, OpType, PauliExpBox, QControlBox, Circuit, CircBox, Qubit
from pytket.pauli import Pauli
from pytket.passes import DecomposeBoxes


def controlled_pauli_gadget_box(paulis,
        rotation
        ):

    circuit_a = Circuit(len(paulis))

    for i, pauli in enumerate(paulis):
        if pauli == Pauli.X:
            circuit_a.H(Qubit('q',i))
        elif pauli == Pauli.Y:
            circuit_a.V(Qubit('q',i))
        else:
            continue

    q_reg = circuit_a.get_q_register('q')
    qubits = list(reversed([q_reg[i] for i in range(len(q_reg))]))

    for q0, q1 in zip(qubits[:-1], qubits[1:]):
        circuit_a.CX(q0, q1)
    
    circuit_b = circuit_a.dagger()

    circuit_rot = Circuit(len(paulis))
    a = circuit_rot.add_q_register('a',1)

    circuit_rot.CRz(rotation, a[0], qubits[-1])
    
    circuit = Circuit()
    circuit.append(circuit_a)
    circuit.append(circuit_rot)
    circuit.append(circuit_b)
    circuit.flatten_registers()
    return CircBox(circuit)


@pytest.fixture
def q2_x0() -> Circuit:
    circuit = Circuit(2)
    circuit.X(0)
    return circuit


@pytest.fixture
def q2_x1() -> Circuit:
    circuit = Circuit(2)
    circuit.X(1)
    return circuit


@pytest.fixture
def q2_v0() -> Circuit:
    circuit = Circuit(2)
    circuit.V(0)
    return circuit


@pytest.fixture
def q2_x0cx01() -> Circuit:
    circuit = Circuit(2)
    circuit.X(0).CX(0, 1)
    return circuit


@pytest.fixture
def q2_x1cx10x1() -> Circuit:
    circuit = Circuit(2)
    circuit.X(1).CX(1, 0).X(1)
    return circuit


@pytest.fixture
def q2_x0cx01cx10() -> Circuit:
    circuit = Circuit(2)
    circuit.X(0).CX(0, 1).CX(1, 0)
    return circuit


@pytest.fixture
def q2_v0cx01cx10() -> Circuit:
    circuit = Circuit(2)
    circuit.V(0).CX(0, 1).CX(1, 0)
    return circuit


@pytest.fixture
def q2_hadamard_test() -> Circuit:
    circuit = Circuit(2)
    circuit.H(0).CRx(0.5, 0, 1).H(0)
    return circuit

@pytest.fixture
def q2_hadamard_test1() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).H(0).CX(0, 1).H(0)
    return circuit


@pytest.fixture
def q2_hadamard_test2() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.48, 1).H(0).CY(0, 1).H(0)
    return circuit


@pytest.fixture
def q2_hadamard_test3() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.12, 1).H(0).CZ(0, 1).H(0)
    return circuit


@pytest.fixture
def q2_hadamard_test4() -> Circuit:
    circ = Circuit(2)
    pbox = PauliExpBox([Pauli.Z], -0.12) # This is failing?
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q3_hadamard_test1() -> Circuit:
    circuit = Circuit(3)
    circuit.Ry(0.78, 1).Ry(0.58, 2).H(0).CX(0, 1).CZ(0, 2).H(0)
    return circuit

@pytest.fixture
def q3_hadamard_test2() -> Circuit:
    circuit = Circuit(3)
    circuit.Ry(0.48, 1).Ry(0.38, 2).H(0).CZ(0, 1).CY(0, 2).H(0)
    return circuit


@pytest.fixture
def q3_hadamard_test3() -> Circuit:
    circuit = Circuit(3)
    circuit.Ry(0.12, 1).Ry(0.68, 2).H(0).CX(0, 2).CZ(0, 1).H(0)
    return circuit

@pytest.fixture
def q3_hadamard_test4() -> Circuit:
    circ = Circuit(3)
    pbox = PauliExpBox([Pauli.X, Pauli.Y], 0.12) # This is failing
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q3_hadamard_test5() -> Circuit:
    circ = Circuit(3)
    pbox = PauliExpBox([Pauli.X, Pauli.Y], -0.12) # This is failing
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2]])
    circ.H(q[0])
    return circ


@pytest.fixture
def q3_hadamard_test6() -> Circuit:
    circ = Circuit(3)
    pbox = PauliExpBox([Pauli.Y, Pauli.X], 0.12) 
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q3_hadamard_test7() -> Circuit:
    circ = Circuit(3)
    pbox = PauliExpBox([Pauli.Y, Pauli.X], -0.12) 
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q3_hadamard_test8() -> Circuit:
    circ = Circuit(3)
    pbox = PauliExpBox([Pauli.X, Pauli.X], 0.12)
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q3_hadamard_test9() -> Circuit:
    circ = Circuit(3)
    pbox = PauliExpBox([Pauli.X, Pauli.X], -0.12) 
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2]])
    circ.H(q[0])
    return circ


@pytest.fixture
def q3_hadamard_test10() -> Circuit:
    circ = Circuit(3)
    q = circ.get_q_register('q')
    qbox = controlled_pauli_gadget_box([Pauli.X, Pauli.Y], 0.12)
    circ.Ry(0.5, q[1]).Ry(0.35, q[2])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q3_hadamard_test11() -> Circuit:
    circ = Circuit(3)
    q = circ.get_q_register('q')
    qbox = controlled_pauli_gadget_box([Pauli.X, Pauli.Y], -0.12)
    circ.Ry(0.5, q[1]).Ry(0.35, q[2])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2]])
    circ.H(q[0])
    return circ


@pytest.fixture
def q4_hadamard_test1() -> Circuit:
    circ = Circuit(4)
    pbox = PauliExpBox([Pauli.Z, Pauli.Z, Pauli.Z], 0.55)
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2]).Ry(0.65, q[3])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2], q[3]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q4_hadamard_test2() -> Circuit:
    circ = Circuit(4)
    pbox = PauliExpBox([Pauli.X, Pauli.Y, Pauli.Z], 0.12) # This is failing?
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2]).Ry(0.65, q[3])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2], q[3]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q4_hadamard_test3() -> Circuit:
    circ = Circuit(4)
    pbox = PauliExpBox([Pauli.X, Pauli.Y, Pauli.Z], -0.12)
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2]).Ry(0.65, q[3])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2], q[3]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q4_hadamard_test4() -> Circuit:
    circ = Circuit(4)
    pbox = PauliExpBox([Pauli.X, Pauli.Y, Pauli.Z], -0.42)
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2]).Ry(0.65, q[3])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2], q[3]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q4_hadamard_test5() -> Circuit:
    circ = Circuit(4)
    pbox = PauliExpBox([Pauli.Z, Pauli.Y, Pauli.Z], 0.32)
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2]).Ry(0.65, q[3])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2], q[3]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q4_hadamard_test6() -> Circuit:
    circ = Circuit(4)
    pbox = PauliExpBox([Pauli.Z, Pauli.Z, Pauli.Z], 0.32)
    qbox = QControlBox(pbox, 1)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[1]).Ry(0.35, q[2]).Ry(0.65, q[3])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2], q[3]])
    circ.H(q[0])
    return circ


@pytest.fixture
def q4_hadamard_test7() -> Circuit:
    circ = Circuit(4)
    q = circ.get_q_register('q')
    qbox = controlled_pauli_gadget_box([Pauli.X, Pauli.Y, Pauli.Z], 0.12)
    circ.Ry(0.5, q[1]).Ry(0.35, q[2]).Ry(0.65, q[3])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2], q[3]])
    circ.H(q[0])
    return circ

@pytest.fixture
def q4_hadamard_test8() -> Circuit:
    circ = Circuit(4)
    q = circ.get_q_register('q')
    qbox = controlled_pauli_gadget_box([Pauli.X, Pauli.Y, Pauli.Z], -0.12)
    circ.Ry(0.5, q[1]).Ry(0.35, q[2]).Ry(0.65, q[3])
    circ.H(q[0])
    circ.add_gate(qbox, [q[0], q[1], q[2], q[3]])
    circ.H(q[0])
    return circ


@pytest.fixture
def q2_lcu1() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Ry(0.27, 0).CX(0, 1).CZ(0, 1).Ry(-0.27, 0)
    return circuit


@pytest.fixture
def q2_lcu2() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Ry(0.27, 0).CZ(0, 1).CY(0, 1).Ry(-0.27, 0)
    return circuit


@pytest.fixture
def q2_lcu3() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Rx(0.67, 0).CX(0, 1).CZ(0, 1).Ry(-0.67, 0)
    return circuit


@pytest.fixture
def q2_lcu1() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Ry(0.27, 0).CX(0, 1).CZ(0, 1).Ry(-0.27, 0)
    return circuit


@pytest.fixture
def q2_lcu2() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Ry(0.27, 0).CZ(0, 1).CY(0, 1).Ry(-0.27, 0)
    return circuit


@pytest.fixture
def q2_lcu3() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Rx(0.67, 0).CX(0, 1).CZ(0, 1).Ry(-0.67, 0)
    return circuit


@pytest.fixture
def q3_v0cx02() -> Circuit:
    circuit = Circuit(3)
    circuit.V(0).CX(0, 2)
    return circuit


@pytest.fixture
def q3_cx01cz12x1rx0() -> Circuit:
    circuit = Circuit(3)
    circuit.CX(0, 1).CZ(1, 2).X(1).Rx(0.3, 0)
    return circuit

@pytest.fixture
def q3_pauli_gadget0() -> Circuit:
    circ = Circuit(3)
    pbox = PauliExpBox([Pauli.X, Pauli.Y, Pauli.Z], 0.12)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[0]).Ry(0.35, q[1]).Ry(0.65, q[2])
    circ.add_gate(pbox, [q[0], q[1], q[2]])
    DecomposeBoxes().apply(circ)
    return circ

@pytest.fixture
def q3_pauli_gadget1() -> Circuit:
    circ = Circuit(3)
    pbox = PauliExpBox([Pauli.X, Pauli.Y, Pauli.Z], -0.12)
    q = circ.get_q_register('q')
    circ.Ry(0.5, q[0]).Ry(0.35, q[1]).Ry(0.65, q[2])
    circ.add_gate(pbox, [q[0], q[1], q[2]])
    DecomposeBoxes().apply(circ)
    return circ


@pytest.fixture
def q4_lcu1() -> Circuit:
    circuit = Circuit(4)
    circuit.Ry(0.78, 3).Ry(0.27, 2).CX(2, 3).Ry(0.58, 2).Ry(0.21, 3)
    circuit.Ry(0.12, 0).Ry(0.56, 1)
    circuit.add_gate(OpType.CnX, [0, 1, 2]).add_gate(OpType.CnX, [0, 1, 3])
    circuit.X(0).X(1).add_gate(OpType.CnY, [0, 1, 2]).add_gate(OpType.CnY, [0, 1, 3]).X(
        0
    ).X(1)
    circuit.Ry(-0.12, 0).Ry(-0.56, 1)
    return circuit
