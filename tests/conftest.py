import pytest
from pytket.circuit import Circuit, OpType


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
def q2_lcu1() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Ry(0.27, 0).CX(0, 1).CZ(0,1).Ry(-0.27, 0)
    return circuit

@pytest.fixture
def q2_lcu2() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Ry(0.27, 0).CZ(0, 1).CY(0,1).Ry(-0.27, 0)
    return circuit


@pytest.fixture
def q2_lcu3() -> Circuit:
    circuit = Circuit(2)
    circuit.Ry(0.78, 1).Rx(0.67, 0).CX(0, 1).CZ(0,1).Ry(-0.67, 0)
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
def q4_lcu1() -> Circuit:
    circuit = Circuit(4)
    circuit.Ry(0.78, 3).Ry(0.27, 2).CX(2, 3).Ry(0.58, 2).Ry(0.21, 3) 
    circuit.Ry(0.12, 0).Ry(0.56, 1)
    circuit.add_gate(OpType.CnX, [0 , 1, 2]).add_gate(OpType.CnX, [0 , 1, 3])
    circuit.X(0).X(1).add_gate(OpType.CnY, [0 , 1, 2]).add_gate(OpType.CnY, [0 , 1, 3]).X(0).X(1)
    circuit.Ry(-0.12, 0).Ry(-0.56, 1)
    return circuit
