import pytest
from pytket.circuit import Circuit


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
def q3_v0cx02() -> Circuit:
    circuit = Circuit(3)
    circuit.V(0).CX(0, 2)
    return circuit


@pytest.fixture
def q3_cx01cz12x1rx0() -> Circuit:
    circuit = Circuit(3)
    circuit.CX(0, 1).CZ(1, 2).X(1).Rx(0.3, 0)
    return circuit
