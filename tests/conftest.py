import pytest
import numpy as np
from scipy.stats import unitary_group  # type: ignore
from pytket.circuit import Circuit, OpType, Unitary2qBox  # type: ignore
from pytket.passes import DecomposeBoxes  # type: ignore


def random_line_circuit(n_qubits: int, layers: int) -> Circuit:
    """Random circuit with line connectivity."""
    c = Circuit(n_qubits)

    for i in range(layers):
        # Layer of TK1 gates
        for q in range(n_qubits):
            c.TK1(np.random.rand(), np.random.rand(), np.random.rand(), q)

        # Layer of CX gates
        offset = np.mod(i, 2)  # Even layers connect (q0,q1), odd (q1,q2)
        qubit_pairs = [
            [c.qubits[i], c.qubits[i + 1]] for i in range(offset, n_qubits - 1, 2)
        ]
        # Direction of each CX gate is random
        for pair in qubit_pairs:
            np.random.shuffle(pair)

        for pair in qubit_pairs:
            c.CX(pair[0], pair[1])

    return c


def quantum_volume_circuit(n_qubits: int) -> Circuit:
    """Random quantum volume circuit."""
    depth = n_qubits
    c = Circuit(n_qubits)

    for _ in range(depth):
        qubits = np.random.permutation([i for i in range(n_qubits)])
        qubit_pairs = [[qubits[i], qubits[i + 1]] for i in range(0, n_qubits - 1, 2)]

        for pair in qubit_pairs:
            # Generate random 4x4 unitary matrix.
            SU4 = unitary_group.rvs(4)  # random unitary in SU4
            SU4 = SU4 / (np.linalg.det(SU4) ** 0.25)
            SU4 = np.matrix(SU4)
            c.add_unitary2qbox(Unitary2qBox(SU4), *pair)

    DecomposeBoxes().apply(c)
    return c


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


@pytest.fixture
def q5_empty() -> Circuit:
    circuit = Circuit(5)
    return circuit


@pytest.fixture
def q5_h0s1rz2ry3tk4tk13() -> Circuit:
    circuit = Circuit(5)
    circuit.H(0)
    circuit.S(1)
    circuit.Rz(0.3, 2)
    circuit.Ry(0.1, 3)
    circuit.TK1(0.2, 0.9, 0.8, 4)
    circuit.TK2(0.6, 0.5, 0.7, 1, 3)
    return circuit


@pytest.fixture
def q5_line_circ_30_layers() -> Circuit:
    np.random.seed(1)
    return random_line_circuit(n_qubits=5, layers=30)


@pytest.fixture
def q20_line_circ_20_layers() -> Circuit:
    np.random.seed(1)
    return random_line_circuit(n_qubits=20, layers=20)


@pytest.fixture
def q6_qvol() -> Circuit:
    np.random.seed(1)
    return quantum_volume_circuit(n_qubits=6)
