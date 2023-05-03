import cuquantum as cq  # type: ignore
import cupy as cp  # type: ignore
import numpy as np  # type: ignore

from pytket.circuit import Op, OpType, Circuit  # type: ignore
from pytket.extensions.cuquantum.mps import Tensor, MPSxGate, MPSxMPO


def test_init() -> None:
    with MPSxGate(n_tensors=5, chi=8) as mps:
        assert mps.is_valid()

    with MPSxMPO(n_tensors=5, chi=8) as mps:
        assert mps.is_valid()


def test_trivial_vdot() -> None:
    with MPSxGate(n_tensors=5, chi=8) as mps:
        assert mps.is_valid()
        assert np.isclose(mps.vdot(mps), 1.0)

    with MPSxMPO(n_tensors=5, chi=8) as mps:
        assert mps.is_valid()
        assert np.isclose(mps.vdot(mps), 1.0)


def test_1q_gates() -> None:
    n_qubits = 5
    circ = Circuit(n_qubits)
    circ.H(0)
    circ.S(1)
    circ.Rz(0.3, 2)
    circ.Ry(0.1, 3)
    circ.TK1(0.6, 0.5, 0.7, 4)
    unitary = circ.get_unitary()

    qubit_pos = {q: i for i, q in enumerate(circ.qubits)}

    with MPSxGate(n_tensors=n_qubits, chi=2) as mps:
        # Apply each of the single qubit gates
        for g in circ.get_commands():
            q = g.qubits[0]
            mps.apply_1q_gate(qubit_pos[q], g.op)
        assert mps.is_valid()

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            with MPSxGate(n_tensors=n_qubits, chi=2) as b_mps:
                bitstring = format(b, f"0{n_qubits}b")
                for i in range(n_qubits):
                    if bitstring[i] == "1":
                        b_mps.apply_1q_gate(i, Op.create(OpType.X))
                assert b_mps.is_valid()

                # Check the amplitude
                assert np.isclose(b_mps.vdot(mps), unitary[b][0])

    with MPSxMPO(n_tensors=n_qubits, chi=2) as mps:
        # Apply each of the single qubit gates
        for g in circ.get_commands():
            q = g.qubits[0]
            mps.apply_1q_gate(qubit_pos[q], g.op)
        assert mps.is_valid()

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            with MPSxMPO(n_tensors=n_qubits, chi=2) as b_mps:
                bitstring = format(b, f"0{n_qubits}b")
                for i in range(n_qubits):
                    if bitstring[i] == "1":
                        b_mps.apply_1q_gate(i, Op.create(OpType.X))
                assert b_mps.is_valid()

                # Check the amplitude
                assert np.isclose(b_mps.vdot(mps), unitary[b][0])


def test_canonicalise() -> None:
    np.random.seed(1)

    with MPSxGate(n_tensors=5, chi=4) as mps:
        # Fill up the tensors with random entries

        # Leftmost tensor
        T_d = cp.empty(shape=(4, 2), dtype=mps._complex_t)
        for i0 in range(T_d.shape[0]):
            for i1 in range(T_d.shape[1]):
                T_d[i0][i1] = np.random.rand() + 1j * np.random.rand()
        mps.tensors[0] = Tensor(T_d, bonds=[1, len(mps)])

        # Middle tensors
        for pos in range(1, len(mps) - 1):
            T_d = cp.empty(shape=(4, 4, 2), dtype=mps._complex_t)
            for i0 in range(T_d.shape[0]):
                for i1 in range(T_d.shape[1]):
                    for i2 in range(T_d.shape[2]):
                        T_d[i0][i1][i2] = np.random.rand() + 1j * np.random.rand()
            mps.tensors[pos] = Tensor(T_d, bonds=[pos, pos + 1, pos + len(mps)])

        # Rightmost tensor
        T_d = cp.empty(shape=(4, 2), dtype=mps._complex_t)
        for i0 in range(T_d.shape[0]):
            for i1 in range(T_d.shape[1]):
                T_d[i0][i1] = np.random.rand() + 1j * np.random.rand()
        mps.tensors[len(mps) - 1] = Tensor(T_d, bonds=[len(mps) - 1, 2 * len(mps) - 1])

        with mps.copy() as mps_copy:
            # Calculate the norm of the MPS
            norm_sq = mps.vdot(mps_copy)

            # Canonicalise around center_pos
            center_pos = 2
            mps.canonicalise(l_pos=center_pos, r_pos=center_pos)

            # Check that canonicalisation did not change the vector
            overlap = mps.vdot(mps_copy)
            assert np.isclose(overlap, norm_sq)

        # Check that the corresponding tensors are in orthogonal form
        for pos in range(len(mps)):
            if pos == center_pos:  # This needs not be in orthogonal form
                continue

            T_d = mps.tensors[pos].data
            std_bonds = list(mps.tensors[pos].bonds)
            conj_bonds = list(mps.tensors[pos].bonds)

            if pos < 2:  # Should be in left orthogonal form
                std_bonds[-2] = -1
                conj_bonds[-2] = -2
            elif pos > 2:  # Should be in right orthogonal form
                std_bonds[0] = -1
                conj_bonds[0] = -2

            result = cq.contract(T_d, std_bonds, T_d.conj(), conj_bonds, [-1, -2])

            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if i == j:
                        assert np.isclose(result[i][j], 1)
                    else:
                        assert np.isclose(result[i][j], 0)


def test_line_circ_exact() -> None:
    # Simulate a circuit with only nearest neighbour interactions
    np.random.seed(1)
    n_qubits = 5
    layers = 30

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

    unitary = c.get_unitary()
    qubit_pos = {q: i for i, q in enumerate(c.qubits)}

    # EXACT CONTRACTION (chi=4 is enough for n_qubits=5)
    # Check that all of the amplitudes are correct

    # Check for MPSxGate
    with MPSxGate(n_tensors=n_qubits, chi=4) as mps:
        # Apply each of the gates
        for g in c.get_commands():
            if len(g.qubits) == 1:
                q = g.qubits[0]
                mps.apply_1q_gate(qubit_pos[q], g.op)
            else:
                q0 = qubit_pos[g.qubits[0]]
                q1 = qubit_pos[g.qubits[1]]
                mps.apply_2q_gate((q0, q1), g.op)
        assert mps.is_valid()

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            with MPSxGate(n_tensors=n_qubits, chi=2) as b_mps:
                bitstring = format(b, f"0{n_qubits}b")
                for i in range(n_qubits):
                    if bitstring[i] == "1":
                        b_mps.apply_1q_gate(i, Op.create(OpType.X))
                assert b_mps.is_valid()

                # Check the amplitudes are similar
                assert np.isclose(b_mps.vdot(mps), unitary[b][0])
                assert np.isclose(mps.fidelity, 1.0)

    # Check for MPSxMPO
    with MPSxMPO(n_tensors=n_qubits, chi=4) as mps:
        # Apply each of the gates
        for g in c.get_commands():
            if len(g.qubits) == 1:
                q = g.qubits[0]
                mps.apply_1q_gate(qubit_pos[q], g.op)
            else:
                q0 = qubit_pos[g.qubits[0]]
                q1 = qubit_pos[g.qubits[1]]
                mps.apply_2q_gate((q0, q1), g.op)
        assert mps.is_valid()

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            with MPSxMPO(n_tensors=n_qubits, chi=2) as b_mps:
                bitstring = format(b, f"0{n_qubits}b")
                for i in range(n_qubits):
                    if bitstring[i] == "1":
                        b_mps.apply_1q_gate(i, Op.create(OpType.X))
                assert b_mps.is_valid()

                # Check the amplitudes are similar
                assert np.isclose(b_mps.vdot(mps), unitary[b][0])
                assert np.isclose(mps.fidelity, 1.0)


def test_line_circ_approx() -> None:
    # Simulate a circuit with only nearest neighbour interactions
    np.random.seed(1)
    n_qubits = 30
    layers = 30

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

    qubit_pos = {q: i for i, q in enumerate(c.qubits)}

    # APPROXIMATE CONTRACTION (chi=8 is insufficient for exact)

    # Check for MPSxGate
    with MPSxGate(n_tensors=n_qubits, chi=8) as mps:
        # Apply each of the gates
        for g in c.get_commands():
            if len(g.qubits) == 1:
                q = g.qubits[0]
                mps.apply_1q_gate(qubit_pos[q], g.op)
            else:
                q0 = qubit_pos[g.qubits[0]]
                q1 = qubit_pos[g.qubits[1]]
                mps.apply_2q_gate((q0, q1), g.op)
        assert mps.is_valid()
        assert np.isclose(mps.fidelity, 0.000130, atol=1e-6)

        # Check that that the state has norm 1
        assert np.isclose(mps.vdot(mps), 1.0)

    # Check for MPSxMPO
    with MPSxMPO(n_tensors=n_qubits, chi=8) as mps:
        # Apply each of the gates
        for g in c.get_commands():
            if len(g.qubits) == 1:
                q = g.qubits[0]
                mps.apply_1q_gate(qubit_pos[q], g.op)
            else:
                q0 = qubit_pos[g.qubits[0]]
                q1 = qubit_pos[g.qubits[1]]
                mps.apply_2q_gate((q0, q1), g.op)
        assert mps.is_valid()
        assert np.isclose(mps.fidelity, 0.000268, atol=1e-6)

        # Check that that the state has norm 1
        assert np.isclose(mps.vdot(mps), 1.0)
