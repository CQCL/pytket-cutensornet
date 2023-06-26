import random  # type: ignore
import pytest

import cuquantum as cq  # type: ignore
import cupy as cp  # type: ignore
import numpy as np  # type: ignore

from pytket.circuit import Circuit  # type: ignore
from pytket.extensions.cutensornet.mps import (
    Tensor,
    MPSxGate,
    MPSxMPO,
    simulate,
    get_amplitude,
    ContractionAlg,
)


def test_init() -> None:
    circ = Circuit(5)

    mps_gate = MPSxGate(qubits=circ.qubits)
    with mps_gate.init_cutensornet():
        assert mps_gate.is_valid()

    mps_mpo = MPSxMPO(qubits=circ.qubits)
    with mps_mpo.init_cutensornet():
        assert mps_mpo.is_valid()


def test_canonicalise() -> None:
    np.random.seed(1)
    circ = Circuit(5)

    mps_gate = MPSxGate(qubits=circ.qubits)
    with mps_gate.init_cutensornet():
        # Fill up the tensors with random entries

        # Leftmost tensor
        T_d = cp.empty(shape=(4, 2), dtype=mps_gate._complex_t)
        for i0 in range(T_d.shape[0]):
            for i1 in range(T_d.shape[1]):
                T_d[i0][i1] = np.random.rand() + 1j * np.random.rand()
        mps_gate.tensors[0] = Tensor(T_d, bonds=[1, len(mps_gate)])

        # Middle tensors
        for pos in range(1, len(mps_gate) - 1):
            T_d = cp.empty(shape=(4, 4, 2), dtype=mps_gate._complex_t)
            for i0 in range(T_d.shape[0]):
                for i1 in range(T_d.shape[1]):
                    for i2 in range(T_d.shape[2]):
                        T_d[i0][i1][i2] = np.random.rand() + 1j * np.random.rand()
            mps_gate.tensors[pos] = Tensor(
                T_d, bonds=[pos, pos + 1, pos + len(mps_gate)]
            )

        # Rightmost tensor
        T_d = cp.empty(shape=(4, 2), dtype=mps_gate._complex_t)
        for i0 in range(T_d.shape[0]):
            for i1 in range(T_d.shape[1]):
                T_d[i0][i1] = np.random.rand() + 1j * np.random.rand()
        mps_gate.tensors[len(mps_gate) - 1] = Tensor(
            T_d, bonds=[len(mps_gate) - 1, 2 * len(mps_gate) - 1]
        )

        mps_copy = mps_gate.copy()
        # Calculate the norm of the MPS
        norm_sq = mps_gate.vdot(mps_copy)

        # Canonicalise around center_pos
        center_pos = 2
        mps_gate.canonicalise(l_pos=center_pos, r_pos=center_pos)

        # Check that canonicalisation did not change the vector
        overlap = mps_gate.vdot(mps_copy)
        assert np.isclose(overlap, norm_sq)

        # Check that the corresponding tensors are in orthogonal form
        for pos in range(len(mps_gate)):
            if pos == center_pos:  # This needs not be in orthogonal form
                continue

            T_d = mps_gate.tensors[pos].data
            std_bonds = list(mps_gate.tensors[pos].bonds)
            conj_bonds = list(mps_gate.tensors[pos].bonds)

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


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01"),  # type: ignore
        pytest.lazy_fixture("q2_x1cx10x1"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_v0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q2_lcu1"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q2_lcu3"),  # type: ignore
        pytest.lazy_fixture("q3_v0cx02"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        # pytest.lazy_fixture("q4_lcu1"),  # MPS doesn't support n-qubit gates with n>2
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        ContractionAlg.MPSxGate,
        ContractionAlg.MPSxMPO,
    ],
)
def test_exact_circ_sim(circuit: Circuit, algorithm: ContractionAlg) -> None:
    state = circuit.get_statevector()
    n_qubits = len(circuit.qubits)

    mps = simulate(circuit, algorithm)
    with mps.init_cutensornet():
        assert mps.is_valid()
        # Check that there was no approximation
        assert np.isclose(mps.fidelity, 1.0)
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0)

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            assert np.isclose(
                np.exp(1j * np.pi * circuit.phase) * get_amplitude(mps, b), state[b]
            )


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01"),  # type: ignore
        pytest.lazy_fixture("q2_x1cx10x1"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_v0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q2_lcu1"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q2_lcu3"),  # type: ignore
        pytest.lazy_fixture("q3_v0cx02"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        # pytest.lazy_fixture("q4_lcu1"),  # MPS doesn't support n-qubit gates with n>2
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        ContractionAlg.MPSxGate,
        ContractionAlg.MPSxMPO,
    ],
)
def test_approx_circ_sim_gate_fid(circuit: Circuit, algorithm: ContractionAlg) -> None:
    mps = simulate(circuit, algorithm, truncation_fidelity=0.99)
    with mps.init_cutensornet():
        assert mps.is_valid()
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01"),  # type: ignore
        pytest.lazy_fixture("q2_x1cx10x1"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_v0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q2_lcu1"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q2_lcu3"),  # type: ignore
        pytest.lazy_fixture("q3_v0cx02"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        # pytest.lazy_fixture("q4_lcu1"),  # MPS doesn't support n-qubit gates with n>2
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        ContractionAlg.MPSxGate,
        ContractionAlg.MPSxMPO,
    ],
)
def test_approx_circ_sim_chi(circuit: Circuit, algorithm: ContractionAlg) -> None:
    mps = simulate(circuit, algorithm, chi=4)
    with mps.init_cutensornet():
        assert mps.is_valid()
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q20_line_circ_20_layers"),  # type: ignore
    ],
)
def test_circ_approx_explicit(circuit: Circuit) -> None:
    random.seed(1)

    # Finite gate fidelity
    # Check for MPSxGate
    mps_gate = simulate(circuit, ContractionAlg.MPSxGate, truncation_fidelity=0.99)
    assert np.isclose(mps_gate.fidelity, 0.45129, atol=1e-5)

    with mps_gate.init_cutensornet():
        assert mps_gate.is_valid()
        assert np.isclose(mps_gate.vdot(mps_gate), 1.0)

    # Check for MPSxMPO
    mps_mpo = simulate(circuit, ContractionAlg.MPSxMPO, truncation_fidelity=0.99)
    assert np.isclose(mps_mpo.fidelity, 0.66917, atol=1e-5)

    with mps_mpo.init_cutensornet():
        assert mps_mpo.is_valid()
        assert np.isclose(mps_mpo.vdot(mps_mpo), 1.0)

    # Fixed virtual bond dimension
    # Check for MPSxGate
    mps_gate = simulate(circuit, ContractionAlg.MPSxGate, chi=8)
    assert np.isclose(mps_gate.fidelity, 0.05830, atol=1e-5)

    with mps_gate.init_cutensornet():
        assert mps_gate.is_valid()
        assert np.isclose(mps_gate.vdot(mps_gate), 1.0)

    # Check for MPSxMPO
    mps_mpo = simulate(circuit, ContractionAlg.MPSxMPO, chi=8)
    assert np.isclose(mps_mpo.fidelity, 0.08466, atol=1e-5)

    with mps_mpo.init_cutensornet():
        assert mps_mpo.is_valid()
        assert np.isclose(mps_mpo.vdot(mps_mpo), 1.0)
