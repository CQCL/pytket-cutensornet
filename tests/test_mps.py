from typing import Any
import random  # type: ignore
import pytest

import cuquantum as cq  # type: ignore
import cupy as cp  # type: ignore
import numpy as np  # type: ignore

from pytket.circuit import Circuit, Qubit, OpType  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.extensions.cutensornet.states import (
    CuTensorNetHandle,
    Config,
    MPS,
    MPSxGate,
    MPSxMPO,
    simulate,
    prepare_circuit,
    ContractionAlg,
)
from pytket.extensions.cutensornet.utils import circuit_statevector_postselect


def test_libhandle_manager() -> None:
    circ = Circuit(5)

    # Proper use of library handle
    with CuTensorNetHandle() as libhandle:
        mps = MPS(libhandle, circ.qubits, Config())
        assert np.isclose(mps.vdot(mps), 1, atol=mps._cfg._atol)

    # Catch exception due to library handle out of scope
    with pytest.raises(RuntimeError):
        mps.vdot(mps)


def test_init() -> None:
    circ = Circuit(5)

    with CuTensorNetHandle() as libhandle:
        mps_gate = MPSxGate(libhandle, circ.qubits, Config())
        assert mps_gate.is_valid()
        mps_mpo = MPSxMPO(libhandle, circ.qubits, Config())
        assert mps_mpo.is_valid()


def test_canonicalise() -> None:
    cp.random.seed(1)
    circ = Circuit(5)

    with CuTensorNetHandle() as libhandle:
        mps_gate = MPSxGate(libhandle, circ.qubits, Config())
        # Fill up the tensors with random entries

        # Leftmost tensor
        T_d = cp.empty(shape=(1, 4, 2), dtype=mps_gate._cfg._complex_t)
        for i1 in range(T_d.shape[1]):
            for i2 in range(T_d.shape[2]):
                T_d[0][i1][i2] = cp.random.rand() + 1j * cp.random.rand()
        mps_gate.tensors[0] = T_d

        # Middle tensors
        for pos in range(1, len(mps_gate) - 1):
            T_d = cp.empty(shape=(4, 4, 2), dtype=mps_gate._cfg._complex_t)
            for i0 in range(T_d.shape[0]):
                for i1 in range(T_d.shape[1]):
                    for i2 in range(T_d.shape[2]):
                        T_d[i0][i1][i2] = cp.random.rand() + 1j * cp.random.rand()
            mps_gate.tensors[pos] = T_d

        # Rightmost tensor
        T_d = cp.empty(shape=(4, 1, 2), dtype=mps_gate._cfg._complex_t)
        for i0 in range(T_d.shape[0]):
            for i2 in range(T_d.shape[2]):
                T_d[i0][0][i2] = cp.random.rand() + 1j * cp.random.rand()
        mps_gate.tensors[len(mps_gate) - 1] = T_d

        assert mps_gate.is_valid()

        # Calculate the norm of the MPS
        norm_sq = mps_gate.vdot(mps_gate)

        # Keep a copy of the non-canonicalised MPS
        mps_copy = mps_gate.copy()

        # Canonicalise around center_pos
        center_pos = 2
        mps_gate.canonicalise(l_pos=center_pos, r_pos=center_pos)

        # Check that canonicalisation did not change the vector
        overlap = mps_gate.vdot(mps_copy)
        assert np.isclose(overlap, norm_sq, atol=mps_gate._cfg._atol)

        # Check that the corresponding tensors are in orthogonal form
        for pos in range(len(mps_gate)):
            if pos == center_pos:  # This needs not be in orthogonal form
                continue

            T_d = mps_gate.tensors[pos]

            if pos < 2:  # Should be in left orthogonal form
                result = cq.contract("lrp,lRp->rR", T_d, T_d.conj())
            elif pos > 2:  # Should be in right orthogonal form
                result = cq.contract("lrp,Lrp->lL", T_d, T_d.conj())

            # Check that the result is the identity
            assert cp.allclose(result, cp.eye(result.shape[0]))


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
    prep_circ, _ = prepare_circuit(circuit)
    n_qubits = len(circuit.qubits)
    state = prep_circ.get_statevector()

    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, prep_circ, algorithm, Config())
        assert mps.is_valid()
        # Check that there was no approximation
        assert np.isclose(mps.fidelity, 1.0, atol=mps._cfg._atol)
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0, atol=mps._cfg._atol)

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            assert np.isclose(
                mps.get_amplitude(b),
                state[b],
                atol=mps._cfg._atol,
            )

        # Check that the statevector is correct
        assert np.allclose(mps.get_statevector(), state, atol=mps._cfg._atol)


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
    prep_circ, _ = prepare_circuit(circuit)
    with CuTensorNetHandle() as libhandle:
        mps = simulate(
            libhandle, prep_circ, algorithm, Config(truncation_fidelity=0.99)
        )
        assert mps.is_valid()
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0, atol=mps._cfg._atol)


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
    prep_circ, _ = prepare_circuit(circuit)
    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, prep_circ, algorithm, Config(chi=4))
        assert mps.is_valid()
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0, atol=mps._cfg._atol)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
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
@pytest.mark.parametrize(
    "fp_precision",
    [
        np.float32,
        np.float64,
    ],
)
def test_float_point_options(
    circuit: Circuit, algorithm: ContractionAlg, fp_precision: Any
) -> None:
    prep_circ, _ = prepare_circuit(circuit)

    with CuTensorNetHandle() as libhandle:
        # Exact
        mps = simulate(
            libhandle, prep_circ, algorithm, Config(float_precision=fp_precision)
        )
        assert mps.is_valid()
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0, atol=mps._cfg._atol)

        # Approximate, bound truncation fidelity
        mps = simulate(
            libhandle,
            prep_circ,
            algorithm,
            Config(truncation_fidelity=0.99, float_precision=fp_precision),
        )
        assert mps.is_valid()
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0, atol=mps._cfg._atol)

        # Approximate, bound chi
        mps = simulate(
            libhandle,
            prep_circ,
            algorithm,
            Config(chi=4, float_precision=fp_precision),
        )
        assert mps.is_valid()
        # Check that overlap is 1
        assert np.isclose(mps.vdot(mps), 1.0, atol=mps._cfg._atol)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q20_line_circ_20_layers"),  # type: ignore
    ],
)
def test_circ_approx_explicit(circuit: Circuit) -> None:
    random.seed(1)

    with CuTensorNetHandle() as libhandle:
        # Finite gate fidelity
        # Check for MPSxGate
        mps_gate = simulate(
            libhandle,
            circuit,
            ContractionAlg.MPSxGate,
            Config(truncation_fidelity=0.99),
        )
        assert np.isclose(mps_gate.fidelity, 0.4, atol=1e-1)
        assert mps_gate.is_valid()
        assert np.isclose(mps_gate.vdot(mps_gate), 1.0, atol=mps_gate._cfg._atol)

        # Check for MPSxMPO
        mps_mpo = simulate(
            libhandle,
            circuit,
            ContractionAlg.MPSxMPO,
            Config(truncation_fidelity=0.99),
        )
        assert np.isclose(mps_mpo.fidelity, 0.6, atol=1e-1)
        assert mps_mpo.is_valid()
        assert np.isclose(mps_mpo.vdot(mps_mpo), 1.0, atol=mps_mpo._cfg._atol)

        # Fixed virtual bond dimension
        # Check for MPSxGate
        mps_gate = simulate(libhandle, circuit, ContractionAlg.MPSxGate, Config(chi=8))
        assert np.isclose(mps_gate.fidelity, 0.03, atol=1e-2)
        assert mps_gate.is_valid()
        assert np.isclose(mps_gate.vdot(mps_gate), 1.0, atol=mps_gate._cfg._atol)

        # Check for MPSxMPO
        mps_mpo = simulate(libhandle, circuit, ContractionAlg.MPSxMPO, Config(chi=8))
        assert np.isclose(mps_mpo.fidelity, 0.04, atol=1e-2)
        assert mps_mpo.is_valid()
        assert np.isclose(mps_mpo.vdot(mps_mpo), 1.0, atol=mps_mpo._cfg._atol)


@pytest.mark.parametrize(
    "circuit",
    [
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
    ],
)
@pytest.mark.parametrize(
    "postselect_dict",
    [
        {Qubit("q", 0): 0},
        {Qubit("q", 0): 1},
        {Qubit("q", 1): 0},
        {Qubit("q", 1): 1},
    ],
)
def test_postselect_2q_circ(circuit: Circuit, postselect_dict: dict) -> None:
    sv = circuit_statevector_postselect(circuit, postselect_dict.copy())
    sv_prob = sv.conj() @ sv
    if not np.isclose(sv_prob, 0.0):
        sv = sv / np.sqrt(sv_prob)  # Normalise

    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, circuit, ContractionAlg.MPSxGate, Config())
        prob = mps.postselect(postselect_dict)
        assert np.isclose(prob, sv_prob, atol=mps._cfg._atol)
        assert np.allclose(mps.get_statevector(), sv, atol=mps._cfg._atol)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "postselect_dict",
    [
        {Qubit("q", 0): 1},
        {Qubit("q", 1): 0},
        {Qubit("q", 0): 0, Qubit("q", 1): 0},
        {Qubit("q", 1): 1, Qubit("q", 2): 1},
        {Qubit("q", 0): 0, Qubit("q", 2): 1},
    ],
)
def test_postselect_circ(circuit: Circuit, postselect_dict: dict) -> None:
    sv = circuit_statevector_postselect(circuit, postselect_dict.copy())
    sv_prob = sv.conj() @ sv
    if not np.isclose(sv_prob, 0.0):
        sv = sv / np.sqrt(sv_prob)  # Normalise

    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, circuit, ContractionAlg.MPSxGate, Config())
        prob = mps.postselect(postselect_dict)
        assert np.isclose(prob, sv_prob, atol=mps._cfg._atol)
        assert np.allclose(mps.get_statevector(), sv, atol=mps._cfg._atol)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01"),  # type: ignore
        pytest.lazy_fixture("q2_x1cx10x1"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q2_lcu1"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q2_lcu3"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "observable",
    [
        QubitPauliString({Qubit(0): Pauli.Z}),
        QubitPauliString({Qubit(1): Pauli.X}),
        QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.Z}),
    ],
)
def test_expectation_value(circuit: Circuit, observable: QubitPauliString) -> None:
    pauli_to_optype = {Pauli.Z: OpType.Z, Pauli.Y: OpType.Z, Pauli.X: OpType.X}

    # Use pytket to generate the expectation value of the observable
    ket_circ = circuit.copy()
    for q, o in observable.map.items():
        ket_circ.add_gate(pauli_to_optype[o], [q])
    ket_sv = ket_circ.get_statevector()

    bra_sv = circuit.get_statevector()

    expectation_value = bra_sv.conj() @ ket_sv

    # Simulate the circuit and obtain the expectation value
    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, circuit, ContractionAlg.MPSxGate, Config())
        assert np.isclose(
            mps.expectation_value(observable), expectation_value, atol=mps._cfg._atol
        )


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01"),  # type: ignore
        pytest.lazy_fixture("q2_v0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
    ],
)
def test_sample_circ_2q(circuit: Circuit) -> None:
    n_samples = 200

    q0 = circuit.qubits[0]
    q1 = circuit.qubits[1]

    # Compute the probabilities of each outcome
    p = dict()
    for outcome in range(4):
        p[outcome] = abs(circuit.get_statevector()[outcome]) ** 2

    # Compute the samples
    sample_dict = {0: 0, 1: 0, 2: 0, 3: 0}
    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, circuit, ContractionAlg.MPSxGate, Config())

        # Take samples measuring both qubits at once
        for _ in range(n_samples):
            outcome_dict = mps.sample()
            outcome = outcome_dict[q0] * 2 + outcome_dict[q1]
            sample_dict[outcome] += 1

    # Check sample frequency consistent with theoretical probability
    for outcome, count in sample_dict.items():
        assert np.isclose(count / n_samples, p[outcome], atol=0.1)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
    ],
)
def test_measure_circ(circuit: Circuit) -> None:
    n_samples = 200

    qA = circuit.qubits[-1]  # Least significant qubit
    qB = circuit.qubits[-3]  # Third list significant qubit

    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, circuit, ContractionAlg.MPSxGate, Config())

        # Compute the probabilities of each outcome
        p = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}
        for outA in range(2):
            for outB in range(2):
                mps_copy = mps.copy()
                p[(outA, outB)] = mps_copy.postselect({qA: outA, qB: outB})

        # Compute the samples
        sample_dict = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for _ in range(n_samples):
            mps_copy = mps.copy()
            outcome_dict = mps_copy.measure({qA, qB})
            sample_dict[(outcome_dict[qA], outcome_dict[qB])] += 1

    # Check sample frequency consistent with theoretical probability
    for outcome, count in sample_dict.items():
        assert np.isclose(count / n_samples, p[outcome], atol=0.1)
