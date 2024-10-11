import random  # type: ignore
from typing import Any

import cupy as cp  # type: ignore
import cuquantum as cq  # type: ignore
import numpy as np  # type: ignore
import pytest

from pytket.circuit import Circuit, OpType, Qubit  # type: ignore
from pytket.extensions.cutensornet.general_state.utils import (
    circuit_statevector_postselect,
)
from pytket.extensions.cutensornet.structured_state import (
    MPS,
    Config,
    CuTensorNetHandle,
    DirTTN,
    LowFidelityException,
    MPSxGate,
    MPSxMPO,
    SimulationAlgorithm,
    TTNxGate,
    prepare_circuit_mps,
    simulate,
)
from pytket.extensions.cutensornet.structured_state.ttn import RootPath
from pytket.pauli import Pauli, QubitPauliString  # type: ignore


def test_libhandle_manager() -> None:
    circ = Circuit(5)

    # Proper use of library handle
    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        mps = MPS(libhandle, circ.qubits, cfg)
        assert np.isclose(mps.vdot(mps), 1, atol=cfg._atol)  # noqa: SLF001

    # Catch exception due to library handle out of scope
    with pytest.raises(RuntimeError):
        mps.vdot(mps)


def test_init() -> None:
    circ = Circuit(8)
    qubit_partition = {i: [q] for i, q in enumerate(circ.qubits)}

    with CuTensorNetHandle() as libhandle:
        mps_gate = MPSxGate(libhandle, circ.qubits, Config())
        assert mps_gate.is_valid()
        mps_mpo = MPSxMPO(libhandle, circ.qubits, Config())
        assert mps_mpo.is_valid()
        ttn_gate = TTNxGate(libhandle, qubit_partition, Config())
        assert ttn_gate.is_valid()


@pytest.mark.parametrize(
    "algorithm",
    [
        SimulationAlgorithm.MPSxGate,
        SimulationAlgorithm.MPSxMPO,
        SimulationAlgorithm.TTNxGate,
    ],
)
def test_copy(algorithm: SimulationAlgorithm) -> None:
    simple_circ = Circuit(2).H(0).H(1).CX(0, 1)

    with CuTensorNetHandle() as libhandle:
        # Default config
        cfg = Config()
        state = simulate(libhandle, simple_circ, algorithm, cfg)
        assert state.is_valid()
        copy_state = state.copy()
        assert copy_state.is_valid()
        assert np.isclose(copy_state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Bounded chi
        cfg = Config(chi=8)
        state = simulate(libhandle, simple_circ, algorithm, cfg)
        assert state.is_valid()
        copy_state = state.copy()
        assert copy_state.is_valid()
        assert np.isclose(copy_state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Bounded truncation_fidelity
        cfg = Config(truncation_fidelity=0.9999)
        state = simulate(libhandle, simple_circ, algorithm, cfg)
        assert state.is_valid()
        copy_state = state.copy()
        assert copy_state.is_valid()
        assert np.isclose(copy_state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001


def test_canonicalise_mps() -> None:
    cp.random.seed(1)
    circ = Circuit(5)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        mps_gate = MPSxGate(libhandle, circ.qubits, cfg)
        # Fill up the tensors with random entries

        # Leftmost tensor
        T_d = cp.empty(shape=(1, 4, 2), dtype=cfg._complex_t)  # noqa: SLF001
        for i1 in range(T_d.shape[1]):
            for i2 in range(T_d.shape[2]):
                T_d[0][i1][i2] = cp.random.rand() + 1j * cp.random.rand()
        mps_gate.tensors[0] = T_d

        # Middle tensors
        for pos in range(1, len(mps_gate) - 1):
            T_d = cp.empty(shape=(4, 4, 2), dtype=cfg._complex_t)  # noqa: SLF001
            for i0 in range(T_d.shape[0]):
                for i1 in range(T_d.shape[1]):
                    for i2 in range(T_d.shape[2]):
                        T_d[i0][i1][i2] = cp.random.rand() + 1j * cp.random.rand()
            mps_gate.tensors[pos] = T_d

        # Rightmost tensor
        T_d = cp.empty(shape=(4, 1, 2), dtype=cfg._complex_t)  # noqa: SLF001
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
        assert np.isclose(overlap, norm_sq, atol=cfg._atol)  # noqa: SLF001

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
    "center",
    [
        (DirTTN.RIGHT,),
        (DirTTN.LEFT, DirTTN.RIGHT),
        (DirTTN.LEFT, DirTTN.RIGHT, DirTTN.RIGHT),
        Qubit("q", [2]),
    ],
)
def test_canonicalise_ttn(center: RootPath | Qubit) -> None:  # noqa: PLR0912
    cp.random.seed(1)
    n_levels = 3
    n_qubits = 2**n_levels
    max_dim = 8

    circ = Circuit(n_qubits)
    qubit_partition = {i: [q] for i, q in enumerate(circ.qubits)}

    with CuTensorNetHandle() as libhandle:
        ttn = TTNxGate(libhandle, qubit_partition, Config())

        # Fill up the tensors with random entries
        for path, node in ttn.nodes.items():
            if node.is_leaf:
                T = cp.empty(shape=(2, max_dim), dtype=ttn._cfg._complex_t)  # noqa: SLF001
                for i0 in range(T.shape[0]):
                    for i1 in range(T.shape[1]):
                        T[i0][i1] = cp.random.rand() + 1j * cp.random.rand()
            else:
                shape = (max_dim, max_dim, max_dim if len(path) != 0 else 1)
                T = cp.empty(shape=shape, dtype=ttn._cfg._complex_t)  # noqa: SLF001
                for i0 in range(shape[0]):
                    for i1 in range(shape[1]):
                        for i2 in range(shape[2]):
                            T[i0][i1][i2] = cp.random.rand() + 1j * cp.random.rand()
            node.tensor = T

        assert ttn.is_valid()

        # Calculate the norm of the TTN
        norm_sq = ttn.vdot(ttn)

        # Keep a copy of the non-canonicalised TTN
        ttn_copy = ttn.copy()

        # Canonicalise at target path
        R = ttn.canonicalise(center)
        assert ttn.is_valid()

        # Check that canonicalisation did not change the vector
        overlap = ttn.vdot(ttn_copy)
        assert np.isclose(overlap / norm_sq, 1.0, atol=ttn._cfg._atol)  # noqa: SLF001

        # Check that the tensor R returned agrees with the norm
        overlap_R = cq.contract("ud,ud->", R, R.conj())
        assert np.isclose(overlap_R / norm_sq, 1.0, atol=ttn._cfg._atol)  # noqa: SLF001

        # Check that the corresponding tensors are in orthogonal form
        for path, node in ttn.nodes.items():
            # If it's the node just below the center of canonicalisation, it
            # cannot be in orthogonal form
            if isinstance(center, Qubit):
                if path == ttn.qubit_position[center][0]:
                    assert node.canonical_form is None
                    continue
            elif path == center[:-1]:
                assert node.canonical_form is None
                continue
            # Otherwise, it should be in orthogonal form
            assert node.canonical_form is not None

            T = node.tensor

            if node.is_leaf:
                assert node.canonical_form == DirTTN.PARENT
                result = cq.contract("qp,qP->pP", T, T.conj())

            elif node.canonical_form == DirTTN.PARENT:
                result = cq.contract("lrp,lrP->pP", T, T.conj())

            elif node.canonical_form == DirTTN.LEFT:
                result = cq.contract("lrp,Lrp->lL", T, T.conj())

            elif node.canonical_form == DirTTN.RIGHT:
                result = cq.contract("lrp,lRp->rR", T, T.conj())

            # Check that the result is the identity
            assert cp.allclose(result, cp.eye(result.shape[0]))


def test_entanglement_entropy():
    circ = Circuit(4)
    circ.H(0).CX(0,1)
    circ.H(2).T(2).H(2).CX(2,3)

    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, Config())

        assert np.isclose(mps.get_entanglement_entropy(0), -np.log(0.5))
        assert np.isclose(mps.get_entanglement_entropy(1), 0)
        assert np.isclose(mps.get_entanglement_entropy(2), 0.4165, atol=0.0001)

@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q1_empty"),  # type: ignore
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q8_empty"),  # type: ignore
        pytest.lazy_fixture("q1_h0rz"),  # type: ignore
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q8_x0h2v5z6"),  # type: ignore
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
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
        pytest.lazy_fixture("q8_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        SimulationAlgorithm.MPSxGate,
        SimulationAlgorithm.MPSxMPO,
        SimulationAlgorithm.TTNxGate,
    ],
)
def test_exact_circ_sim(circuit: Circuit, algorithm: SimulationAlgorithm) -> None:
    n_qubits = len(circuit.qubits)
    state_vec = circuit.get_statevector()

    with CuTensorNetHandle() as libhandle:
        cfg = Config(leaf_size=2)
        state = simulate(libhandle, circuit, algorithm, cfg)
        assert state.is_valid()
        # Check that there was no approximation
        assert np.isclose(state.get_fidelity(), 1.0, atol=cfg._atol)  # noqa: SLF001
        # Check that overlap is 1
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            assert np.isclose(
                state.get_amplitude(b),
                state_vec[b],
                atol=cfg._atol,  # noqa: SLF001
            )

        # Check that the statevector is correct
        assert np.allclose(state.get_statevector(), state_vec, atol=cfg._atol)  # noqa: SLF001


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q1_empty"),  # type: ignore
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q1_h0rz"),  # type: ignore
        pytest.lazy_fixture("q2_lcu1"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q2_lcu3"),  # type: ignore
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
        pytest.lazy_fixture("q8_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        SimulationAlgorithm.MPSxGate,
        SimulationAlgorithm.MPSxMPO,
    ],
)
def test_prepare_circuit_mps(circuit: Circuit, algorithm: SimulationAlgorithm) -> None:
    state_vec = circuit.get_statevector()
    n_qubits = len(circuit.qubits)

    # Prepare the circuit (i.e. add SWAPs so that all gates act on adjacent qubits)
    circuit, qubit_map = prepare_circuit_mps(circuit)
    # Check that the qubit adjacency is satisfied
    for cmd in circuit.get_commands():
        qs = cmd.qubits
        assert len(qs) in {1, 2}
        if len(qs) == 2:
            assert abs(qs[0].index[0] - qs[1].index[0]) == 1

    with CuTensorNetHandle() as libhandle:
        cfg = Config(leaf_size=2)
        state = simulate(libhandle, circuit, algorithm, cfg)
        state.apply_qubit_relabelling(qubit_map)
        assert state.is_valid()
        # Check that there was no approximation
        assert np.isclose(state.get_fidelity(), 1.0, atol=cfg._atol)  # noqa: SLF001
        # Check that overlap is 1
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            assert np.isclose(
                state.get_amplitude(b),
                state_vec[b],
                atol=cfg._atol,  # noqa: SLF001
            )

        # Check that the statevector is correct
        assert np.allclose(state.get_statevector(), state_vec, atol=cfg._atol)  # noqa: SLF001


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q1_empty"),  # type: ignore
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q8_empty"),  # type: ignore
        pytest.lazy_fixture("q1_h0rz"),  # type: ignore
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q8_x0h2v5z6"),  # type: ignore
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
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
        pytest.lazy_fixture("q8_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        SimulationAlgorithm.MPSxGate,
        SimulationAlgorithm.MPSxMPO,
        SimulationAlgorithm.TTNxGate,
    ],
)
def test_approx_circ_sim_gate_fid(
    circuit: Circuit, algorithm: SimulationAlgorithm
) -> None:
    with CuTensorNetHandle() as libhandle:
        cfg = Config(truncation_fidelity=0.99, leaf_size=2)
        state = simulate(libhandle, circuit, algorithm, cfg)
        assert state.is_valid()
        # Check that overlap is 1
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q8_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        SimulationAlgorithm.MPSxGate,
        SimulationAlgorithm.MPSxMPO,
        SimulationAlgorithm.TTNxGate,
    ],
)
def test_kill_threshold(circuit: Circuit, algorithm: SimulationAlgorithm) -> None:
    with CuTensorNetHandle() as libhandle:
        cfg = Config(truncation_fidelity=0.99, kill_threshold=0.9999, leaf_size=2)
        with pytest.raises(LowFidelityException):
            simulate(libhandle, circuit, algorithm, cfg)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q1_empty"),  # type: ignore
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q8_empty"),  # type: ignore
        pytest.lazy_fixture("q1_h0rz"),  # type: ignore
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q8_x0h2v5z6"),  # type: ignore
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
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
        pytest.lazy_fixture("q8_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        SimulationAlgorithm.MPSxGate,
        SimulationAlgorithm.MPSxMPO,
        SimulationAlgorithm.TTNxGate,
    ],
)
def test_approx_circ_sim_chi(circuit: Circuit, algorithm: SimulationAlgorithm) -> None:
    with CuTensorNetHandle() as libhandle:
        cfg = Config(chi=4, leaf_size=2)
        state = simulate(libhandle, circuit, algorithm, cfg)
        assert state.is_valid()
        # Check that overlap is 1
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q1_empty"),  # type: ignore
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q1_h0rz"),  # type: ignore
        pytest.lazy_fixture("q2_x0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        SimulationAlgorithm.MPSxGate,
        SimulationAlgorithm.MPSxMPO,
        SimulationAlgorithm.TTNxGate,
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
    circuit: Circuit, algorithm: SimulationAlgorithm, fp_precision: Any
) -> None:
    with CuTensorNetHandle() as libhandle:
        # Exact
        cfg = Config(float_precision=fp_precision, leaf_size=2)
        state = simulate(libhandle, circuit, algorithm, cfg)
        assert state.is_valid()
        # Check that overlap is 1
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Approximate, bound truncation fidelity
        cfg = Config(
            truncation_fidelity=0.99, float_precision=fp_precision, leaf_size=2
        )
        state = simulate(
            libhandle,
            circuit,
            algorithm,
            cfg,
        )
        assert state.is_valid()
        # Check that overlap is 1
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Approximate, bound chi
        cfg = Config(chi=4, float_precision=fp_precision, leaf_size=2)
        state = simulate(
            libhandle,
            circuit,
            algorithm,
            cfg,
        )
        assert state.is_valid()
        # Check that overlap is 1
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q20_line_circ_20_layers"),  # type: ignore
    ],
)
def test_circ_approx_explicit_mps(circuit: Circuit) -> None:
    random.seed(1)

    with CuTensorNetHandle() as libhandle:
        # Finite gate fidelity
        # Check for MPSxGate
        cfg = Config(truncation_fidelity=0.99, leaf_size=4, float_precision=np.float32)
        mps_gate = simulate(
            libhandle,
            circuit,
            SimulationAlgorithm.MPSxGate,
            cfg,
        )
        assert mps_gate.get_fidelity() >= 0.3
        assert mps_gate.is_valid()
        assert np.isclose(mps_gate.vdot(mps_gate), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Check for MPSxMPO
        mps_mpo = simulate(
            libhandle,
            circuit,
            SimulationAlgorithm.MPSxMPO,
            cfg,
        )
        assert mps_mpo.get_fidelity() >= 0.5
        assert mps_mpo.is_valid()
        assert np.isclose(mps_mpo.vdot(mps_mpo), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Fixed virtual bond dimension
        # Check for MPSxGate
        cfg = Config(chi=8, leaf_size=4, float_precision=np.float32)
        mps_gate = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, cfg)
        assert mps_gate.get_fidelity() >= 0.02
        assert mps_gate.is_valid()
        assert np.isclose(mps_gate.vdot(mps_gate), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Check for MPSxMPO
        mps_mpo = simulate(libhandle, circuit, SimulationAlgorithm.MPSxMPO, cfg)
        assert mps_mpo.get_fidelity() >= 0.04
        assert mps_mpo.is_valid()
        assert np.isclose(mps_mpo.vdot(mps_mpo), 1.0, atol=cfg._atol)  # noqa: SLF001


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q15_qvol"),  # type: ignore
    ],
)
def test_circ_approx_explicit_ttn(circuit: Circuit) -> None:
    random.seed(1)

    with CuTensorNetHandle() as libhandle:
        # Finite gate fidelity
        # Check for TTNxGate
        cfg = Config(truncation_fidelity=0.99, leaf_size=3, float_precision=np.float32)
        ttn_gate = simulate(libhandle, circuit, SimulationAlgorithm.TTNxGate, cfg)
        assert ttn_gate.get_fidelity() >= 0.75
        assert ttn_gate.is_valid()
        assert np.isclose(ttn_gate.vdot(ttn_gate), 1.0, atol=cfg._atol)  # noqa: SLF001

        # Fixed virtual bond dimension
        # Check for TTNxGate
        cfg = Config(chi=120, leaf_size=3, float_precision=np.float32)
        ttn_gate = simulate(libhandle, circuit, SimulationAlgorithm.TTNxGate, cfg)
        assert ttn_gate.get_fidelity() >= 0.85
        assert ttn_gate.is_valid()
        assert np.isclose(ttn_gate.vdot(ttn_gate), 1.0, atol=cfg._atol)  # noqa: SLF001


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
        cfg = Config()
        mps = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, cfg)
        prob = mps.postselect(postselect_dict)
        assert np.isclose(prob, sv_prob, atol=cfg._atol)  # noqa: SLF001
        assert np.allclose(mps.get_statevector(), sv, atol=cfg._atol)  # noqa: SLF001


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
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
        cfg = Config()

        mps = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, cfg)

        prob = mps.postselect(postselect_dict)
        assert np.isclose(prob, sv_prob, atol=cfg._atol)  # noqa: SLF001
        assert np.allclose(mps.get_statevector(), sv, atol=cfg._atol)  # noqa: SLF001


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
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
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
        cfg = Config()
        mps = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, cfg)
        assert np.isclose(
            mps.expectation_value(observable),
            expectation_value,
            atol=cfg._atol,  # noqa: SLF001
        )


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q1_h0rz"),  # type: ignore
        pytest.lazy_fixture("q2_v0cx01cx10"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
    ],
)
def test_sample_with_seed(circuit: Circuit) -> None:
    n_samples = 10
    config = Config(seed=1234)

    with CuTensorNetHandle() as libhandle:
        mps_0 = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, config)
        mps_1 = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, config)
        mps_2 = mps_0.copy()

        all_outcomes = []
        for _ in range(n_samples):
            # Check that all copies of the MPS result in the same sample
            outcomes_0 = mps_0.sample()
            outcomes_1 = mps_1.sample()
            outcomes_2 = mps_2.sample()
            assert outcomes_0 == outcomes_1 and outcomes_0 == outcomes_2

            all_outcomes.append(outcomes_0)

        # Check that the outcomes change between different samples
        assert not all(outcome == outcomes_0 for outcome in all_outcomes)


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
    p = dict()  # noqa: C408
    for outcome in range(4):
        p[outcome] = abs(circuit.get_statevector()[outcome]) ** 2

    # Compute the samples
    sample_dict = {0: 0, 1: 0, 2: 0, 3: 0}
    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, Config())

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
        pytest.lazy_fixture("q3_toffoli_box_with_implicit_swaps"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
    ],
)
def test_measure_circ(circuit: Circuit) -> None:
    n_samples = 200

    qA = circuit.qubits[-1]  # Least significant qubit
    qB = circuit.qubits[-3]  # Third list significant qubit

    with CuTensorNetHandle() as libhandle:
        mps = simulate(libhandle, circuit, SimulationAlgorithm.MPSxGate, Config())

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


def test_measure_non_destructive_singleton() -> None:
    # Example of failing test from issue #191
    with CuTensorNetHandle() as libhandle:
        config = Config()
        mps = MPSxGate(
            libhandle,
            qubits=[Qubit(0)],
            config=config,
        )

        result = mps.measure({Qubit(0)}, destructive=False)
        assert result[Qubit(0)] == 0
        sv = mps.get_statevector()
        assert sv[0] == 1 and len(sv) == 2


def test_mps_qubit_addition_and_measure() -> None:
    with CuTensorNetHandle() as libhandle:
        config = Config()
        mps = MPSxGate(
            libhandle,
            qubits=[Qubit(0), Qubit(1), Qubit(2), Qubit(3)],
            config=config,
        )

        x = cp.asarray(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=config._complex_t,  # noqa: SLF001
        )
        cx = cp.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=config._complex_t,  # noqa: SLF001
        )

        # Apply some gates
        mps.apply_unitary(x, [Qubit(1)])  # |0100>
        mps.apply_unitary(cx, [Qubit(1), Qubit(2)])  # |0110>
        mps.apply_unitary(cx, [Qubit(2), Qubit(3)])  # |0111>
        # Add a qubit at the end of the MPS
        mps.add_qubit(new_qubit=Qubit(4), position=len(mps))  # |01110>
        # Apply some more gates acting on the new qubit
        mps.apply_unitary(cx, [Qubit(3), Qubit(4)])  # |01111>
        mps.apply_unitary(cx, [Qubit(4), Qubit(3)])  # |01101>
        # Add a qubit at position 3
        mps.add_qubit(new_qubit=Qubit(6), position=3)  # |011001>
        # Apply some more gates acting on the new qubit
        mps.apply_unitary(x, [Qubit(6)])  # |011101>
        mps.apply_unitary(cx, [Qubit(6), Qubit(2)])  # |010101>
        mps.apply_unitary(cx, [Qubit(6), Qubit(3)])  # |010111>
        # Add another qubit at the end of the MPS
        mps.add_qubit(new_qubit=Qubit(5), position=len(mps), state=1)  # |0101111>
        # Apply some more gates acting on the new qubit
        mps.apply_unitary(cx, [Qubit(4), Qubit(5)])  # |0101110>

        # The resulting state should be |0101110>
        sv = np.zeros(2**7)
        sv[int("0101110", 2)] = 1

        # However, since mps.get_statevector will sort qubits in ILO, the bits would
        # change position. Instead, we can relabel the qubits.
        mps.apply_qubit_relabelling(
            {q: Qubit(i) for q, i in mps.qubit_position.items()}
        )

        # Compare the state vectors
        assert np.allclose(mps.get_statevector(), sv)

        # Measure some of the qubits destructively
        outcomes = mps.measure({Qubit(0), Qubit(2), Qubit(4)}, destructive=True)
        # Since the state is |0101110>, the outcomes are deterministic
        assert outcomes[Qubit(0)] == 0
        assert outcomes[Qubit(2)] == 0
        assert outcomes[Qubit(4)] == 1

        # Note that the qubit identifiers have not been updated,
        # so the qubits that were measured are no longer in the MPS.
        with pytest.raises(ValueError, match="not a qubit in the MPS"):
            mps.measure({Qubit(0)})

        # Measure some of the remaining qubits non-destructively
        outcomes = mps.measure({Qubit(1), Qubit(6)}, destructive=False)
        assert outcomes[Qubit(1)] == 1
        assert outcomes[Qubit(6)] == 0

        # The resulting state should be |1110>, verify it
        sv = np.zeros(2**4)
        sv[int("1110", 2)] = 1
        assert np.allclose(mps.get_statevector(), sv)

        # Apply a few more gates to check it works
        mps.apply_unitary(x, [Qubit(1)])  # |0110>
        mps.apply_unitary(cx, [Qubit(3), Qubit(5)])  # |0100>

        # The resulting state should be |0100>, verify it
        sv = np.zeros(2**4)
        sv[int("0100", 2)] = 1
        assert np.allclose(mps.get_statevector(), sv)
