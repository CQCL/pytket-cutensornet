from typing import Any, Union
import random  # type: ignore
import math
import pytest

import cuquantum as cq  # type: ignore
import cupy as cp  # type: ignore
import numpy as np  # type: ignore

from pytket.circuit import Circuit, Qubit, OpType  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.extensions.cutensornet.tnstate import (
    CuTensorNetHandle,
    Config,
    TTN,
    TTNxGate,
    DirTTN,
)
from pytket.extensions.cutensornet.tnstate.ttn import RootPath
from pytket.extensions.cutensornet.utils import circuit_statevector_postselect


def test_libhandle_manager() -> None:
    circ = Circuit(4)
    qubit_partition = {i: [q] for i, q in enumerate(circ.qubits)}

    # Proper use of library handle
    with CuTensorNetHandle() as libhandle:
        ttn = TTN(libhandle, qubit_partition, Config())
        assert np.isclose(ttn.vdot(ttn), 1, atol=ttn._cfg._atol)

    # Catch exception due to library handle out of scope
    with pytest.raises(RuntimeError):
        ttn.vdot(ttn)


def test_init() -> None:
    circ = Circuit(4)
    qubit_partition = {i: [q] for i, q in enumerate(circ.qubits)}

    with CuTensorNetHandle() as libhandle:
        ttn_gate = TTNxGate(libhandle, qubit_partition, Config())
        assert ttn_gate.is_valid()


@pytest.mark.parametrize(
    "center",
    [
        (DirTTN.RIGHT,),
        (DirTTN.LEFT, DirTTN.RIGHT),
        (DirTTN.LEFT, DirTTN.RIGHT, DirTTN.RIGHT),
        Qubit("q", [2]),
    ],
)
def test_canonicalise(center: Union[RootPath, Qubit]) -> None:
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
                T = cp.empty(shape=(2, max_dim), dtype=ttn._cfg._complex_t)
                for i0 in range(T.shape[0]):
                    for i1 in range(T.shape[1]):
                        T[i0][i1] = cp.random.rand() + 1j * cp.random.rand()
            else:
                shape = (max_dim, max_dim, max_dim if len(path) != 0 else 1)
                T = cp.empty(shape=shape, dtype=ttn._cfg._complex_t)
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
        assert np.isclose(overlap / norm_sq, 1.0, atol=ttn._cfg._atol)

        # Check that the tensor R returned agrees with the norm
        overlap_R = cq.contract("ud,ud->", R, R.conj())
        assert np.isclose(overlap_R / norm_sq, 1.0, atol=ttn._cfg._atol)

        # Check that the corresponding tensors are in orthogonal form
        for path, node in ttn.nodes.items():
            # If it's the node just below the center of canonicalisation, it
            # cannot be in orthogonal form
            if isinstance(center, Qubit):
                if path == ttn.qubit_position[center][0]:
                    assert node.canonical_form is None
                    continue
            else:
                if path == center[:-1]:
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


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q8_empty"),  # type: ignore
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
        # pytest.lazy_fixture("q4_lcu1"),  # TTN doesn't support n-qubit gates with n>2
        pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        pytest.lazy_fixture("q6_qvol"),  # type: ignore
    ],
)
def test_exact_circ_sim(circuit: Circuit) -> None:
    n_qubits = len(circuit.qubits)
    n_groups = 2**math.floor(math.log2(n_qubits))
    qubit_partition: dict[int, list[Qubit]] = {i: [] for i in range(n_groups)}
    for i, q in enumerate(circuit.qubits):
        qubit_partition[i % n_groups].append(q)

    state = circuit.get_statevector()

    with CuTensorNetHandle() as libhandle:
        ttn = TTNxGate(libhandle, qubit_partition, Config())
        for g in circuit.get_commands():
            ttn.apply_gate(g)

        assert ttn.is_valid()
        # Check that there was no approximation
        assert np.isclose(ttn.fidelity, 1.0, atol=ttn._cfg._atol)
        # Check that overlap is 1
        assert np.isclose(ttn.vdot(ttn), 1.0, atol=ttn._cfg._atol)

        # Check that all of the amplitudes are correct
        for b in range(2**n_qubits):
            assert np.isclose(
                ttn.get_amplitude(b),
                state[b],
                atol=ttn._cfg._atol,
            )

        # Check that the statevector is correct
        assert np.allclose(ttn.get_statevector(), state, atol=ttn._cfg._atol)
