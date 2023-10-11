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
    TTN,
    TTNxGate,
)
from pytket.extensions.cutensornet.utils import circuit_statevector_postselect


def test_libhandle_manager() -> None:
    circ = Circuit(5)

    # Proper use of library handle
    with CuTensorNetHandle() as libhandle:
        ttn = TTN(libhandle, circ.qubits, Config())
        assert np.isclose(ttn.vdot(ttn), 1, atol=ttn._cfg._atol)

    # Catch exception due to library handle out of scope
    with pytest.raises(RuntimeError):
        ttn.vdot(ttn)


def test_init() -> None:
    circ = Circuit(5)

    with CuTensorNetHandle() as libhandle:
        ttn_gate = TTNxGate(libhandle, circ.qubits, Config())
        assert ttn_gate.is_valid()

@pytest.mark.parametrize(
    "circuit",
    [
        # pytest.lazy_fixture("q5_empty"),  # type: ignore
        pytest.lazy_fixture("q8_empty"),  # type: ignore
        pytest.lazy_fixture("q2_x0"),  # type: ignore
        pytest.lazy_fixture("q2_x1"),  # type: ignore
        pytest.lazy_fixture("q2_v0"),  # type: ignore
        pytest.lazy_fixture("q8_x0h2v5z6"),  # type: ignore
        # pytest.lazy_fixture("q2_x0cx01"),  # type: ignore
        # pytest.lazy_fixture("q2_x1cx10x1"),  # type: ignore
        # pytest.lazy_fixture("q2_x0cx01cx10"),  # type: ignore
        # pytest.lazy_fixture("q2_v0cx01cx10"),  # type: ignore
        # pytest.lazy_fixture("q2_hadamard_test"),  # type: ignore
        # pytest.lazy_fixture("q2_lcu1"),  # type: ignore
        # pytest.lazy_fixture("q2_lcu2"),  # type: ignore
        # pytest.lazy_fixture("q2_lcu3"),  # type: ignore
        # pytest.lazy_fixture("q3_v0cx02"),  # type: ignore
        # pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        # pytest.lazy_fixture("q4_lcu1"),  # TTN doesn't support n-qubit gates with n>2
        # pytest.lazy_fixture("q5_h0s1rz2ry3tk4tk13"),  # type: ignore
        # pytest.lazy_fixture("q5_line_circ_30_layers"),  # type: ignore
        # pytest.lazy_fixture("q6_qvol"),  # type: ignore
    ],
)
def test_exact_circ_sim(circuit: Circuit) -> None:
    n_qubits = len(circuit.qubits)
    qubit_partition = {i: q for i, q in enumerate(circuit.qubits)}
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