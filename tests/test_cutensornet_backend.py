import numpy as np
import pytest
from pytket.circuit import Circuit, BasisOrder, ToffoliBox, DummyBox, OpType  # type: ignore
from pytket.passes import CliffordSimp  # type: ignore
from pytket.pauli import QubitPauliString, Pauli  # type: ignore
from pytket.utils.operators import QubitPauliOperator
from pytket import Qubit  # type: ignore
from pytket.extensions.cutensornet.backends import (
    CuTensorNetStateBackend,
    CuTensorNetShotsBackend,
)


def test_bell() -> None:
    c = Circuit(2)
    c.H(0)
    c.CX(0, 1)
    b = CuTensorNetStateBackend()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c)
    assert np.allclose(
        b.get_result(h).get_state(), np.asarray([1, 0, 0, 1]) * 1 / np.sqrt(2)
    )


def test_sampler_bell() -> None:
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    b = CuTensorNetShotsBackend()
    c = b.get_compiled_circuit(c)
    res = b.run_circuit(c, n_shots=10, seed=3)
    assert res.get_shots().shape == (10, 2)
    assert res.get_counts() == {(0, 0): 5, (1, 1): 5}


def test_basisorder() -> None:
    c = Circuit(2)
    c.X(1)
    b = CuTensorNetStateBackend()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c)
    r = b.get_result(h)
    assert np.allclose(r.get_state(), np.asarray([0, 1, 0, 0]))
    assert np.allclose(r.get_state(basis=BasisOrder.dlo), np.asarray([0, 0, 1, 0]))


def test_sampler_basisorder() -> None:
    c = Circuit(2, 2)
    c.X(1)
    c.measure_all()
    b = CuTensorNetShotsBackend()
    c = b.get_compiled_circuit(c)
    res = b.run_circuit(c, n_shots=10, seed=0)
    assert res.get_counts() == {(0, 1): 10}
    assert res.get_counts(basis=BasisOrder.dlo) == {(1, 0): 10}


def test_implicit_perm() -> None:
    c = Circuit(2)
    c.CX(0, 1)
    c.CX(1, 0)
    c.Ry(0.1, 1)
    c1 = c.copy()
    CliffordSimp().apply(c1)
    b = CuTensorNetStateBackend()
    c = b.get_compiled_circuit(c, optimisation_level=1)
    c1 = b.get_compiled_circuit(c1, optimisation_level=1)
    assert c.implicit_qubit_permutation() != c1.implicit_qubit_permutation()
    h, h1 = b.process_circuits([c, c1])
    r, r1 = b.get_results([h, h1])
    for bo in [BasisOrder.ilo, BasisOrder.dlo]:
        s = r.get_state(basis=bo)
        s1 = r1.get_state(basis=bo)
        assert np.allclose(s, s1)


def test_compilation_pass() -> None:
    perm = {
        (False, False): (True, True),
        (False, True): (False, False),
        (True, False): (True, False),
        (True, True): (False, True),
    }

    b = CuTensorNetStateBackend()
    for opt_level in range(3):
        c = Circuit(2)
        c.CX(0, 1)
        u = np.asarray([[0, 1], [-1j, 0]])
        c.add_toffolibox(ToffoliBox(perm), [0, 1])  # type: ignore
        c.CX(0, 1)
        c.add_gate(OpType.CRz, 0.35, [1, 0])

        assert b.valid_circuit(c)  # Already valid before decomposition
        c = b.get_compiled_circuit(c, optimisation_level=opt_level)
        assert b.valid_circuit(c)


def test_unitarity_predicate() -> None:
    b = CuTensorNetStateBackend()
    c = Circuit(2)
    c.CX(0, 1)
    c.add_dummybox(DummyBox(2, 0, c.get_resources()), [0, 1], [])  # type: ignore
    c.CX(0, 1)
    c.add_gate(OpType.CRz, 0.35, [1, 0])
    assert not b.valid_circuit(c)


def test_invalid_measures() -> None:
    c = Circuit(2)
    c.H(0).CX(0, 1)
    c.measure_all()  # Invalid mid-circuit measurement
    c.H(0)
    b = CuTensorNetStateBackend()
    c = b.get_compiled_circuit(c)
    assert not (b.valid_circuit(c))


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
        pytest.lazy_fixture("q3_v0cx02"),  # type: ignore
        pytest.lazy_fixture("q3_cx01cz12x1rx0"),  # type: ignore
        pytest.lazy_fixture("q4_lcu1"),  # type: ignore
        pytest.lazy_fixture("q4_multicontrols"),  # type: ignore
        pytest.lazy_fixture("q4_with_creates"),  # type: ignore
    ],
)
def test_compile_convert_statevec(circuit: Circuit) -> None:
    b = CuTensorNetStateBackend()
    c = b.get_compiled_circuit(circuit)
    h = b.process_circuit(c)
    assert np.allclose(
        b.get_result(h).get_state(), np.array([circuit.get_statevector()])
    )
