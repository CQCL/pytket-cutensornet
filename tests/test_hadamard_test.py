import numpy as np
import pytest
# from pytket.extensions.qiskit import AerStateBackend
from pytket.extensions.qulacs import QulacsBackend
from pytket.circuit import Circuit, Qubit
from pytket.extensions.cuquantum.backends import CuTensorNetBackend
from pytket.utils import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.extensions.cuquantum.backends.hadamard_test import hadamard_test

def gen_hadamard_test_post_select(dist, n_prep):
    return {k: v for k, v in dist.items() if k[1:n_prep+1] == tuple(np.zeros(n_prep).tolist())}

def generalised_hadamard_test(dist, n_prep):
    ps_dist = gen_hadamard_test_post_select(dist, n_prep)
    p0 = sum([v for k, v in ps_dist.items() if k[0] == 0])
    p1 = sum([v for k, v in ps_dist.items() if k[0] == 1])
    return (p0-p1)/(p0+p1) # Do you need renorm?

def hadamard_test_postprocess(dist):
    p0 = sum([v for k, v in dist.items() if k[0] == 0])
    p1 = sum([v for k, v in dist.items() if k[0] == 1])
    return p0-p1 # Do you need renorm?

@pytest.mark.parametrize(
    "ht_circuit",
    [
        pytest.lazy_fixture("q2_hadamard_test_reg_names"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test1"),
        pytest.lazy_fixture("q2_hadamard_test2"),
        pytest.lazy_fixture("q2_hadamard_test3"),
        pytest.lazy_fixture("q2_hadamard_test4"),
        pytest.lazy_fixture("q3_hadamard_test1"),
        pytest.lazy_fixture("q3_hadamard_test2"),
        pytest.lazy_fixture("q3_hadamard_test3"),
        pytest.lazy_fixture("q3_hadamard_test4"),
        pytest.lazy_fixture("q3_hadamard_test5"),
        pytest.lazy_fixture("q3_hadamard_test6"),
        pytest.lazy_fixture("q3_hadamard_test7"),
        pytest.lazy_fixture("q3_hadamard_test8"),
        pytest.lazy_fixture("q3_hadamard_test9"),
        pytest.lazy_fixture("q3_hadamard_test10"),
        pytest.lazy_fixture("q3_hadamard_test11"),
        pytest.lazy_fixture("q4_hadamard_test1"),
        pytest.lazy_fixture("q4_hadamard_test2"),
        pytest.lazy_fixture("q4_hadamard_test3"),
        pytest.lazy_fixture("q4_hadamard_test4"),
        pytest.lazy_fixture("q4_hadamard_test5"),
        pytest.lazy_fixture("q4_hadamard_test6"),
        pytest.lazy_fixture("q4_hadamard_test7"),
        pytest.lazy_fixture("q4_hadamard_test8"),
    ],
)

def test_hadamard_test(ht_circuit: Circuit) -> None:

    backend = QulacsBackend()
    compiled_circ = backend.get_compiled_circuit(ht_circuit.copy())
    dist = backend.run_circuit(compiled_circ).get_probability_distribution().as_dict()
    qulacs_exp = hadamard_test_postprocess(dist)
    cu_exp = hadamard_test(ht_circuit)
    assert np.isclose(qulacs_exp, cu_exp, atol=1e-10)





