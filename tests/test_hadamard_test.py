from typing import Any
import numpy as np
import pytest

# from pytket.extensions.qiskit import AerStateBackend
from pytket.extensions.qulacs import QulacsBackend #type: ignore
from pytket.circuit import Circuit #type: ignore
from pytket.extensions.cutensornet.backends.hadamard_test import (
    hadamard_test,
    general_hadamard_test,
)


def gen_hadamard_test_post_select(dist: dict, n_prep: int) -> dict:
    return {
        k: v
        for k, v in dist.items()
        if k[1 : n_prep + 1] == tuple(np.zeros(n_prep).tolist())
    }


def generalised_hadamard_test(dist: dict, n_prep: int) -> Any:
    ps_dist = gen_hadamard_test_post_select(dist, n_prep)
    p0 = sum([v for k, v in ps_dist.items() if k[0] == 0])
    p1 = sum([v for k, v in ps_dist.items() if k[0] == 1])
    return p0 - p1  # Do you need renorm?


def hadamard_test_postprocess(dist: dict) -> Any:
    p0 = sum([v for k, v in dist.items() if k[0] == 0])
    p1 = sum([v for k, v in dist.items() if k[0] == 1])
    return p0 - p1  # Do you need renorm?


@pytest.mark.parametrize(
    "ht_circuit",
    [
        pytest.lazy_fixture("q2_hadamard_test_reg_names"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test1"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test2"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test3"),  # type: ignore
        pytest.lazy_fixture("q2_hadamard_test4"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test1"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test2"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test3"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test4"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test5"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test6"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test7"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test8"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test9"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test10"),  # type: ignore
        pytest.lazy_fixture("q3_hadamard_test11"),  # type: ignore
        pytest.lazy_fixture("q4_hadamard_test1"),  # type: ignore
        pytest.lazy_fixture("q4_hadamard_test2"),  # type: ignore
        pytest.lazy_fixture("q4_hadamard_test3"),  # type: ignore
        pytest.lazy_fixture("q4_hadamard_test4"),  # type: ignore
        pytest.lazy_fixture("q4_hadamard_test5"),  # type: ignore
        pytest.lazy_fixture("q4_hadamard_test6"),  # type: ignore
        pytest.lazy_fixture("q4_hadamard_test7"),  # type: ignore
        pytest.lazy_fixture("q4_hadamard_test8"),  # type: ignore
    ],
)
def test_hadamard_test(ht_circuit: Circuit) -> None:
    backend = QulacsBackend()
    compiled_circ = backend.get_compiled_circuit(ht_circuit.copy())
    dist = backend.run_circuit(compiled_circ).get_probability_distribution().as_dict()
    qulacs_exp = hadamard_test_postprocess(dist)
    cu_exp = hadamard_test(ht_circuit)
    assert np.isclose(qulacs_exp, cu_exp, atol=1e-10)


@pytest.mark.parametrize(
    "circuit",
    [
        pytest.lazy_fixture("q3_lcu_hadamardtest0"),  # type: ignore
        pytest.lazy_fixture("q3_lcu_hadamardtest1"),  # type: ignore
        pytest.lazy_fixture("q3_lcu_hadamardtest2"),  # type: ignore
        pytest.lazy_fixture("q4_lcu_hadamard_test"),  # type: ignore
        pytest.lazy_fixture("q5_lcu_hadamard_test0"),  # type: ignore
        # TODO Write a test for this
        # comparing get_probability_distribution and get_distrobition
        pytest.lazy_fixture("q5_lcu_hadamard_test1"),  # type: ignore
        pytest.lazy_fixture("q6_lcu_hadamard_test"),  # type: ignore
    ],
)
def test_lcu_hadamard_test(circuit):
    backend = QulacsBackend()
    compiled_circ = backend.get_compiled_circuit(circuit.copy())
    dist = backend.run_circuit(compiled_circ).get_probability_distribution().as_dict()
    p_reg = circuit.get_q_register("p")
    post_select = {p: 0 for p in p_reg}
    qulacs_exp = generalised_hadamard_test(dist, len(p_reg))
    cu_exp = general_hadamard_test(circuit, post_select)
    assert np.isclose(qulacs_exp, cu_exp, atol=1e-10)
