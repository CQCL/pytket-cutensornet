from typing import List, Union
import warnings
import numpy as np
from numpy.typing import NDArray
import pytest

try:
    import cuquantum as cq  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cuquantum", ImportWarning)
from pytket.circuit import Circuit

from pytket.extensions.cuquantum.tensor_network_convert import (  # type: ignore
    tk_to_tensor_network,
    TensorNetwork,
)


def state_contract(tn: List[Union[NDArray, List]], nqubit: int) -> NDArray:
    """Calls cuQuantum contract function to contract an input state tensor network."""
    state_tn = tn.copy()
    state_tn.append(list(range(1, nqubit + 1)))  # This ensures the right ordering
    state: NDArray = cq.contract(*state_tn).flatten()
    return state


def circuit_overlap_contract(circuit_ket: Circuit) -> float:
    """Calculates an overlap of a state circuit with its adjoint."""
    ket_net = TensorNetwork(circuit_ket)
    overlap_net_interleaved = ket_net.vdot(TensorNetwork(circuit_ket))
    overlap: float = cq.contract(*overlap_net_interleaved)
    return overlap


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
    ],
)
def test_convert_statevec_overlap(circuit: Circuit) -> None:
    tn = tk_to_tensor_network(circuit)
    result_cu = state_contract(tn, circuit.n_qubits).flatten().round(10)
    state_vector = np.array([circuit.get_statevector()])
    assert np.allclose(result_cu, state_vector)
    ovl = circuit_overlap_contract(circuit)
    assert ovl == pytest.approx(1.0)
