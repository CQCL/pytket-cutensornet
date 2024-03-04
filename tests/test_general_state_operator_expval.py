from typing import List, Union
import warnings
import numpy as np
from numpy.typing import NDArray
import pytest
from pytket.circuit import ToffoliBox, Qubit  # type: ignore
from pytket.passes import DecomposeBoxes, CnXPairwiseDecomposition  # type: ignore
from pytket.transform import Transform  # type: ignore

try:
    import cuquantum as cq  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)
from pytket.circuit import Circuit

from pytket.extensions.cutensornet.general_state import (  # type: ignore
    GeneralState,
    GeneralOperator,
    GeneralExpectationValue,
)
from pytket.extensions.cutensornet.structured_state import CuTensorNetHandle


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
    ],
)
def test_convert_statevec(circuit: Circuit) -> None:
    sv = None
    with CuTensorNetHandle() as libhandle:
        state = GeneralState(circuit, libhandle)
        sv = state.configure().prepare().compute()
        state.destroy()
    sv_pytket = np.array([circuit.get_statevector()])
    assert np.allclose(sv.round(10), sv_pytket.round(10))
    # ovl = circuit_overlap_contract(circuit)
    # assert ovl == pytest.approx(1.0)
