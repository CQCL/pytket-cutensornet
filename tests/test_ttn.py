import cuquantum as cq  # type: ignore
import cupy as cp  # type: ignore
import numpy as np  # type: ignore
from scipy.stats import unitary_group  # type: ignore

from pytket.circuit import Op, OpType, Circuit, Unitary2qBox  # type: ignore
from pytket.extensions.cuquantum.approximate import Tensor, TTN


def test_init() -> None:
    circ = Circuit(5)

    with TTN(qubits=circ.qubits, chi=8) as ttn:
        assert ttn.is_valid()
