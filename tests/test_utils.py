import pytest
import numpy
from pytket.extensions.cutensornet.general import CuTensorNetHandle, set_logger
from pytket.extensions.cutensornet.general_state.utils import (
    circuit_statevector_postselect,
)
from pytket import Circuit, Qubit  # type: ignore


def test_circuit_statevector_postselect() -> None:
    circ = Circuit(3).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)
    sv = circ.get_statevector()

    n_state_qubits = 2

    sv_post_select = sv[: 2**n_state_qubits]

    post_select_dict = {Qubit(0): 0}

    sv_postselect = circuit_statevector_postselect(circ, post_select_dict)

    numpy.testing.assert_array_equal(sv_postselect, sv_post_select)

    circ = Circuit(3).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)
    sv = circ.get_statevector()

    n_state_qubits = 1

    sv_post_select = sv[: 2**n_state_qubits]

    post_select_dict = {Qubit(0): 0, Qubit(1): 0}

    sv_postselect = circuit_statevector_postselect(circ, post_select_dict)

    numpy.testing.assert_array_equal(sv_postselect, sv_post_select)


def test_device_properties_logger() -> None:
    try:
        with CuTensorNetHandle() as libhandle:
            libhandle.print_device_properties(set_logger("GeneralState", 10))
    except:
        pytest.fail("Could not print device properties")
