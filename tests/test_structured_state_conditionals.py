import pytest
import numpy as np

from pytket.circuit import (
    Bit,
    Circuit,
    OpType,
    Qubit,
    if_not_bit,
    reg_eq,
    RangePredicateOp,
)
from pytket.circuit.logic_exp import BitWiseOp, create_bit_logic_exp

from pytket.extensions.cutensornet.structured_state import (
    CuTensorNetHandle,
    Config,
    simulate,
    SimulationAlgorithm,
)


# These first suite of tests comes from the pytket-qir extension
# (see https://github.com/CQCL/pytket-qir/blob/main/tests/conditional_test.py)
# TODO: Currently, these tests simply check that the operations can be interpreted
# by pytket-cutensornet.
# Further down, there are tests to check that the simulation works correctly.

def test_circuit_with_classicalexpbox_i() -> None:
    # test conditional handling

    circ = Circuit(3)
    a = circ.add_c_register("a", 5)
    b = circ.add_c_register("b", 5)
    c = circ.add_c_register("c", 5)
    d = circ.add_c_register("d", 5)
    circ.H(0)
    circ.add_classicalexpbox_register(a | b, c)  # type: ignore
    circ.add_classicalexpbox_register(c | b, d)  # type: ignore
    circ.add_classicalexpbox_register(c | b, d, condition=a[4])  # type: ignore
    circ.H(0)
    circ.Measure(Qubit(0), d[4])
    circ.H(1)
    circ.Measure(Qubit(1), d[3])
    circ.H(2)
    circ.Measure(Qubit(2), d[2])

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
        assert state.get_fidelity() == 1.0


def test_circuit_with_classicalexpbox_ii() -> None:
    # test conditional handling with else case

    circ = Circuit(3)
    a = circ.add_c_register("a", 5)
    b = circ.add_c_register("b", 5)
    c = circ.add_c_register("c", 5)
    d = circ.add_c_register("d", 5)
    circ.H(0)
    circ.add_classicalexpbox_register(a | b, c)  # type: ignore
    circ.add_classicalexpbox_register(c | b, d)  # type: ignore
    circ.add_classicalexpbox_register(
        c | b, d, condition=if_not_bit(a[4])  # type: ignore
    )
    circ.H(0)
    circ.Measure(Qubit(0), d[4])
    circ.H(1)
    circ.Measure(Qubit(1), d[3])
    circ.H(2)
    circ.Measure(Qubit(2), d[2])

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
        assert state.get_fidelity() == 1.0


def test_circuit_with_classicalexpbox_iii() -> None:
    # test complicated conditions and recursive classical op

    circ = Circuit(2)

    a = circ.add_c_register("a", 15)
    b = circ.add_c_register("b", 15)
    c = circ.add_c_register("c", 15)
    d = circ.add_c_register("d", 15)
    e = circ.add_c_register("e", 15)

    circ.H(0)
    bits = [Bit(i) for i in range(10)]
    big_exp = bits[4] | bits[5] ^ bits[6] | bits[7] & bits[8]
    circ.H(0, condition=big_exp)

    circ.add_classicalexpbox_register(a + b - d, c)  # type: ignore
    circ.add_classicalexpbox_register(a * b * d * c, e)  # type: ignore

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
        assert state.get_fidelity() == 1.0


def test_circuit_with_conditional_gate_i() -> None:
    # test complicated conditions and recursive classical op

    circ = Circuit(2, 2).H(0).H(1).measure_all()

    circ.add_gate(OpType.H, [0], condition_bits=[0, 1], condition_value=3)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
        assert state.get_fidelity() == 1.0


def test_circuit_with_conditional_gate_ii() -> None:
    # test complicated conditions and recursive classical op

    circ = Circuit(2, 3).H(0).H(1).measure_all()

    circ.add_gate(OpType.H, [0], condition_bits=[0, 1, 2], condition_value=3)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
        assert state.get_fidelity() == 1.0


def test_pcircuit_with_conditional_gate_iii() -> None:
    # test conditional for manual added gates

    circ = Circuit(2, 3).H(0).H(1)

    circ.add_gate(
        OpType.PhasedX, [0.1, 0.2], [0], condition_bits=[0, 1, 2], condition_value=3
    )

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
        assert state.get_fidelity() == 1.0


def test_circuit_with_conditional_gate_iv() -> None:
    circ = Circuit(7, name="testcirc")

    syn = circ.add_c_register("syn", 4)

    circ.X(0, condition=reg_eq(syn, 1))
    circ.X(0, condition=reg_eq(syn, 2))
    circ.X(0, condition=reg_eq(syn, 2))
    circ.X(0, condition=reg_eq(syn, 3))
    circ.X(0, condition=reg_eq(syn, 4))
    circ.X(0, condition=reg_eq(syn, 4))

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
        assert state.get_fidelity() == 1.0


# TODO: Currently not supporting CircBox, so these tests are omitted.
#
# def test_pytket_qir_conditional_8() -> None:
#    c = Circuit(4)
#    c.H(0)
#    c.H(1)
#    c.H(2)
#    c.H(3)
#    cbox = CircBox(c)
#    d = Circuit(4)
#    a = d.add_c_register("a", 4)
#    d.add_circbox(cbox, [0, 2, 1, 3], condition=a[0])
#
#    with CuTensorNetHandle() as libhandle:
#        cfg = Config()
#        state = simulate(libhandle, c, SimulationAlgorithm.MPSxGate, cfg)
#        assert state.is_valid()
#        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
#        assert state.get_fidelity() == 1.0
#
#
# def test_pytket_qir_conditional_9() -> None:
#    c = Circuit(4)
#    c.X(0)
#    c.Y(1)
#    c.Z(2)
#    c.H(3)
#    cbox = CircBox(c)
#    d = Circuit(4)
#    a = d.add_c_register("a", 4)
#    d.add_circbox(cbox, [0, 2, 1, 3], condition=a[0])
#
#    with CuTensorNetHandle() as libhandle:
#        cfg = Config()
#        state = simulate(libhandle, c, SimulationAlgorithm.MPSxGate, cfg)
#        assert state.is_valid()
#        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
#        assert state.get_fidelity() == 1.0
#
#
# def test_pytket_qir_conditional_10() -> None:
#    box_circ = Circuit(4)
#    box_circ.X(0)
#    box_circ.Y(1)
#    box_circ.Z(2)
#    box_circ.H(3)
#    box_c = box_circ.add_c_register("c", 5)
#
#    box_circ.H(0)
#    box_circ.add_classicalexpbox_register(box_c | box_c, box_c)  # type: ignore
#
#    cbox = CircBox(box_circ)
#    d = Circuit(4, 5)
#    a = d.add_c_register("a", 4)
#    d.add_circbox(cbox, [0, 2, 1, 3, 0, 1, 2, 3, 4], condition=a[0])
#
#    with CuTensorNetHandle() as libhandle:
#        cfg = Config()
#        state = simulate(libhandle, d, SimulationAlgorithm.MPSxGate, cfg)
#        assert state.is_valid()
#        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
#        assert state.get_fidelity() == 1.0


def test_circuit_with_conditional_gate_v() -> None:
    # test conditional with no register

    circ = Circuit(7, name="testcirc")

    exp = create_bit_logic_exp(BitWiseOp.ONE, [])
    circ.H(0, condition=exp)
    exp2 = create_bit_logic_exp(BitWiseOp.ZERO, [])
    circ.H(0, condition=exp2)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
        assert state.get_fidelity() == 1.0


# The tests below check correctness of the simulator.

def test_correctness_reset_bits() -> None:
    # This circuit does reset on two qubits.
    n_shots = 10

    circ = Circuit(2, 2).H(0).X(1).measure_all()

    circ.add_gate(OpType.X, [0], condition_bits=[0], condition_value=1)
    circ.add_gate(OpType.X, [1], condition_bits=[1], condition_value=1)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()

        for _ in range(n_shots):
            state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
            assert state.is_valid()
            assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
            assert state.get_fidelity() == 1.0
            # The outcome is the |00> state
            assert np.isclose(abs(state.get_amplitude(0)), 1.0)


def test_correctness_reset_register() -> None:
    # Test reset on register, including RangePredicate
    n_shots = 10

    circ = Circuit()

    q_reg = circ.add_q_register("a", 3)
    circ.H(q_reg[0])
    circ.X(q_reg[1])
    circ.Rx(1.5, q_reg[2])

    c_reg = circ.add_c_register("c", 3)
    for q, c in zip(q_reg, c_reg):
        circ.Measure(q, c)

    # Correct the least significant qubit (in an unnecessarily complicated way)
    circ.add_gate(OpType.X, [q_reg[0]], condition_bits=c_reg, condition_value=1)
    circ.add_gate(OpType.X, [q_reg[0]], condition_bits=c_reg, condition_value=3)
    circ.add_gate(OpType.X, [q_reg[0]], condition_bits=c_reg, condition_value=5)
    circ.add_gate(OpType.X, [q_reg[0]], condition_bits=c_reg, condition_value=7)
    # Correct the middle qubit (straightforwad way)
    circ.add_gate(OpType.X, [q_reg[1]], condition_bits=[c_reg[1]], condition_value=1)
    # Correct the last bit using RangePredicateOp to create the flag
    flag = circ.add_c_register("flag", 1)
    circ.add_c_range_predicate(minval=4, maxval=7, args_in=c_reg, arg_out=flag[0])
    circ.add_gate(OpType.X, [q_reg[2]], condition_bits=flag, condition_value=1)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()

        for _ in range(n_shots):
            state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
            assert state.is_valid()
            assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
            assert state.get_fidelity() == 1.0
            # The outcome is the |000> state
            assert np.isclose(abs(state.get_amplitude(0)), 1.0)


def test_correctness_teleportation_bit() -> None:
    # A circuit to teleport a single qubit

    n_shots = 10

    circ = Circuit(3,2)

    # Generate an "interesting" state to be teleported
    circ.Rx(0.42, 0)

    # Generate a Bell state
    circ.H(1)
    circ.CX(1, 2)

    # Apply Bell measurement
    circ.CX(0, 1)
    circ.H(0)
    circ.Measure(0, 0)
    circ.Measure(1, 1)

    # Apply conditional corrections
    circ.add_gate(OpType.Z, [2], condition_bits=[0, 1], condition_value=1)
    circ.add_gate(OpType.X, [2], condition_bits=[0, 1], condition_value=2)
    circ.add_gate(OpType.Y, [2], condition_bits=[0, 1], condition_value=3)

    # Reset the other qubits
    circ.add_gate(OpType.Reset, [0])
    circ.add_gate(OpType.Reset, [1])

    with CuTensorNetHandle() as libhandle:
        cfg = Config()

        for _ in range(n_shots):
            state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
            assert state.is_valid()
            assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)
            assert state.get_fidelity() == 1.0
            # The outcome is cos(0.42*pi/2) |000> - j*sin2(0.42*pi/2) |001>
            assert np.isclose(abs(state.get_amplitude(0))**2, 0.6243, atol=1e-4)
            assert np.isclose(abs(state.get_amplitude(1))**2, 0.3757, atol=1e-4)
