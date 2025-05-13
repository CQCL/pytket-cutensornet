import numpy as np
import pytest

from pytket.circuit import (
    Bit,
    CircBox,
    Circuit,
    ClExpr,
    ClOp,
    OpType,
    Qubit,
    WiredClExpr,
    if_not_bit,
    reg_eq,
)
from pytket.circuit.clexpr import wired_clexpr_from_logic_exp
from pytket.circuit.logic_exp import BitWiseOp, create_bit_logic_exp
from pytket.extensions.cutensornet.structured_state import (
    Config,
    CuTensorNetHandle,
    SimulationAlgorithm,
    simulate,
)

# This first suite of tests comes from the pytket-qir extension
# (see https://github.com/CQCL/pytket-qir/blob/main/tests/conditional_test.py)
# Further down, there are tests to check that the simulation works correctly.


def test_circuit_with_clexpr_i() -> None:
    # test conditional handling

    circ = Circuit(3)
    a = circ.add_c_register("a", 5)
    b = circ.add_c_register("b", 5)
    c = circ.add_c_register("c", 5)
    d = circ.add_c_register("d", 5)
    circ.H(0)
    wexpr, args = wired_clexpr_from_logic_exp(a | b, c.to_list())
    circ.add_clexpr(wexpr, args)
    wexpr, args = wired_clexpr_from_logic_exp(c | b, d.to_list())
    circ.add_clexpr(wexpr, args)
    wexpr, args = wired_clexpr_from_logic_exp(c | b, d.to_list())
    circ.add_clexpr(wexpr, args, condition=a[4])
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
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0


def test_circuit_with_clexpr_ii() -> None:
    # test conditional handling with else case

    circ = Circuit(3)
    a = circ.add_c_register("a", 5)
    b = circ.add_c_register("b", 5)
    c = circ.add_c_register("c", 5)
    d = circ.add_c_register("d", 5)
    circ.H(0)
    wexpr, args = wired_clexpr_from_logic_exp(a | b, c.to_list())
    circ.add_clexpr(wexpr, args)
    wexpr, args = wired_clexpr_from_logic_exp(c | b, d.to_list())
    circ.add_clexpr(wexpr, args)
    wexpr, args = wired_clexpr_from_logic_exp(c | b, d.to_list())
    circ.add_clexpr(wexpr, args, condition=if_not_bit(a[4]))
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
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0


@pytest.mark.skip(reason="Currently not supporting arithmetic operations in ClExpr")
def test_circuit_with_clexpr_iii() -> None:
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

    wexpr, args = wired_clexpr_from_logic_exp(a + b - d, c.to_list())
    circ.add_clexpr(wexpr, args)
    wexpr, args = wired_clexpr_from_logic_exp(a * b * d * c, e.to_list())
    circ.add_clexpr(wexpr, args)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0


def test_circuit_with_conditional_gate_i() -> None:
    # test complicated conditions and recursive classical op

    circ = Circuit(2, 2).H(0).H(1).measure_all()

    circ.add_gate(OpType.H, [0], condition_bits=[0, 1], condition_value=3)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0


def test_circuit_with_conditional_gate_ii() -> None:
    # test complicated conditions and recursive classical op

    circ = Circuit(2, 3).H(0).H(1).measure_all()

    circ.add_gate(OpType.H, [0], condition_bits=[0, 1, 2], condition_value=3)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
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
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
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
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0


def test_pytket_basic_conditional_i() -> None:
    c = Circuit(4)
    c.H(0)
    c.H(1)
    c.H(2)
    c.H(3)
    cbox = CircBox(c)
    d = Circuit(4)
    a = d.add_c_register("a", 4)
    d.add_circbox(cbox, [0, 2, 1, 3], condition=a[0])

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, c, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0


def test_pytket_basic_conditional_ii() -> None:
    c = Circuit(4)
    c.X(0)
    c.Y(1)
    c.Z(2)
    c.H(3)
    cbox = CircBox(c)
    d = Circuit(4)
    a = d.add_c_register("a", 4)
    d.add_circbox(cbox, [0, 2, 1, 3], condition=a[0])

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, c, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0


def test_pytket_basic_conditional_iii_clexpr() -> None:
    box_circ = Circuit(4)
    box_circ.X(0)
    box_circ.Y(1)
    box_circ.Z(2)
    box_circ.H(3)
    box_c = box_circ.add_c_register("c", 5)

    box_circ.H(0)

    wexpr, args = wired_clexpr_from_logic_exp(box_c | box_c, box_c.to_list())
    box_circ.add_clexpr(wexpr, args)

    cbox = CircBox(box_circ)
    d = Circuit(4, 5)
    a = d.add_c_register("a", 4)
    d.add_circbox(cbox, [0, 2, 1, 3, 0, 1, 2, 3, 4], condition=a[0])

    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, d, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0


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
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
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
            assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
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
    for q, c in zip(q_reg, c_reg, strict=False):
        circ.Measure(q, c)

    # Correct the least significant qubit (in an unnecessarily complicated way)
    circ.add_gate(OpType.X, [q_reg[0]], condition_bits=c_reg, condition_value=1)
    circ.add_gate(OpType.X, [q_reg[0]], condition_bits=c_reg, condition_value=3)
    circ.add_gate(OpType.X, [q_reg[0]], condition_bits=c_reg, condition_value=5)
    circ.add_gate(OpType.X, [q_reg[0]], condition_bits=c_reg, condition_value=7)
    # Correct the middle qubit (straightforwad way)
    circ.add_gate(OpType.X, [q_reg[1]], condition_bits=[c_reg[1]], condition_value=1)
    # Correct the last bit using RangePredicate to create the flag
    flag = circ.add_c_register("flag", 1)
    circ.add_c_range_predicate(
        minval=4,
        maxval=7,
        args_in=[b for b in c_reg],  # noqa: C416
        arg_out=flag[0],
    )
    circ.add_gate(OpType.X, [q_reg[2]], condition_bits=flag, condition_value=1)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()

        for _ in range(n_shots):
            state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
            assert state.is_valid()
            assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
            assert state.get_fidelity() == 1.0
            # The outcome is the |000> state
            assert np.isclose(abs(state.get_amplitude(0)), 1.0)


def test_correctness_copy_bits() -> None:
    # Dummy circuit where two bits are set and then copied
    circ = Circuit(1)
    orig = circ.add_c_register("orig", 2)
    copied = circ.add_c_register("copied", 2)
    # Set the `orig` bits to 0 and 1
    circ.add_c_setbits([False, True], [orig[0], orig[1]])
    # Copy the bits to the `copied` register
    circ.add_c_copybits([orig[0], orig[1]], [copied[0], copied[1]])
    # Simulate
    with CuTensorNetHandle() as libhandle:
        cfg = Config()
        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
    # Check that the copied register has the correct values
    assert state.get_bits()[copied[0]] == False and state.get_bits()[copied[1]] == True  # noqa: E712


def test_correctness_teleportation_bit() -> None:
    # A circuit to teleport a single qubit

    n_shots = 10

    circ = Circuit(3, 2)

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
            assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
            assert state.get_fidelity() == 1.0
            # The outcome is cos(0.42*pi/2) |000> - j*sin2(0.42*pi/2) |001>
            assert np.isclose(abs(state.get_amplitude(0)) ** 2, 0.6243, atol=1e-4)
            assert np.isclose(abs(state.get_amplitude(1)) ** 2, 0.3757, atol=1e-4)


def test_repeat_until_success_i() -> None:
    # From Figure 8 of https://arxiv.org/pdf/1311.1074

    attempts = 100

    circ = Circuit()
    qin = circ.add_q_register("qin", 1)
    qaux = circ.add_q_register("aux", 1)
    flag = circ.add_c_register("flag", 1)
    circ.add_c_setbits([True], [flag[0]])  # Set flag bit to 1

    for _ in range(attempts):
        circ.add_gate(OpType.Reset, [qaux[0]], condition_bits=flag, condition_value=1)
        circ.add_gate(OpType.H, [qaux[0]], condition_bits=flag, condition_value=1)
        circ.add_gate(OpType.T, [qaux[0]], condition_bits=flag, condition_value=1)
        circ.add_gate(
            OpType.CX, [qaux[0], qin[0]], condition_bits=flag, condition_value=1
        )
        circ.add_gate(OpType.H, [qaux[0]], condition_bits=flag, condition_value=1)
        circ.add_gate(
            OpType.CX, [qaux[0], qin[0]], condition_bits=flag, condition_value=1
        )
        circ.add_gate(OpType.T, [qaux[0]], condition_bits=flag, condition_value=1)
        circ.add_gate(OpType.H, [qaux[0]], condition_bits=flag, condition_value=1)
        circ.Measure(qaux[0], flag[0], condition_bits=flag, condition_value=1)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()

        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0

        # The flag bit should have turned False
        assert not state.get_bits()[flag[0]]
        # The auxiliary qubits should be in state |0>
        prob = state.postselect({qaux[0]: 0})
        assert np.isclose(prob, 1.0)

        target_state = [np.sqrt(1 / 3), np.sqrt(2 / 3) * 1j]
        output_state = state.get_statevector()
        # As indicated in the paper, the gate is implemented up to global phase
        global_phase = target_state[0] / output_state[0]
        assert np.isclose(abs(global_phase), 1.0)
        output_state *= global_phase
        assert np.allclose(target_state, output_state)


def test_repeat_until_success_ii_clexpr() -> None:
    # From Figure 1(c) of https://arxiv.org/pdf/1311.1074

    attempts = 100

    circ = Circuit()
    qin = circ.add_q_register("qin", 1)
    qaux = circ.add_q_register("aux", 2)
    flag = circ.add_c_register("flag", 3)
    circ.add_c_setbits([True, True], [flag[0], flag[1]])  # Set flag bits to 11
    circ.H(qin[0])  # Use to convert gate to sqrt(1/5)*I + i*sqrt(4/5)*X (i.e. Z -> X)

    for _ in range(attempts):
        wexpr, args = wired_clexpr_from_logic_exp(
            flag[0] | flag[1],
            [flag[2]],  # Success if both are zero
        )
        circ.add_clexpr(wexpr, args)

        circ.add_gate(
            OpType.Reset, [qaux[0]], condition_bits=[flag[2]], condition_value=1
        )
        circ.add_gate(
            OpType.Reset, [qaux[1]], condition_bits=[flag[2]], condition_value=1
        )
        circ.add_gate(OpType.H, [qaux[0]], condition_bits=[flag[2]], condition_value=1)
        circ.add_gate(OpType.H, [qaux[1]], condition_bits=[flag[2]], condition_value=1)

        circ.add_gate(OpType.T, [qin[0]], condition_bits=[flag[2]], condition_value=1)
        circ.add_gate(OpType.Z, [qin[0]], condition_bits=[flag[2]], condition_value=1)
        circ.add_gate(
            OpType.Tdg, [qaux[0]], condition_bits=[flag[2]], condition_value=1
        )
        circ.add_gate(
            OpType.CX, [qaux[1], qaux[0]], condition_bits=[flag[2]], condition_value=1
        )
        circ.add_gate(OpType.T, [qaux[0]], condition_bits=[flag[2]], condition_value=1)
        circ.add_gate(
            OpType.CX, [qin[0], qaux[1]], condition_bits=[flag[2]], condition_value=1
        )
        circ.add_gate(OpType.T, [qaux[1]], condition_bits=[flag[2]], condition_value=1)

        circ.add_gate(OpType.H, [qaux[0]], condition_bits=[flag[2]], condition_value=1)
        circ.add_gate(OpType.H, [qaux[1]], condition_bits=[flag[2]], condition_value=1)
        circ.Measure(qaux[0], flag[0], condition_bits=[flag[2]], condition_value=1)
        circ.Measure(qaux[1], flag[1], condition_bits=[flag[2]], condition_value=1)

        # From chat with Silas and exploring the RUS as a block matrix, we have noticed
        # that the circuit is missing an X correction when this condition is satisfied
        wexpr, args = wired_clexpr_from_logic_exp(flag[0] ^ flag[1], [flag[2]])
        circ.add_clexpr(wexpr, args)
        circ.add_gate(OpType.Z, [qin[0]], condition_bits=[flag[2]], condition_value=1)

    circ.H(qin[0])  # Use to convert gate to sqrt(1/5)*I + i*sqrt(4/5)*X (i.e. Z -> X)

    with CuTensorNetHandle() as libhandle:
        cfg = Config()

        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0

        # All of the flag bits should have turned False
        assert all(not state.get_bits()[bit] for bit in flag)
        # The auxiliary qubits should be in state |0>
        prob = state.postselect({qaux[0]: 0, qaux[1]: 0})
        assert np.isclose(prob, 1.0)

        target_state = [np.sqrt(1 / 5), np.sqrt(4 / 5) * 1j]
        output_state = state.get_statevector()
        # As indicated in the paper, the gate is implemented up to global phase
        global_phase = target_state[0] / output_state[0]
        assert np.isclose(abs(global_phase), 1.0)
        output_state *= global_phase
        assert np.allclose(target_state, output_state)


def test_clexpr_on_regs() -> None:
    """Non-exhaustive test on some ClOp on registers."""
    circ = Circuit(2)
    a = circ.add_c_register("a", 5)
    b = circ.add_c_register("b", 5)
    c = circ.add_c_register("c", 5)
    d = circ.add_c_register("d", 5)
    e = circ.add_c_register("e", 5)

    w_expr_regone = WiredClExpr(ClExpr(ClOp.RegOne, []), output_posn=list(range(5)))
    circ.add_clexpr(w_expr_regone, a.to_list())  # a = 0b11111 = 31
    circ.add_c_setbits([True, True, False, False, False], b.to_list())  # b = 3
    circ.add_c_setbits([False, True, False, True, False], c.to_list())  # c = 10
    circ.add_clexpr(*wired_clexpr_from_logic_exp(b | c, d.to_list()))  # d = 11
    circ.add_clexpr(*wired_clexpr_from_logic_exp(a - d, e.to_list()))  # e = 20

    with CuTensorNetHandle() as libhandle:
        cfg = Config()

        state = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
        assert state.is_valid()
        assert np.isclose(state.vdot(state), 1.0, atol=cfg._atol)  # noqa: SLF001
        assert state.get_fidelity() == 1.0

        # Check the bits
        bits_dict = state.get_bits()
        a_bitstring = list(bits_dict[bit] for bit in a)  # noqa: C400
        assert all(a_bitstring)  # a = 0b11111
        b_bitstring = list(bits_dict[bit] for bit in b)  # noqa: C400
        assert b_bitstring == [True, True, False, False, False]  # b = 0b11000
        c_bitstring = list(bits_dict[bit] for bit in c)  # noqa: C400
        assert c_bitstring == [False, True, False, True, False]  # c = 0b01010
        d_bitstring = list(bits_dict[bit] for bit in d)  # noqa: C400
        assert d_bitstring == [True, True, False, True, False]  # d = 0b11010
        e_bitstring = list(bits_dict[bit] for bit in e)  # noqa: C400
        assert e_bitstring == [False, False, True, False, True]  # e = 0b00101
