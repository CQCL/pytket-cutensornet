# Copyright 2019-2024 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
#     http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Any

from pytket.circuit import (
    Op,
    OpType,
    Bit,
    BitRegister,
    SetBitsOp,
    CopyBitsOp,
    RangePredicateOp,
    ClassicalExpBox,
    LogicExp,
    BitWiseOp,
    RegWiseOp,
)


ExtendedLogicExp = Union[LogicExp, Bit, BitRegister, int]


def apply_classical_command(
    op: Op, bits: list[Bit], args: list[Any], bits_dict: dict[Bit, bool]
) -> None:
    """Evaluate classical commands and update the `bits_dict` accordingly."""
    if isinstance(op, SetBitsOp):
        for b, v in zip(bits, op.values):
            bits_dict[b] = v

    elif isinstance(op, CopyBitsOp):
        output_bits = bits
        input_bits = args[: len(output_bits)]
        for i, o in zip(input_bits, output_bits):
            assert isinstance(i, Bit)
            bits_dict[o] = bits_dict[i]

    elif isinstance(op, RangePredicateOp):
        assert len(bits) == 1
        res_bit = bits[0]
        input_bits = args[:-1]
        # The input_bits encode a "value" int in little-endian
        val = from_little_endian([bits_dict[b] for b in input_bits])  # type: ignore
        # Check that the value is in the range
        bits_dict[res_bit] = val >= op.lower and val <= op.upper

    elif isinstance(op, ClassicalExpBox):
        the_exp = op.get_exp()
        result = evaluate_logic_exp(the_exp, bits_dict)

        # The result is an int in little-endian encoding. We update the
        # output register accordingly.
        for b in bits:
            bits_dict[b] = (result % 2) == 1
            result = result >> 1
        assert result == 0  # All bits consumed

    elif op.type == OpType.Barrier:
        pass

    else:
        raise NotImplementedError(f"Commands of type {op.type} are not supported.")


def evaluate_logic_exp(exp: ExtendedLogicExp, bits_dict: dict[Bit, bool]) -> int:
    """Recursive evaluation of a LogicExp."""

    if isinstance(exp, int):
        return exp
    elif isinstance(exp, Bit):
        return 1 if bits_dict[exp] else 0
    elif isinstance(exp, BitRegister):
        return from_little_endian([bits_dict[b] for b in exp])
    else:

        arg_values = [evaluate_logic_exp(arg, bits_dict) for arg in exp.args]

        if exp.op in [BitWiseOp.AND, RegWiseOp.AND]:
            return arg_values[0] & arg_values[1]
        elif exp.op in [BitWiseOp.OR, RegWiseOp.OR]:
            return arg_values[0] | arg_values[1]
        elif exp.op in [BitWiseOp.XOR, RegWiseOp.XOR]:
            return arg_values[0] ^ arg_values[1]
        elif exp.op in [BitWiseOp.EQ, RegWiseOp.EQ]:
            return int(arg_values[0] == arg_values[1])
        elif exp.op in [BitWiseOp.NEQ, RegWiseOp.NEQ]:
            return int(arg_values[0] != arg_values[1])
        elif exp.op == BitWiseOp.NOT:
            return 1 - arg_values[0]
        elif exp.op == BitWiseOp.ZERO:
            return 0
        elif exp.op == BitWiseOp.ONE:
            return 1
        # elif exp.op == RegWiseOp.ADD:
        #     return arg_values[0] + arg_values[1]
        # elif exp.op == RegWiseOp.SUB:
        #     return arg_values[0] - arg_values[1]
        # elif exp.op == RegWiseOp.MUL:
        #     return arg_values[0] * arg_values[1]
        # elif exp.op == RegWiseOp.POW:
        #     return int(arg_values[0] ** arg_values[1])
        # elif exp.op == RegWiseOp.LSH:
        #     return arg_values[0] << arg_values[1]
        elif exp.op == RegWiseOp.RSH:
            return arg_values[0] >> arg_values[1]
        # elif exp.op == RegWiseOp.NEG:
        #     return -arg_values[0]
        else:
            # TODO: Currently not supporting RegWiseOp's DIV, EQ, NEQ, LT, GT, LEQ,
            # GEQ and NOT, since these do not return int, so I am unsure what the
            # semantic is meant to be.
            # TODO: Similarly, it is not clear what to do with overflow of ADD, etc.
            # so I have decided to not support them for now.
            raise NotImplementedError(
                f"Evaluation of {exp.op} not supported in ClassicalExpBox ",
                "by pytket-cutensornet.",
            )


def from_little_endian(bitstring: list[bool]) -> int:
    """Obtain the integer from the little-endian encoded bitstring (i.e. bitstring
    [False, True] is interpreted as the integer 2)."""
    return sum(1 << i for i, b in enumerate(bitstring) if b)
