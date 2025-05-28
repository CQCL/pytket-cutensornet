# Copyright Quantinuum
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

from typing import Any

from pytket._tket.circuit import ClBitVar, ClExpr, ClOp, ClRegVar
from pytket.circuit import (  # type: ignore
    Bit,
    ClExprOp,
    CopyBitsOp,
    Op,
    OpType,
    RangePredicateOp,
    SetBitsOp,
)


def apply_classical_command(
    op: Op, bits: list[Bit], args: list[Any], bits_dict: dict[Bit, bool]
) -> None:
    """Evaluate classical commands and update the `bits_dict` accordingly."""
    if isinstance(op, SetBitsOp):
        for b, v in zip(bits, op.values, strict=False):
            bits_dict[b] = v  # noqa: PERF403

    elif isinstance(op, CopyBitsOp):
        output_bits = bits
        input_bits = args[: len(output_bits)]
        for i, o in zip(input_bits, output_bits, strict=False):
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

    elif isinstance(op, ClExprOp):
        # Convert bit_posn to dictionary of `ClBitVar` index to its value
        bitvar_val = {
            var_id: int(bits_dict[args[bit_pos]])
            for var_id, bit_pos in op.expr.bit_posn.items()
        }
        # Convert reg_posn to dictionary of `ClRegVar` index to its value
        regvar_val = {
            var_id: from_little_endian(
                [bits_dict[args[bit_pos]] for bit_pos in reg_pos_list]
            )
            for var_id, reg_pos_list in op.expr.reg_posn.items()
        }
        # Identify number of bits on each register
        regvar_size = {
            var_id: len(reg_pos_list)
            for var_id, reg_pos_list in op.expr.reg_posn.items()
        }
        # Identify number of bits in output register
        output_size = len(op.expr.output_posn)
        result = evaluate_clexpr(
            op.expr.expr, bitvar_val, regvar_val, regvar_size, output_size
        )

        # The result is an int in little-endian encoding. We update the
        # output register accordingly.
        for bit_pos in op.expr.output_posn:
            bits_dict[args[bit_pos]] = (result % 2) == 1
            result = result >> 1
        # If there has been overflow in the operations, error out.
        # This can be detected if `result != 0`
        if result != 0:
            raise ValueError("Evaluation of the ClExpr resulted in overflow.")

    elif op.type == OpType.Barrier:
        pass

    else:
        raise NotImplementedError(f"Commands of type {op.type} are not supported.")


def evaluate_clexpr(  # noqa: PLR0912, PLR0915
    expr: ClExpr,
    bitvar_val: dict[int, int],
    regvar_val: dict[int, int],
    regvar_size: dict[int, int],
    output_size: int,
) -> int:
    """Recursive evaluation of a ClExpr."""

    # Evaluate arguments to operation
    args_val = []
    for arg in expr.args:
        if isinstance(arg, int):
            value = arg
        elif isinstance(arg, ClBitVar):
            value = bitvar_val[arg.index]
        elif isinstance(arg, ClRegVar):
            value = regvar_val[arg.index]
        elif isinstance(arg, ClExpr):
            value = evaluate_clexpr(
                arg, bitvar_val, regvar_val, regvar_size, output_size
            )
        else:
            raise Exception(f"Unrecognised argument type of ClExpr: {type(arg)}.")

        args_val.append(value)

    # Apply the operation at the root of this ClExpr
    if expr.op in [ClOp.BitAnd, ClOp.RegAnd]:
        result = args_val[0] & args_val[1]
    elif expr.op in [ClOp.BitOr, ClOp.RegOr]:
        result = args_val[0] | args_val[1]
    elif expr.op in [ClOp.BitXor, ClOp.RegXor]:
        result = args_val[0] ^ args_val[1]
    elif expr.op in [ClOp.BitEq, ClOp.RegEq]:
        result = int(args_val[0] == args_val[1])
    elif expr.op in [ClOp.BitNeq, ClOp.RegNeq]:
        result = int(args_val[0] != args_val[1])
    elif expr.op == ClOp.RegGeq:
        result = int(args_val[0] >= args_val[1])
    elif expr.op == ClOp.RegGt:
        result = int(args_val[0] > args_val[1])
    elif expr.op == ClOp.RegLeq:
        result = int(args_val[0] <= args_val[1])
    elif expr.op == ClOp.RegLt:
        result = int(args_val[0] < args_val[1])
    elif expr.op == ClOp.BitNot:
        result = 1 - args_val[0]
    elif expr.op == ClOp.RegNot:  # Bit-wise NOT (flip all bits)
        n_bits = regvar_size[expr.args[0].index]  # type: ignore
        result = (2**n_bits - 1) ^ args_val[0]  # XOR with all 1s bitstring
    elif expr.op in [ClOp.BitZero, ClOp.RegZero]:
        result = 0
    elif expr.op == ClOp.BitOne:
        result = 1
    elif expr.op == ClOp.RegOne:  # All 1s bitstring
        n_bits = output_size
        result = 2**n_bits - 1
    elif expr.op == ClOp.RegAdd:
        result = args_val[0] + args_val[1]
    elif expr.op == ClOp.RegSub:
        if args_val[0] < args_val[1]:
            raise NotImplementedError(
                "Currently not supporting ClOp.RegSub where the outcome is negative."
            )
        result = args_val[0] - args_val[1]
    elif expr.op == ClOp.RegMul:
        result = args_val[0] * args_val[1]
    elif expr.op == ClOp.RegDiv:  # floor(a / b)
        result = args_val[0] // args_val[1]
    elif expr.op == ClOp.RegPow:
        result = int(args_val[0] ** args_val[1])
    elif expr.op == ClOp.RegLsh:
        result = args_val[0] << args_val[1]
    elif expr.op == ClOp.RegRsh:
        result = args_val[0] >> args_val[1]
    # elif expr.op == ClOp.RegNeg:
    #     result = -args_val[0]
    else:
        # TODO: Not supporting RegNeg because I do not know if we have agreed how to
        # specify signed ints.
        raise NotImplementedError(
            f"Evaluation of {expr.op} not supported in ClExpr ",
            "by pytket-cutensornet.",
        )

    return result


def from_little_endian(bitstring: list[bool]) -> int:
    """Obtain the integer from the little-endian encoded bitstring (i.e. bitstring
    [False, True] is interpreted as the integer 2)."""
    # TODO: Assumes unisigned integer. What are the specs for signed integers?
    return sum(1 << i for i, b in enumerate(bitstring) if b)
