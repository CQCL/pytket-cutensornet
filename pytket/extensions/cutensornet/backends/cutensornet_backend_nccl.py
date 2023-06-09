# Copyright 2019-2023 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Methods to allow tket circuits to be run on the cuTensorNet simulator."""

import warnings

try:
    import cuquantum as cq  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)
from logging import warning
from typing import List, Union, Optional, Sequence
from uuid import uuid4
import numpy as np
from sympy import Expr  # type: ignore
from pytket.circuit import Circuit, OpType, Qubit  # type: ignore
from pytket.backends import ResultHandle, CircuitStatus, StatusEnum, CircuitNotRunError
from pytket.backends.backend import KwargTypes, Backend, BackendResult
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.cutensornet.tensor_network_convert import (
    TensorNetwork,
    ExpectationValueTensorNetwork,
    tk_to_tensor_network,
    measure_qubits_state,
)
from pytket.predicates import Predicate, GateSetPredicate, NoClassicalBitsPredicate  # type: ignore
from pytket.passes import (  # type: ignore
    BasePass,
    SequencePass,
    DecomposeBoxes,
    SynthesiseTket,
    FullPeepholeOptimise,
    RebaseCustom,
    SquashCustom,
)
from pytket.utils.operators import QubitPauliOperator

import cupy as cp
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI
from cupy.cypyx.distributed import NCCLBackend, init_process_group


# TODO: this is temporary - probably don't need it eventually?
def _sq(a: Expr, b: Expr, c: Expr) -> Circuit:
    circ = Circuit(1)
    if c != 0:
        circ.Rz(c, 0)
    if b != 0:
        circ.Rx(b, 0)
    if a != 0:
        circ.Rz(a, 0)
    return circ


class CuTensorNetBackend(Backend):
    """A pytket Backend wrapping around the cuTensorNet simulator."""

    _supports_state = True
    _supports_expectation = True
    _persistent_handles = False

    # TODO: add self._backend_info?
    def __init__(self) -> None:
        """Constructs a new cuTensorNet backend object."""
        super().__init__()

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    # TODO: return some info? Should it return self._backend_info instantiated on
    #  construction?
    @property
    def backend_info(self) -> Optional[BackendInfo]:
        """Returns information on the backend."""
        return None

    # TODO: Surely we can allow for more gate sets - needs thorough testing though.
    @property
    def required_predicates(self) -> List[Predicate]:
        """Returns the minimum set of predicates that a circuit must satisfy.

        Predicates need to be satisfied before the circuit can be successfully run on
        this backend.

        Returns:
            List of required predicates.
        """
        preds = [
            NoClassicalBitsPredicate(),
            GateSetPredicate(
                {
                    OpType.Rx,
                    OpType.Ry,
                    OpType.Rz,
                    OpType.ZZMax,
                    OpType.SWAP,
                }
            ),
        ]
        return preds

    # TODO: also probably needs improvement.
    def rebase_pass(self) -> BasePass:
        """Defines rebasing method.

        Returns:
            Custom rebase pass object.
        """
        cx_circ = Circuit(2)
        cx_circ.Sdg(0)
        cx_circ.V(1)
        cx_circ.Sdg(1)
        cx_circ.Vdg(1)
        cx_circ.add_gate(OpType.ZZMax, [0, 1])
        cx_circ.Vdg(1)
        cx_circ.Sdg(1)
        cx_circ.add_phase(0.5)
        return RebaseCustom(
            {OpType.Rx, OpType.Ry, OpType.Rz, OpType.ZZMax}, cx_circ, _sq
        )

    # TODO: same as above?
    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        """Returns a default compilation pass.

        A suggested compilation pass that will guarantee the resulting circuit
        will be suitable to run on this backend with as few preconditions as
        possible.

        Args:
            optimisation_level: The level of optimisation to perform during
                compilation. Level 0 just solves the device constraints without
                optimising. Level 1 additionally performs some light optimisations.
                Level 2 adds more intensive optimisations that can increase compilation
                time for large circuits. Defaults to 1.
        Returns:
            Compilation pass guaranteeing required predicates.
        """
        assert optimisation_level in range(3)
        squash = SquashCustom({OpType.Rz, OpType.Rx, OpType.Ry}, _sq)
        seq = [DecomposeBoxes()]  # Decompose boxes into basic gates
        if optimisation_level == 1:
            seq.append(SynthesiseTket())  # Optional fast optimisation
        elif optimisation_level == 2:
            seq.append(FullPeepholeOptimise())  # Optional heavy optimisation
        seq.append(self.rebase_pass())  # Map to target gate set
        if optimisation_level != 0:
            seq.append(
                squash
            )  # Optionally simplify 1qb gate chains within this gate set
        return SequencePass(seq)

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        """Returns circuit status object.

        Returns:
            CircuitStatus object.

        Raises:
            CircuitNotRunError: if there is no handle object in cache.
        """
        if handle in self._cache:
            return CircuitStatus(StatusEnum.COMPLETED)
        raise CircuitNotRunError(handle)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Optional[Union[int, Sequence[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """Submits circuits to the backend for running.

        The results will be stored in the backend's result cache to be retrieved by the
        corresponding get_<data> method.

        Args:
            circuits: List of circuits to be submitted.
            n_shots: Number of shots in case of shot-based calculation.
            valid_check: Whether to check for circuit correctness.

        Returns:
            Results handle objects.

        Raises:
            TypeError: If global phase is dependent on a symbolic parameter.
        """
        circuit_list = list(circuits)
        if valid_check:
            self._check_all_circuits(circuit_list)
        handle_list = []
        for circuit in circuit_list:
            state_tnet = tk_to_tensor_network(circuit)
            state = cq.contract(*state_tnet).flatten()
            try:  # This constraint (from pytket-Qulacs) seems reasonable?
                phase = float(circuit.phase)
                coeff = np.exp(phase * np.pi * 1j)
                state *= coeff  # type: ignore
            except TypeError:
                warning(
                    "Global phase is dependent on a symbolic parameter, so cannot "
                    "adjust for phase"
                )
            # Qubits order:
            # implicit_perm = circuit.implicit_qubit_permutation()
            # res_qubits = [
            #     implicit_perm[qb] for qb in sorted(circuit.qubits, reverse=False)
            # ]  # reverse was set to True in the pytket-example but this fails tests.
            res_qubits = [qb for qb in sorted(circuit.qubits)]
            # The below line is as per pytket-Qulacs, but this alone fails the implicit
            # permutation test result.
            # res_qubits = sorted(circuit.qubits, reverse=False)
            handle = ResultHandle(str(uuid4()))
            self._cache[handle] = {
                "result": BackendResult(q_bits=res_qubits, state=state)
            }
            handle_list.append(handle)
        return handle_list

    # TODO: this should be optionally parallelised with MPI
    #  (both wrt Pauli strings and contraction itself).
    def get_operator_expectation_value_nccl(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        valid_check: bool = True,
    ) -> float:
        """Calculates expectation value of an operator using cuTensorNet contraction.

        Args:
            state_circuit: Circuit representing state.
            operator: Operator which expectation value is to be calculated.
            valid_check: Whether to perform circuit validity check.

        Returns:
            Real part of the expectation value.
        """
        if valid_check:
            self._check_all_circuits([state_circuit])

        state_circuit.replace_implicit_wire_swaps()

        expectation = 0
        for qos, coeff in operator._dict.items():
            ket_network = TensorNetwork(state_circuit)
            bra_network = ket_network.dagger()
            expectation_value_network = ExpectationValueTensorNetwork(
                bra_network, qos, ket_network
            )
            if isinstance(coeff, Expr):
                numeric_coeff = complex(coeff.evalf())  # type: ignore
            else:
                numeric_coeff = complex(coeff)



            expectation_term = numeric_coeff * cq.contract(
                *expectation_value_network.cuquantum_interleaved
            )
            expectation += expectation_term
        return expectation.real

    def get_circuit_overlap(
        self,
        circuit_ket: Circuit,
        circuit_bra: Optional[Circuit] = None,
        valid_check: bool = True,
    ) -> float:
        """Calculates an overlap of two states represented by two circuits.

        Args:
            circuit_bra: Circuit representing the bra state.
            circuit_ket: Circuit representing the ket state.
            valid_check: Whether to perform circuit validity check.

        Returns:
            Overlap value.
        """
        if circuit_bra is None:
            circuit_bra = circuit_ket
        if valid_check:
            self._check_all_circuits([circuit_bra, circuit_ket])

        ket_net = TensorNetwork(circuit_ket)
        overlap_net_interleaved = ket_net.vdot(TensorNetwork(circuit_bra))
        overlap: float = cq.contract(*overlap_net_interleaved)
        return overlap

    def get_operator_expectation_value_postselect(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        post_selection: dict[Qubit, int],
        valid_check: bool = True,
    ) -> float:
        """Calculates expectation value of an operator using
        cuTensorNet contraction where the is a post selection on an ancilla register.

        Args:
            state_circuit: Circuit representing state.
            operator: Operator which expectation value is to be calculated.
            valid_check: Whether to perform circuit validity check.
            post_selection: Dictionary of qubits to post select where the key is
                qubit and the value is bit outcome.

        Returns:
            Expectation value.
        """
        if valid_check:
            self._check_all_circuits([state_circuit])

        post_select_qubits = list(post_selection.keys())
        if set(post_select_qubits).issubset(operator.all_qubits):
            raise ValueError(
                "Post selection qubit must not be a not be a subset of operator qubits"
            )

        ket_network = TensorNetwork(state_circuit)
        bra_network = ket_network.dagger()
        ket_network = measure_qubits_state(ket_network, post_selection)
        bra_network = measure_qubits_state(
            bra_network, post_selection
        )  # This needed because dagger does not work with post selection

        expectation = 0

        for qos, coeff in operator._dict.items():
            expectation_value_network = ExpectationValueTensorNetwork(
                bra_network, qos, ket_network
            )
            if isinstance(coeff, Expr):
                numeric_coeff = complex(coeff.evalf())  # type: ignore
            else:
                numeric_coeff = complex(coeff)
            expectation_term = numeric_coeff * cq.contract(
                *expectation_value_network.cuquantum_interleaved
            )
            expectation += expectation_term
        return expectation.real
    

def slice_contract_ncclcommunicator(tensor_network: TensorNetwork, n_slices: int, exp_name:str):

    root = 0
    comm_mpi = MPI.COMM_WORLD
    rank, size = comm_mpi.Get_rank(), comm_mpi.Get_size()

    nccl_comm = init_process_group(rank,size)

    time0 = MPI.Wtime()
    path, info = network.contract_path(optimize={'samples': 8, 'slicing': {'min_slices': max(16, size)}})

    # Select the best path from all ranks. Note that we still use the MPI communicator here for simplicity.
    opt_cost, sender = nccl_comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
    time1 = MPI.Wtime()
    if rank == root:
        print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

    # Broadcast info from the sender to all other ranks.
    info = nccl_comm.broadcast(info, sender)

    # Set path and slices.
    path, info = network.contract_path(optimize={'path': info.path, 'slicing': info.slices})
    
    # Calculate this process's share of the slices.
    num_slices = info.num_slices
    chunk, extra = num_slices // size, num_slices % size
    slice_begin = rank * chunk + min(rank, extra)
    slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
    slices = range(slice_begin, slice_end)

    time2 = MPI.Wtime()

    print(f"Process {rank} is processing slice range: {slices}.")

    # Create dataframe for info with nameds column strings
    df = pd.DataFrame(info)
  
  
    #Where does auutotune come in?

    # Contract the group of slices the process is responsible for.
    result = network.contract(slices=slices)

    result = nccl_comm.reduce(sendobj=result, op=MPI.SUM)

    time3 = MPI.Wtime()

    info.total_time = time3 - time0
    info.slicing_time = time1 - time0
    info.repotiming_slice_path_time = time2 - time1
    info.contract_slices_time = time3 - time2

    df = df.append(info, ignore_index=True)
    df.to_csv(f'{exp_name}.csv')


    return result



# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating slice-based parallel tensor network contraction with cuQuantum using NCCL and MPI. Here
we create the input tensors directly on the GPU using CuPy since NCCL only supports GPU buffers.

The low-level Python wrapper for NCCL is provided by CuPy. MPI (through mpi4py) is only needed to bootstrap
the multiple processes, set up the NCCL communicator, and to communicate data on the CPU. NCCL can be used
without MPI for a "single process multiple GPU" model.

For users who do not have NCCL installed already, CuPy provides detailed instructions on how to install
it for both pip and conda users when "import cupy.cuda.nccl" fails.

We recommend that those using CuPy v10+ use CuPy's high-level "cupyx.distributed" module to avoid having to
manipulate GPU pointers in Python.

Note that with recent NCCL, GPUs cannot be oversubscribed (not more than one process per GPU). Users will
see an NCCL error if the number of processes on a node exceeds the number of GPUs on that node.

$ mpiexec -n 4 python example4_mpi_nccl.py
"""

import cupy as cp
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

from cuquantum import Network

# Set up the MPI environment.
root = 0
comm_mpi = MPI.COMM_WORLD
rank, size = comm_mpi.Get_rank(), comm_mpi.Get_size()

# Assign the device for each process.
device_id = rank % getDeviceCount()

# Define the tensor network topology.
expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

# Note that all NCCL operations must be performed in the correct device context.
cp.cuda.Device(device_id).use()

# Set up the NCCL communicator.
nccl_id = nccl.get_unique_id() if rank == root else None
nccl_id = comm_mpi.bcast(nccl_id, root)
comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

# Set the operand data on root.
if rank == root:
    operands = [cp.random.rand(*shape) for shape in shapes]
else:
    operands = [cp.empty(shape) for shape in shapes]

# Broadcast the operand data. We pass in the CuPy ndarray data pointers to the NCCL APIs.
stream_ptr = cp.cuda.get_current_stream().ptr
for operand in operands:
    comm_nccl.broadcast(operand.data.ptr, operand.data.ptr, operand.size, nccl.NCCL_FLOAT64, root, stream_ptr)

# Create network object.
network = Network(expr, *operands)

# Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
path, info = network.contract_path(optimize={'samples': 8, 'slicing': {'min_slices': max(16, size)}})

# Select the best path from all ranks. Note that we still use the MPI communicator here for simplicity.
opt_cost, sender = comm_mpi.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
if rank == root:
    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

# Broadcast info from the sender to all other ranks.
info = comm_mpi.bcast(info, sender)

# Set path and slices.
path, info = network.contract_path(optimize={'path': info.path, 'slicing': info.slices})

# Calculate this process's share of the slices.
num_slices = info.num_slices
chunk, extra = num_slices // size, num_slices % size
slice_begin = rank * chunk + min(rank, extra)
slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
slices = range(slice_begin, slice_end)

print(f"Process {rank} is processing slice range: {slices}.")

# Contract the group of slices the process is responsible for.
result = network.contract(slices=slices)

# Sum the partial contribution from each process on root.
stream_ptr = cp.cuda.get_current_stream().ptr
comm_nccl.reduce(result.data.ptr, result.data.ptr, result.size, nccl.NCCL_FLOAT64, nccl.NCCL_SUM, root, stream_ptr)

# Check correctness.
if rank == root:
    result_cp = cp.einsum(expr, *operands, optimize=True)
    print("Does the cuQuantum parallel contraction result match the cupy.einsum result?", cp.allclose(result, result_cp))











from tensor_network import TensorNetwork
import pickle

from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI
import numpy as np

import cuquantum as cq

def load_pickle(path):
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    return obj

root = 0
comm = MPI.COMM_WORLD

rank, size = comm.Get_rank(), comm.Get_size()

time0 = MPI.Wtime()

# Read in a pytket circuit (same on each process)
circuit = load_pickle("./18_in_36_k_1_ansatz_circuit.pickle")

# Set the operand data (same on all processes).
ket = TensorNetwork(circuit)
ovl = ket.vdot(ket)
if rank == root: print(f"Contracting {int(len(ovl)/2)} tensors.")

# Assign the device for each process.
device_id = rank % getDeviceCount()

# Create network object.
network = cq.Network(*ovl, options={'device_id' : device_id})

# Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
path, info = network.contract_path(optimize={'samples': 4, 'slicing': {'min_slices': max(16, size)}})

# Select the best path from all ranks.
opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
if rank == root: print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

# Broadcast info from the sender to all other ranks.
info = comm.bcast(info, sender)

# Set path and slices.
path, info = network.contract_path(optimize={'path': info.path, 'slicing': info.slices})

time1 = MPI.Wtime()
duration = time1 - time0
print(f"Optimising contraction path at {rank} took {duration} sec.")

# Calculate this process's share of the slices.
num_slices = info.num_slices
chunk, extra = num_slices // size, num_slices % size
slice_begin = rank * chunk + min(rank, extra)
slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
slices = range(slice_begin, slice_end)

print(f"Process {rank} is processing slice range: {slices}.")

# Contract the group of slices the process is responsible for.
result = network.contract(slices=slices)

# Sum the partial contribution from each process on root.
result = comm.reduce(sendobj=result, op=MPI.SUM, root=root)
if rank == root: print(f"Result: {result}")

time2 = MPI.Wtime()
duration = time2 - time1
print(f"Contraction at {rank} took {duration} sec.")