"""
This example showcases the use of MPI to run an embarrasingly parallel task on multiple
GPUs. The task is to find the inner products of all pairs of ``n_circ`` circuits; each
inner product can be computed separately from the rest, hence the parallelism.
All of the circuits in this example are defined in terms of the same symbolic circuit
``sym_circ`` on ``n_qubits``, with the symbols taking random values for each circuit.

This is one script in a set of three scripts (all named ``mpi_overlap_bcast_*``)
where the difference is which object is broadcasted between the processes. These
are ordered from most efficient to least:
- ``mpi_overlap_bcast_mps.py`` broadcasts ``MPS``.
- ``mpi_overlap_bcast_net.py`` broadcasts ``TensorNetwork``.
- ``mpi_overlap_bcast_circ.py`` broadcasts ``pytket.Circuit``.

In the present script, we proceed as follows:
- Create the same symbolic circuit on every process
- Each process creates a fraction of the ``n_circs`` instances of the symbolic circuit.
    - Then, do *exact* simulation of each of the circuits using an MPS approach.
- Broadcast the resulting ``MPS`` objects to all other processes.
- Distribute calculation of inner products uniformly accross processes. Each process:
    - Obtains the inner product ``<0|C_i^dagger C_j|0>`` using ``vdot`` of the MPS.

The script is able to run on any number of processes; each process must have access to
a GPU of its own.

Notes:
    - We used a very shallow circuit with low entanglement so that contraction time is
      short. Other circuits may be used with varying cost in runtime and memory.
    - Here we are using ``cq.contract`` directly (i.e. cuTensorNet API), but other
      functionalities from our extension (and the backend itself) could be used
      in a similar script.
    - The reason this is the fastest approach is that we want to do as much work as
      possible outside of the loop that computes the inner products: this loop iterates
      ``O(n_circs^2)`` times, but we only need to contract ``O(n_circ)`` MPS objects.
"""

import sys
from random import random

from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

from pytket.circuit import Circuit, fresh_symbol
from pytket.extensions.cutensornet.structured_state import (
    Config,
    CuTensorNetHandle,
    SimulationAlgorithm,
    simulate,
)

# Parameters
if len(sys.argv) < 3:  # noqa: PLR2004
    print(f"You need call this script as {sys.argv[0]} <n_qubits> <n_circs>")
n_qubits = int(sys.argv[1])
n_circs = int(sys.argv[2])

root = 0
comm = MPI.COMM_WORLD

rank, n_procs = comm.Get_rank(), comm.Get_size()
# Assign GPUs uniformly to processes
device_id = rank % getDeviceCount()

time_start = MPI.Wtime()
mps_list = []

if n_circs % n_procs != 0:
    raise RuntimeError(
        "Current version requires that n_circs is a multiple of n_procs."
    )

if rank == root:
    print("\nGenerating the circuits.")
    time0 = MPI.Wtime()

# Generate the list of circuits in parallel
circs_per_proc = n_circs // n_procs
this_proc_circs = []

# Generate the symbolic circuit
sym_circ = Circuit(n_qubits)
even_qs = sym_circ.qubits[0::2]
odd_qs = sym_circ.qubits[1::2]

for q0, q1 in zip(even_qs, odd_qs, strict=False):
    sym_circ.TK2(fresh_symbol(), fresh_symbol(), fresh_symbol(), q0, q1)
for q in sym_circ.qubits:
    sym_circ.H(q)
for q0, q1 in zip(even_qs[1:], odd_qs, strict=False):
    sym_circ.TK2(fresh_symbol(), fresh_symbol(), fresh_symbol(), q0, q1)
free_symbols = sym_circ.free_symbols()

# Create each of the circuits
for _ in range(circs_per_proc):
    symbol_map = {symbol: random() for symbol in free_symbols}
    my_circ = sym_circ.copy()
    my_circ.symbol_substitution(symbol_map)
    this_proc_circs.append(my_circ)

if rank == root:
    time1 = MPI.Wtime()
    print(f"Circuit list generated. Time taken: {time1-time0} seconds.\n")
    print("Contracting the MPS of the circuits.")
    sys.stdout.flush()
    time0 = MPI.Wtime()

# Contract the MPS of each of the circuits in this process
this_proc_mps = []
with CuTensorNetHandle(device_id) as libhandle:  # Different handle for each process
    for circ in this_proc_circs:
        mps = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, Config())
        this_proc_mps.append(mps)

if rank == root:
    time1 = MPI.Wtime()
    print(f"All MPS contracted. Time taken: {time1-time0} seconds.\n")
    print("Broadcasting the MPS of the circuits.")
    sys.stdout.flush()

# Broadcast the list of MPS
time0 = MPI.Wtime()
for proc_i in range(n_procs):
    mps_list += comm.bcast(this_proc_mps, proc_i)
time1 = MPI.Wtime()
print(f"MPS broadcasted to {rank} in {time1-time0} seconds")
time0 = MPI.Wtime()

# Enumerate all pairs of circuits to be calculated
pairs = [(i, j) for i in range(n_circs) for j in range(n_circs) if i < j]

# Parallelise across all available processes
with CuTensorNetHandle(device_id) as libhandle:  # Different handle for each process
    for mps in mps_list:
        mps.update_libhandle(libhandle)  # Update libhandle of this local copy of mps

    iterations, remainder = len(pairs) // n_procs, len(pairs) % n_procs
    progress_bar, progress_checkpoint = 0, iterations // 10
    for k in range(iterations):
        # Run contraction
        (i, j) = pairs[k * n_procs + rank]
        mps0 = mps_list[i]
        mps1 = mps_list[j]
        overlap = mps0.vdot(mps1)
        # Report back to user
        # print(f"Sample of circuit pair {(i, j)} taken. Overlap: {overlap}")
        if rank == root and progress_bar * progress_checkpoint < k:
            print(f"{progress_bar*10}%")
            sys.stdout.flush()
            progress_bar += 1

    if rank < remainder:
        # Run contraction
        (i, j) = pairs[iterations * n_procs + rank]
        mps0 = mps_list[i]
        mps1 = mps_list[j]
        overlap = mps0.vdot(mps1)
        # Report back to user
        # print(f"Sample of circuit pair {(i, j)} taken. Overlap: {overlap}")

# Report back to user
time1 = MPI.Wtime()
time_end = MPI.Wtime()
duration = time1 - time0
print(f"Runtime at {rank} is {duration}")
totaltime = comm.reduce(duration, op=MPI.SUM, root=root)

if rank == root:
    print("\nBroadcasting MPS.")
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of circuits: {n_circs}")
    print(f"Number of processes used: {n_procs}")
    print(f"Average time per process: {totaltime / n_procs} seconds\n")

full_duration = time_end - time_start
actual_walltime = comm.reduce(full_duration, op=MPI.MAX, root=root)
if rank == root:
    print(f"\n**Full walltime duration** {actual_walltime} seconds\n")
