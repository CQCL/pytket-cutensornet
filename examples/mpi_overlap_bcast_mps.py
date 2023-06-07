import sys
from random import random

from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

from pytket.circuit import Circuit, fresh_symbol

from pytket.extensions.cuquantum.mps import simulate

# Parameters
if len(sys.argv) < 3:
    print(f"You need call this script as {sys.argv[0]} <n_qubits> <n_circs>")
n_qubits = int(sys.argv[1])
n_circs = int(sys.argv[2])
# Set chi for exact contraction
chi = 2 ** (n_qubits // 2)

root = 0
comm = MPI.COMM_WORLD

rank, n_procs = comm.Get_rank(), comm.Get_size()
# Assign GPUs uniformly to processes
device_id = rank % getDeviceCount()

mps_list = []

if n_circs % n_procs != 0:
    raise RuntimeError(
        "Current version requires that n_circss is a multiple of n_procs."
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

for q0, q1 in zip(even_qs, odd_qs):
    sym_circ.TK2(fresh_symbol(), fresh_symbol(), fresh_symbol(), q0, q1)
for q in sym_circ.qubits:
    sym_circ.H(q)
for q0, q1 in zip(even_qs[1:], odd_qs):
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
for circ in this_proc_circs:
    this_proc_mps.append(simulate(circ, "MPSxGate", chi, device_id=device_id))

if rank == root:
    time1 = MPI.Wtime()
    print(f"All MPS contracted. Time taken: {time1-time0} seconds.\n")
    print("Broadcasting the MPS of the circuits.")
    sys.stdout.flush()

# Broadcast the list of MPS
time0 = MPI.Wtime()
for proc_i in range(n_procs):
    mps_list += comm.bcast(this_proc_mps, proc_i)
# Change device ID
# TODO: I don't think this is moving mem between GPUs on same node. Look into NCCL.
for mps in mps_list:
    mps._device_id = device_id
    mps.init_cutensornet()

time1 = MPI.Wtime()
print(f"MPS broadcasted to {rank} in {time1-time0} seconds")
time0 = MPI.Wtime()

# Enumerate all pairs of circuits to be calculated
pairs = [(i, j) for i in range(n_circs) for j in range(n_circs) if i < j]

# Parallelise across all available processes
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
duration = time1 - time0
print(f"Runtime at {rank} is {duration}")
totaltime = comm.reduce(duration, op=MPI.SUM, root=root)

if rank == root:
    print(f"\nBroadcasting MPS.")
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of circuits: {n_circs}")
    print(f"Number of processes used: {n_procs}")
    print(f"Average time per process: {totaltime / n_procs} seconds\n")
