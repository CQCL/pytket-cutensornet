import sys
from random import random

from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI
import cuquantum as cq

from pytket.circuit import Circuit, fresh_symbol

from pytket.extensions.cuquantum import TensorNetwork

# Parameters
if len(sys.argv) < 3:
    print(f"You need call this script as {sys.argv[0]} <n_qubits> <n_circs>")
n_qubits = int(sys.argv[1])
n_circs = int(sys.argv[2])

root = 0
comm = MPI.COMM_WORLD

rank, n_procs = comm.Get_rank(), comm.Get_size()
# Assign GPUs uniformly to processes
device_id = rank % getDeviceCount()

time_start = MPI.Wtime()
net_list = None

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
    print("Converting circuits to nets.")
    sys.stdout.flush()
    time0 = MPI.Wtime()

# Convert the circuit to a TensorNetwork for of each of the circuits in this process
this_proc_nets = [TensorNetwork(circ) for circ in this_proc_circs]

if rank == root:
    time1 = MPI.Wtime()
    print(f"Net list generated. Time taken: {time1-time0} seconds.\n")
    print("Broadcasting the net of the circuits.")
    sys.stdout.flush()

# Broadcast the list of nets
time0 = MPI.Wtime()
for proc_i in range(n_procs):
    net_list += comm.bcast(this_proc_nets, proc_i)

time1 = MPI.Wtime()
print(f"Nets broadcasted to {rank} in {time1-time0} seconds")

# Enumerate all pairs of circuits to be calculated
pairs = [(i, j) for i in range(n_circs) for j in range(n_circs) if i < j]

# Find an efficient contraction path to be used by all contractions
time0 = MPI.Wtime()
# Prepare the Network object
net0 = net_list[0]  # Since all circuits have the same structure
net1 = net_list[1]  # we use these two as a template
overlap_network = cq.Network(*net0.vdot(net1), options={"device_id": device_id})
# Compute the path on each process with 8 samples for hyperoptimization
path, info = overlap_network.contract_path(optimize={"samples": 8})
# Select the best path from all ranks.
opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
if rank == root:
    print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")
# Broadcast path from the sender to all other processes
path = comm.bcast(path, sender)
# Report back to user
time1 = MPI.Wtime()
if rank == root:
    print(f"Contraction path found in {time1-time0} seconds.\n")
    sys.stdout.flush()

# Parallelise across all available processes
time0 = MPI.Wtime()

iterations, remainder = len(pairs) // n_procs, len(pairs) % n_procs
progress_bar, progress_checkpoint = 0, iterations // 10
for k in range(iterations):
    # Run contraction
    (i, j) = pairs[k * n_procs + rank]
    net0 = net_list[i]
    net1 = net_list[j]
    overlap = cq.contract(
        *net0.vdot(net1), options={"device_id": device_id}, optimize={"path": path}
    )
    # Report back to user
    # print(f"Sample of circuit pair {(i, j)} taken. Overlap: {overlap}")
    if rank == root and progress_bar * progress_checkpoint < k:
        print(f"{progress_bar*10}%")
        sys.stdout.flush()
        progress_bar += 1

if rank < remainder:
    # Run contraction
    (i, j) = pairs[iterations * n_procs + rank]
    net0 = net_list[i]
    net1 = net_list[j]
    overlap = cq.contract(
        *net0.vdot(net1), options={"device_id": device_id}, optimize={"path": path}
    )
    # Report back to user
    # print(f"Sample of circuit pair {(i, j)} taken. Overlap: {overlap}")

time1 = MPI.Wtime()
time_end = MPI.Wtime()

# Report back to user
duration = time1 - time0
print(f"Runtime at {rank} is {duration}")
totaltime = comm.reduce(duration, op=MPI.SUM, root=root)

if rank == root:
    print(f"\nBroadcasting net.")
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of circuits: {n_circs}")
    print(f"Number of processes used: {n_procs}")
    print(f"Average time per process: {totaltime / n_procs} seconds\n")

full_duration = time_end - time_start
actual_walltime = comm.reduce(full_duration, op=MPI.MAX, root=root)
if rank == root:
    print(f"\n**Full walltime duration** {actual_walltime} seconds\n")
