from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI
import numpy as np
from random import random

import cuquantum as cq
from pytket.circuit import Circuit, fresh_symbol

from pytket.extensions.cuquantum import TensorNetwork

# Parameters
n_qubits = 5
n_circs = 30

root = 0
comm = MPI.COMM_WORLD

rank, n_procs = comm.Get_rank(), comm.Get_size()

circ_list = None

# Generate the list of circuits at root
if rank == root:
    print("\nGenerating list of circuits.")
    time0 = MPI.Wtime()
    circ_list = []

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
    for i in range(n_circs):
        symbol_map = {symbol: random() for symbol in free_symbols}
        my_circ = sym_circ.copy()
        my_circ.symbol_substitution(symbol_map)
        circ_list.append(my_circ)
    time1 = MPI.Wtime()
    print(f"Circuit list generated. Time taken: {time1-time0}\n")

# Broadcast the list of circuits
circ_list = comm.bcast(circ_list, root)

# Calculate all pairs of circuits to be calculated
pairs = [(i,j) for i in range(n_circs) for j in range(n_circs) if i < j]

# Parallelise across all available processes
time0 = MPI.Wtime()

iterations, remainder = len(pairs) // n_procs, len(pairs) % n_procs
for k in range(iterations):
    # Run contraction
    (i, j) = pairs[k*n_procs + rank]
    net0 = TensorNetwork(circ_list[i])
    net1 = TensorNetwork(circ_list[j])
    overlap = cq.contract(*net0.vdot(net1))
    # Report back to user
    print(f"Sample of circuit pair {(i, j)} taken. Overlap: {overlap}")

if rank < remainder:
    # Run contraction
    (i, j) = pairs[iterations*n_procs + rank]
    net0 = TensorNetwork(circ_list[i])
    net1 = TensorNetwork(circ_list[j])
    overlap = cq.contract(*net0.vdot(net1))
    # Report back to user
    print(f"Sample of circuit pair {(i, j)} taken. Overlap: {overlap}")

time1 = MPI.Wtime()

# Report back to user
duration = time1 - time0
print(f"Runtime at {rank} is {duration}")
totaltime = comm.reduce(duration,op = MPI.SUM, root = root)

if rank == root:
    print(f"Total runtime: {totaltime}")
