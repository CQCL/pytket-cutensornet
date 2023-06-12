import sys
import pickle
from pathlib import Path
from pytket.circuit import Circuit, Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator
from pytket.extensions.cutensornet.backends import CuTensorNetBackend
import numpy

 
n_layers = int(sys.argv[1])

root_path = Path("./benchmarks/hea_100qubits_depth") # executes in work directory shifter image
root_path.mkdir(parents=True, exist_ok=True)
path_circuits = root_path / 'circuits' # executes in work directory shifter image
path_circuits.mkdir(parents=True, exist_ok=True)
path_results = root_path / 'results' # executes in work directory shifter image
path_results.mkdir(parents=True, exist_ok=True)

n_qubits = 100

assert n_layers in numpy.linspace(2, 100, 50, dtype=int)

exp_name = f'hea_ry_{n_qubits}_q_{n_layers}_layers_rand'

path_name = path_circuits / f'{exp_name}.pkl'

with open(path_circuits / f'{exp_name}.pkl','rb')as file:
    circuit = Circuit.from_dict(pickle.load(file))

op = QubitPauliOperator(
    {
        QubitPauliString({Qubit(10): Pauli.X, Qubit(30): Pauli.X}): 1.0,
    }
)

b = CuTensorNetBackend()
c = b.get_compiled_circuit(circuit)
n_max_slices = 100

# exp_name = path_results.name + f'/{exp_name}'
# print('exp_name', exp_name)

expval = b.get_operator_expectation_value_sliced(c, op, n_max_slices,exp_name)

