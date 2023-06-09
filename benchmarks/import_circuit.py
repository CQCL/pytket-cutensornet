import pickle
from pathlib import Path
from pytket.circuit import Circuit

path = Path("./hea_50qubits_depth")
path.mkdir(parents=True, exist_ok=True)

n_qubits = 50
n_layers = 6

file = open(path / f'hea_ry_{n_qubits}_q_{n_layers}_layers_rand.pkl','rb')

with open(path / f'hea_ry_{n_qubits}_q_{n_layers}_layers_rand.pkl','rb')as file:
    circuit = Circuit.from_dict(pickle.load(file))