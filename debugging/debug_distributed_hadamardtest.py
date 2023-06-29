import pickle
from pytket.circuit import Circuit
from pathlib import Path
from pytket.extensions.cutensornet.backends import CuTensorNetBackend
from pytket.extensions.cutensornet.backends.hadamard_test import general_hadamard_test

root_path = Path() # executes in work directory shifter image

circuits_path = root_path / 'circuits' # executes in work directory shifter image
circuits_path.mkdir(parents=True, exist_ok=True)

n_sites = 4
n_gpus = 1
exp_name = f'hubbard_{n_sites}sites'
with open(circuits_path/ f'{exp_name}.pkl', "rb") as handle:
    obj = pickle.load(handle)

circ = Circuit.from_dict(obj)
a = circ.get_q_register('a')
p = circ.get_q_register('p')
print(circ)

b = CuTensorNetBackend()
c = b.get_compiled_circuit(circ)

postselect = {a[0]: 0}
postselect.update({p_q: 0 for p_q in p})

filename = root_path / f"{exp_name}"

n_slices = n_gpus
expval = general_hadamard_test(c,postselect, n_slices, filename)

print(f'tensor_net_expval: {expval}')