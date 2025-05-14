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
"""Module for circuit simulation by state evolution, where the state is
represented by a tensor network with a predefined structure.
Approximate tensor network contraction is supported. Both ``MPS`` and ``TTN``
methods are provided.
For an example of its use, see the ``examples/`` folder at
https://github.com/CQCL/pytket-cutensornet.
"""

from pytket.extensions.cutensornet import CuTensorNetHandle

from .general import Config, LowFidelityException, StructuredState
from .mps import MPS, DirMPS
from .mps_gate import MPSxGate
from .mps_mpo import MPSxMPO
from .simulation import SimulationAlgorithm, prepare_circuit_mps, simulate
from .ttn import TTN, DirTTN
from .ttn_gate import TTNxGate
