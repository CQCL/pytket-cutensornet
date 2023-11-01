# Copyright 2019-2023 Quantinuum
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
"""Module for circuit simulation by state evolution.
Approximate tensor network contraction is supported. Both ``MPS`` and ``TTN``
methods are provided.
For an example of its use, see ``examples/mps_tutorial.ipynb`` in
https://github.com/CQCL/pytket-cutensornet.
"""

from .general import CuTensorNetHandle, Config, TNState
from .simulation import ContractionAlg, simulate, prepare_circuit_mps

from .mps import DirMPS, MPS
from .mps_gate import MPSxGate
from .mps_mpo import MPSxMPO

from .ttn import TTN, DirTTN
from .ttn_gate import TTNxGate
