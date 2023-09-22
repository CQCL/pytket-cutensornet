# Copyright 2019 Cambridge Quantum Computing
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
"""Module for circuit simulation by state evolution, with states represented as
Matrix Product States (MPS). Approximate tensor network contraction is supported.
For an example of its use, see ``examples/mps_tutorial.ipynb`` in
https://github.com/CQCL/pytket-cutensornet.
"""

from .mps import (
    CuTensorNetHandle,
    DirectionMPS,
    Handle,
    Tensor,
    MPS,
)

from .mps_gate import (
    MPSxGate,
)

from .mps_mpo import (
    MPSxMPO,
)

from .simulation import ContractionAlg, simulate, prepare_circuit
