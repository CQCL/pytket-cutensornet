# Copyright Quantinuum
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
"""Module for simulating circuits with no predetermined tensor network structure."""

from .tensor_network_convert import (
    ExpectationValueTensorNetwork,
    PauliOperatorTensorNetwork,
    TensorNetwork,
    get_circuit_overlap,
    get_operator_expectation_value,
    measure_qubits_state,
    tk_to_tensor_network,
)
from .tensor_network_state import GeneralBraOpKet, GeneralState
from .utils import circuit_statevector_postselect
