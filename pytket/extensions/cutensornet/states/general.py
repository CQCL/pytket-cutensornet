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
from __future__ import annotations  # type: ignore
import warnings
from typing import Any, Optional

try:
    import cupy as cp  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cupy", ImportWarning)
try:
    import cuquantum.cutensornet as cutn  # type: ignore
except ImportError:
    warnings.warn("local settings failed to import cutensornet", ImportWarning)


class CuTensorNetHandle:
    """Initialise the cuTensorNet library with automatic workspace memory
    management.

    Note:
        Always use as ``with CuTensorNetHandle() as libhandle:`` so that cuTensorNet
        handles are automatically destroyed at the end of execution.

    Attributes:
        handle (int): The cuTensorNet library handle created by this initialisation.
        device_id (int): The ID of the device (GPU) where cuTensorNet is initialised.
            If not provided, defaults to ``cp.cuda.Device()``.
    """

    def __init__(self, device_id: Optional[int] = None):
        self.handle = cutn.create()
        self._is_destroyed = False

        # Make sure CuPy uses the specified device
        cp.cuda.Device(device_id).use()

        dev = cp.cuda.Device()
        self.device_id = int(dev)

    def __enter__(self) -> CuTensorNetHandle:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        cutn.destroy(self.handle)
        self._is_destroyed = True


class Config:
    """Configuration class for simulation using MPS."""

    def __init__(
        self,
        chi: Optional[int] = None,
        truncation_fidelity: Optional[float] = None,
        k: int = 4,
        optim_delta: float = 1e-5,
        float_precision: Union[np.float32, np.float64] = np.float64,  # type: ignore
        value_of_zero: float = 1e-16,
        loglevel: int = logging.WARNING,
    ):
        """Instantiate a configuration object for MPS simulation.

        Note:
            Providing both a custom ``chi`` and ``truncation_fidelity`` will raise an
            exception. Choose one or the other (or neither, for exact simulation).

        Args:
            chi: The maximum value allowed for the dimension of the virtual
                bonds. Higher implies better approximation but more
                computational resources. If not provided, ``chi`` will be unbounded.
            truncation_fidelity: Every time a two-qubit gate is applied, the virtual
                bond will be truncated to the minimum dimension that satisfies
                ``|<psi|phi>|^2 >= trucantion_fidelity``, where ``|psi>`` and ``|phi>``
                are the states before and after truncation (both normalised).
                If not provided, it will default to its maximum value 1.
            k: If using MPSxMPO, the maximum number of layers the MPO is allowed to
                have before being contracted. Increasing this might increase fidelity,
                but it will also increase resource requirements exponentially.
                Ignored if not using MPSxMPO. Default value is 4.
            optim_delta: If using MPSxMPO, stopping criteria for the optimisation when
                contracting the ``k`` layers of MPO. Stops when the increase of fidelity
                between iterations is smaller than ``optim_delta``.
                Ignored if not using MPSxMPO. Default value is ``1e-5``.
            float_precision: The floating point precision used in tensor calculations;
                choose from ``numpy`` types: ``np.float64`` or ``np.float32``.
                Complex numbers are represented using two of such
                ``float`` numbers. Default is ``np.float64``.
            value_of_zero: Any number below this value will be considered equal to zero.
                Even when no ``chi`` or ``truncation_fidelity`` is provided, singular
                values below this number will be truncated.
                We suggest to use a value slightly below what your chosen
                ``float_precision`` can reasonably achieve. For instance, ``1e-16`` for
                ``np.float64`` precision (default) and ``1e-7`` for ``np.float32``.
            loglevel: Internal logger output level. Use 30 for warnings only, 20 for
                verbose and 10 for debug mode.

        Raises:
            ValueError: If both ``chi`` and ``truncation_fidelity`` are fixed.
            ValueError: If the value of ``chi`` is set below 2.
            ValueError: If the value of ``truncation_fidelity`` is not in [0,1].
        """
        if (
            chi is not None
            and truncation_fidelity is not None
            and truncation_fidelity != 1.0
        ):
            raise ValueError("Cannot fix both chi and truncation_fidelity.")
        if chi is None:
            chi = 2**60  # In practice, this is like having it be unbounded
        if truncation_fidelity is None:
            truncation_fidelity = 1

        if chi < 2:
            raise ValueError("The max virtual bond dim (chi) must be >= 2.")
        if truncation_fidelity < 0 or truncation_fidelity > 1:
            raise ValueError("Provide a value of truncation_fidelity in [0,1].")

        self.chi = chi
        self.truncation_fidelity = truncation_fidelity

        if float_precision is None or float_precision == np.float64:  # Double precision
            self._real_t = np.float64  # type: ignore
            self._complex_t = np.complex128  # type: ignore
            self._atol = 1e-12
        elif float_precision == np.float32:  # Single precision
            self._real_t = np.float32  # type: ignore
            self._complex_t = np.complex64  # type: ignore
            self._atol = 1e-4
        else:
            allowed_precisions = [np.float64, np.float32]
            raise TypeError(
                f"Value of float_precision must be in {allowed_precisions}."
            )
        self.zero = value_of_zero

        if value_of_zero > self._atol / 1000:
            warnings.warn(
                "Your chosen value_of_zero is relatively large. "
                "Faithfulness of final fidelity estimate is not guaranteed.",
                UserWarning,
            )

        self.k = k
        self.optim_delta = 1e-5
        self.loglevel = loglevel

    def copy(self) -> ConfigMPS:
        """Standard copy of the contents."""
        return ConfigMPS(
            chi=self.chi,
            truncation_fidelity=self.truncation_fidelity,
            k=self.k,
            optim_delta=self.optim_delta,
            float_precision=self._real_t,  # type: ignore
        )