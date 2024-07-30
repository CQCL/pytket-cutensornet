# Copyright 2019-2024 Quantinuum
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
import logging
from logging import Logger

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
        self._is_destroyed = False

        # Make sure CuPy uses the specified device
        dev = cp.cuda.Device(device_id)
        dev.use()

        self.dev = dev
        self.device_id = dev.id

        self._handle = cutn.create()

    @property
    def handle(self) -> Any:
        if self._is_destroyed:
            raise RuntimeError(
                "The cuTensorNet library handle is out of scope.",
                "See the documentation of CuTensorNetHandle.",
            )
        return self._handle

    def destroy(self) -> None:
        """Destroys the memory handle, releasing memory.

        Only call this method if you are initialising a ``CuTensorNetHandle`` outside
        a ``with CuTensorNetHandle() as libhandle`` statement.
        """
        cutn.destroy(self._handle)
        self._is_destroyed = True

    def __enter__(self) -> CuTensorNetHandle:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.destroy()

    def print_device_properties(self, logger: Logger) -> None:
        """Prints local GPU properties."""
        device_props = cp.cuda.runtime.getDeviceProperties(self.dev.id)
        logger.info("===== device info ======")
        logger.info("GPU-name: " + device_props["name"].decode())
        logger.info("GPU-clock: " + str(device_props["clockRate"]))
        logger.info("GPU-memoryClock: " + str(device_props["memoryClockRate"]))
        logger.info("GPU-nSM: " + str(device_props["multiProcessorCount"]))
        logger.info("GPU-major: " + str(device_props["major"]))
        logger.info("GPU-minor: " + str(device_props["minor"]))
        logger.info("========================")


def set_logger(
    logger_name: str,
    level: int = logging.WARNING,
    fmt: str = "[%(asctime)s.%(msecs)03d] %(name)s (%(levelname)s) - %(message)s",
) -> Logger:
    """Initialises and configures a logger object.

    Args:
        logger_name: Name for the logger object.
        level: Logger output level.
        fmt: Logger output format.

    Returns:
        New configured logger object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
