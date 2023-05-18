import inspect

import numpy as np
import numpy.typing as npt

from typing import Any, Callable, Optional, Union, Tuple, Iterable, Type
from typing_extensions import override

from abc import ABC, abstractmethod, abstractproperty

from modcma import Parameters
from modcma.typing_utils import XType, YType


# TODO
class AcquisitionFunctionBase(ABC):
    ModelName = "Base"

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()

    def calculate(self, x: XType, y: YType, var: Optional[YType] = None):
        assert x.shape[0] == y.shape[0]

        if var is not None:
            assert var.shape[0] == y.shape[0]
        else:
            pass
