import abc
from typing import Callable, Optional, List
import numpy as np


class Optimizer(abc.ABC):
    def run(self):
        '''Add docstring'''
        while self.step():
            pass
        return self

    @property
    def break_conditions(self) -> List[bool]:
        '''Add docstring'''
        return [
            self.target >= self.fopt,
            self.used_budget >= self.budget
        ]

    def fitness_func(self, x: np.ndarray) -> float:
        '''Add docstring'''
        self.used_budget += 1
        return self._fitness_func(x.flatten())
