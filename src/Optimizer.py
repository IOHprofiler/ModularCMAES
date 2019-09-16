import abc
from typing import List
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
            self.parameters.target >= self.parameters.fopt,
            self.parameters.used_budget >= self.parameters.budget
        ]

    def fitness_func(self, x: np.ndarray) -> float:
        '''Add docstring'''
        self.parameters.used_budget += 1
        return self._fitness_func(x.flatten())

    @abc.abstractmethod
    def step(self):
        raise NotImplemented()
