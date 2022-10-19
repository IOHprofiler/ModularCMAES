from .parameters import Parameters
from .population import Population
import numpy as np
import numpy.typing as npt

from typing import Any, Union, Tuple

from abc import abstractmethod, ABCMeta

#####################
# Population Storage Management


class PopHistory:
    '''Saves Entire Population History

        += Population to add another
    '''

    def __init__(self, x, y, f, f_true, s, generation=None):
        self.population: Population = Population(
                x=x, y=y, f=f, f_true=f_true, s=s
            )
        self.generation: npt.NDArray[np.int32] = \
            np.empty(shape=(1, 0), dtype=np.int32) if generation is None else \
            generation

    @staticmethod
    def empty(d: int) -> "PopHistory":
        return PopHistory(
                x=np.empty(shape=(d, 0)),
                y=np.empty(shape=(d, 0)),
                f=np.empty(shape=(1, 0)),
                f_true=np.empty(shape=(1, 0)),
                s=np.empty(shape=(1, 0))
            )

    def __iadd__(self, parameters: Parameters, population: Population) -> None:
        self.population += population
        # extra info
        self.generation = np.append(
            self.generation,
            parameters.t + np.zeros_like(population.f, dtype=np.int32)
        )

    def __getitem__(self, key: npt.NDArray[np.bool_]) -> "PopHistory":
        return PopHistory(
            self.population.x[:, key],
            self.population.y[:, key],
            self.population.f[key],
            self.population.f_true[key],
            self.population.s[key],
            self.generation[key]
        )


#####################
# FILTER CLASSES for surrogate models

class FilterRealEvaluation:
    def __call__(self, pop: PopHistory) -> PopHistory:
        mask = pop.population.f_true
        return pop[mask]


class FilterDistance(metaclass=ABCMeta):
    def __init__(self, parameters: Parameters, distance: float):
        self.max_distance = distance
        self.parameters = parameters

    @abstractmethod
    def _compute_distance(self, pop: PopHistory) -> npt.NDArray[np.float32]:
        pass

    def _get_mask(self, pop: PopHistory) -> npt.NDArray[np.bool_]:
        distance = self._compute_distance(pop)
        return distance <= self.max_distance

    def __call__(self, pop: PopHistory) -> PopHistory:
        mask = self._get_mask(pop)
        return pop[mask]


class FilterDistanceMahalanobis(FilterDistance):
    def __init__(self, parameters: Parameters, distance: float):
        super().__init__(parameters, distance)
        B = self.parameters.B
        sigma = self.parameters.sigma
        D = self.parameters.D

        self.transformation = np.linalg.inv(B) @ np.diag((1./sigma)*(1./D))

    def _compute_distance(self, pop: PopHistory) -> npt.NDArray[np.float32]:
        center_x = pop.population.x - self.parameters.m
        return np.sqrt(self.transformation @ center_x)


FILTER_TYPE = Union[FilterRealEvaluation, FilterDistanceMahalanobis]

####################
# Surrogate Model


class SurrogateModel:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def __call__(self, x) -> np.ndarray:
        return np.repeat(np.nan, [x.shape[0], 1])
