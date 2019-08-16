'''
Configurable parts:
    1. active update: (true, false)
        works on adapt covariance matrix
    2. elitist: (true, false)
        works on selection
    3. mirrored: (true, false)
        used in get_sampler
    4. orthogonal: (true, false)
        used in get_sampler
    5. sequential: (true, false)
        used in eval population
    6. threshold_convergence (opts threshold) (true, false):
        works on mutation
    7. tpa (true, false)
        works in run one generation:
            reduces new population size,
            runs one generation,
            applies tpa update
        works in adapt covariance matrix
        is also of inlfuence in calculate depedencies
    8.  selection (None (best), 'pairwise'):
         defines select function,
         called in run one generation
         In case of pairwise selection, sequential evaluation may only stop after 2mu instead of mu individuals
            seq_cutoff = 2
         is also of inlfuence in calculate depedencies
    9. weights_option (None (else in get_weights), 1/n, '1/2^n' (not used in papers)):
        CMA ES always uses weighted recombination,
        these weights are used (get_weights)
        Also in update_covariance matrix
    10. base-sampler (None,  'quasi-sobol', 'quasi-halton'):
         used in get_sampler, defines base sampler class

    11. local_restart (None, 'IPOP', 'BIPOP'):
        lets not do this for now
'''
from typing import Callable, Optional, List

import numpy as np

import bbobbenchmarks
from Parameters import Parameters
from Population import Population, Individual
from Sampling import GaussianSampling

DISTANCE_TO_TARGET = [pow(10, p) for p in [
    -8.,  # 1
    -8.,  # 2
    .4,  # 3
    .8,  # 4
    -8.,  # 5
    -8.,  # 6
    .0,  # 7
    -8.,  # 8
    -8.,  # 9
    -8.,  # 10
    -8.,  # 11
    -8.,  # 12
    -8.,  # 13
    -8.,  # 14
    .4,  # 15
    -2.,  # 16
    -4.4,  # 17
    -4.0,  # 18
    -.6,  # 19
    .2,  # 20
    -.6,  # 21
    .0,  # 22
    -.8,  # 23
    1.0,  # 24
]]


class ConfigurableCMA:
    '''Add docstring'''

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], int],
        target: float,
        d: int,
        lambda_: Optional[int] = None,
        **kwargs
    ) -> None:
        '''Add docstring'''
        self._fitness_func = fitness_func
        self.parameters = Parameters(
            d=d, lambda_=lambda_, target=target, **kwargs)

        self.parameters.sampler = GaussianSampling(self.parameters.d)

        self.population = Population.new_population(
            self.parameters.mu, self.parameters.d, self.parameters.wcm
        )

    def run(self) -> 'ConfigurableCMA':
        '''Add docstring'''
        while self.step():
            pass
        return self

    def step(self) -> bool:
        self.new_population = self.population.recombine(self.parameters)
        for i, individual in enumerate(self.new_population, 1):
            # TODO:  check performance difference of mutate vectorized
            individual.mutate(self.parameters)
            individual.fitness = self.fitness_func(individual)
            if self.sequential_break_conditions(i, individual):
                break
        self.new_population = self.new_population[:i]
        self.population = self.select(self.new_population)
        self.parameters.adapt(self.new_population)
        return not any(self.break_conditions)

    def sequential_break_conditions(self, i: int, ind: 'Individual'
                                    ) -> bool:
        '''Add docstring'''
        if self.parameters.sequential:
            return (ind.fitness < self.new_population.best_individual.fitness and
                    i > self.parameters.seq_cutoff)
        return False

    def select(self, new_pop: 'Population') -> 'Population':
        '''Add docstring'''
        if self.parameters.elitist:
            new_pop += self.population
        new_pop.sort()
        return new_pop[:self.parameters.mu]

    def fitness_func(self, individual: 'Individual') -> float:
        '''Add docstring'''
        self.parameters.used_budget += 1
        return self._fitness_func(individual.genome.flatten())

    @property
    def break_conditions(self) -> List[bool]:
        '''Add docstring'''
        target = self.parameters.target + self.parameters.rtol
        # For some reason np.close doesn't work here

        target_reached = target >= self.parameters.fce
        # new_population.best_individual.fitness
        return [
            target_reached,
            self.parameters.used_budget >= self.parameters.budget
        ]

    @property
    def fce(self):
        return min(self.population.best_individual.fitness,
                   self.new_population.best_individual.fitness)

    @staticmethod
    def make(ffid: int, *args, **kwargs) -> 'ConfigurableCMA':
        fitness_func, target = bbobbenchmarks.instantiate(
            ffid, iinstance=1)
        rtol = DISTANCE_TO_TARGET[ffid - 1]
        return ConfigurableCMA(fitness_func, target, *args,
                               rtol=rtol,
                               ** kwargs), target
