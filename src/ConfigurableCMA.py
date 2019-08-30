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
from datetime import datetime

import numpy as np

from Utils import bbobfunction
from Parameters import Parameters
from Population import Population, Individual
from Sampling import GaussianSampling
from Optimizer import Optimizer


class ConfigurableCMA(Optimizer):
    '''Add docstring'''

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], int],
        absolute_target: float,
        d: int,
        lambda_: Optional[int] = None,
        old_order: Optional[bool] = False,
        **kwargs
    ) -> None:
        '''Add docstring'''
        self._fitness_func = fitness_func
        self.parameters = Parameters(
            d=d, lambda_=lambda_, absolute_target=absolute_target, **kwargs)

        self.parameters.sampler = GaussianSampling(self.parameters.d)

        self.population = Population.new_population(
            self.parameters.mu, self.parameters.d, self.parameters.wcm
        )
        if old_order:
            self.new_population = self.population.recombine(self.parameters)
            self.step = self.step_old

    def step_old(self) -> bool:
        for i, individual in enumerate(self.new_population, 1):
            individual.mutate(self.parameters)
            individual.fitness = self.fitness_func(individual.genome)
            if self.sequential_break_conditions(i, individual):
                break
        self.parameters.record_statistics(self.new_population)
        self.new_population = self.new_population[:i]
        self.population = self.select(self.new_population)
        self.new_population = self.population.recombine(self.parameters)
        self.parameters.adapt()
        return not any(self.break_conditions)

    def step(self) -> bool:
        self.new_population = self.population.recombine(self.parameters)
        for i, individual in enumerate(self.new_population, 1):
            # TODO:  check performance difference of mutate vectorized
            individual.mutate(self.parameters)
            individual.fitness = self.fitness_func(individual)
            if self.sequential_break_conditions(i, individual):
                break
        self.parameters.record_statistics(self.new_population)
        self.new_population = self.new_population[:i]
        self.population = self.select(self.new_population)
        self.parameters.adapt()
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
        # Weird stucture to put this here
        self.parameters.offset = new_pop.mutation_vectors
        return new_pop[:self.parameters.mu]

    @property
    def used_budget(self):
        return self.parameters.used_budget

    @used_budget.setter
    def used_budget(self, value):
        self.parameters.used_budget = value

    @property
    def fopt(self):
        return self.parameters.fopt

    @property
    def budget(self):
        return self.parameters.budget

    @property
    def target(self):
        return self.parameters.target


if __name__ == "__main__":
    np.random.seed(12)
    from Utils import evaluate
    for i in range(1, 25):
        evals, fopts = evaluate(i, 5, ConfigurableCMA)
