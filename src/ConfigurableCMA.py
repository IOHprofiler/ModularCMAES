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
from __future__ import annotations
import itertools
from time import time
from argparse import ArgumentParser
from typing import Callable, Optional, List

import numpy as np
import bbobbenchmarks

from Parameters import Parameters
from Population import Population, Individual
from Sampling import GaussianSampling


class ConfigurableCMA:
    '''Add docstring'''

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], int],
        target: float,
        d: int,
        lambda_: Optional[int] = None
    ) -> None:
        '''Add docstring'''
        self._fitness_func = fitness_func
        self.parameters = Parameters(
            d=d, lambda_=lambda_, target=target)

        self.parameters.sampler = GaussianSampling(self.parameters.d)

        self.population = Population.new_population(
            self.parameters.mu, self.parameters.d, self.parameters.wcm
        )

    def run(self) -> ConfigurableCMA:
        '''Add docstring'''
        for generation in range(1, self.parameters.max_generations):
            self.new_population = self.population.recombine(self.parameters)
            for i, individual in enumerate(self.new_population, 1):
                individual.mutate(self.parameters)
                individual.fitness = self.fitness_func(individual)
                if self.sequential_break_conditions(i, individual):
                    break
            if any(self.normal_break_conditions):
                break
            self.new_population = self.new_population[:i]
            self.population = self.select(self.new_population)
            self.parameters.adapt(self.new_population)
        return self

    def sequential_break_conditions(self, i: int, ind: Individual
                                    ) -> bool:
        '''Add docstring'''
        if self.parameters.sequential:
            return (ind.fitness < self.pop.best_individual.fitness and
                    i > self.parameters.seq_cutoff)
        return False

    def select(self, new_pop: Population) -> Population:
        '''Add docstring'''
        if self.parameters.elitist:
            new_pop += self.population
        new_pop.sort()
        return new_pop[:self.parameters.mu]

    def fitness_func(self, individual: Individual) -> float:
        '''Add docstring'''
        self.parameters.used_budget += 1
        return self._fitness_func(individual.genome.flatten())

    @property
    def normal_break_conditions(self) -> List[bool, bool]:
        '''Add docstring'''
        target_reached = np.isclose(
            self.parameters.target,
            self.new_population.best_individual.fitness,
            rtol=self.parameters.rtol)
        return [
            target_reached,
            self.parameters.used_budget >= self.parameters.budget
        ]

    @staticmethod
    def make(ffid: int, *args, **kwargs) -> ConfigurableCMA:
        fitness_func, target = bbobbenchmarks.instantiate(
            ffid, iinstance=1)
        return ConfigurableCMA(fitness_func, target, *args, **kwargs)


def evaluate(functionid, dim, iterations):
    start = time()
    ets, fce = [], []
    for i in range(iterations):
        cma = ConfigurableCMA.make(functionid, d=dim).run()
        ets.append(cma.parameters.used_budget)
        fce.append(cma.population.best_individual.fitness)

    ets = np.array(ets)

    print("FCE:\t", np.round(np.mean(fce), 4), "\t", np.round(np.std(fce), 4))
    print(
        "ERT:\t", np.round(
            ets.sum() / (ets != cma.parameters.budget).sum(), 4),
        "\t", np.round(np.std(ets), 4)
    )
    # breakpoint()
    print("Time:\t", time() - start)


def benchmark(functionid, dim, iterations):
    import cma
    f, t = bbobbenchmarks.instantiate(
        functionid, iinstance=1)
    params = Parameters(
        d=dim, mu=3, lambda_=None, target=t)

    start = time()
    ets, fce = [], []
    for i in range(iterations):
        es = cma.CMAEvolutionStrategy(
            params.wcm.flatten().tolist(), params.sigma,
            {
                'verbose': -8,
                'CMA_active': False,
                "CMA_mirrormethod": 0,
                'CMA_mu': 3,

            }
        )
        while not es.stop():
            es.tell(*es.ask_and_eval(f))
            if np.isclose(es.best.f, t):
                ets.append(es.countevals)
                fce.append(es.best.f)
                break

    ets = np.array(ets)
    print("FCE:\t", np.round(np.mean(fce), 4), "\t", np.round(np.std(fce), 4))
    print(
        "ERT:\t", np.round(
            ets.sum() / (ets != params.budget).sum(), 4),
        "\t", np.round(np.std(ets), 4)
    )
    print("Time:\t", time() - start)


def main():
    parser = ArgumentParser(
        description='Run single function MAB exp iid 1')
    parser.add_argument(
        '-f', "--functionid", type=int,
        help="bbob function id", required=False, default=5
    )
    parser.add_argument(
        '-d', "--dim", type=int,
        help="dimension", required=False, default=5
    )
    parser.add_argument(
        '-i', "--iterations", type=int,
        help="number of iterations per agent",
        required=False, default=50
    )
    args = vars(parser.parse_args())
    print("rewrite")
    evaluate(**args)
    # print("Benchmark")
    # benchmark(**args)
    import subprocess
    print("old")
    subprocess.run(
        "/home/jacob/Documents/thesis/.env/bin/python "
        f"/home/jacob/Documents/thesis/OnlineCMA-ES/src/main.py "
        f"-f {args['functionid']} -d {args['dim']} -i {args['iterations']} --clear", shell=True
    )


if __name__ == "__main__":
    main()
