import itertools
from typing import Optional, List, Any
import numpy as np


class Individual:
    def __init__(
        self,
        d: int,
        genome: Optional[np.ndarray] = None,
        last_z: Optional[np.ndarray] = None,
        mutation_vector: Optional[np.ndarray] = None,
        fitness: Optional[float] = None
    ) -> None:
        self.d = d
        self.fitness = fitness or np.inf

        for name in ("genome", "last_z", "mutation_vector"):
            value = eval(name)
            if type(value) == np.ndarray:
                setattr(self, name, value.copy())
            else:
                setattr(self, name, np.ones((self.d, 1)))

    def mutate(self, parameters: "Parameters") -> None:
        self.last_z = parameters.sampler.next()

        if parameters.threshold_convergence:
            # Scale with threshold function
            length = np.linalg.norm(self.last_z)
            if length < parameters.threshold:
                self.last_z *= ((
                    parameters.threshold + (parameters.threshold - length)
                ) / length)

        self.mutation_vector = np.dot(
            parameters.B, (parameters.D * self.last_z))

        self.genome = np.add(self.genome,
                             self.mutation_vector * parameters.sigma
                             )
        # correct out of bound
        out_of_bounds = np.logical_or(
            self.genome > parameters.ub, self.genome < parameters.lb)
        y = (self.genome[out_of_bounds] - parameters.lb) / \
            (parameters.ub - parameters.lb)

        self.genome[out_of_bounds] = parameters.lb + (
            parameters.ub - parameters.lb) * (
            1. - np.abs(y - np.floor(y))
        )

    def __lt__(self, other: "Individual") -> bool:
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        return f"<Individual d:{self.d} fitness:{self.fitness} x1:{self.genome[0]}>"


class Population:
    def __init__(self, individuals: List['Individual']) -> None:
        self.individuals = individuals

    def recombine(self, parameters: "Parameters") -> "Population":
        '''There is only one function used by all CMA-ES
        variants, only the recombination weights are different '''

        parameters.wcm_old = parameters.wcm.copy()
        parameters.wcm = np.dot(
            self.genomes,
            parameters.recombination_weights
        )
        return Population.new_population(
            parameters.lambda_, parameters.d,
            parameters.wcm
        )

    def copy(self) -> "Population":
        return Population(
            [Individual(**i.__dict__) for i in self.individuals]
        )

    def sort(self) -> None:
        self.individuals.sort()

    @property
    def genomes(self) -> np.ndarray:
        return np.column_stack(
            [ind.genome for ind in self.individuals])

    @property
    def fitnesses(self) -> np.ndarray:
        return np.array([ind.fitness for ind in self.individuals])

    @property
    def mutation_vectors(self) -> np.ndarray:
        return np.column_stack(
            [ind.mutation_vector for ind in self.individuals])

    @property
    def last_zs(self) -> np.ndarray:
        return np.column_stack(
            [ind.last_z for ind in self.individuals])

    @property
    def best_individual(self) -> "Individual":
        return self.individuals[np.argmin(self.fitnesses)]

    @staticmethod
    def new_population(n: int, d: int, genome: Optional[np.ndarray] = None
                       ) -> 'Population':
        return Population(
            [Individual(d, genome) for _ in range(n)]
        )

    @property
    def d(self) -> int:
        return self.best_individual.d

    @property
    def n(self) -> int:
        return len(self.individuals)

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, int):
            return self.individuals[key]
        elif isinstance(key, slice):
            return Population(
                list(itertools.islice(
                    self.individuals, key.start, key.stop, key.step)))
        else:
            raise KeyError("Key must be non-negative integer or slice, not {}"
                           .format(key))

    def __add__(self, other: "Population") -> "Population":
        assert isinstance(other, self.__class__)
        return Population(
            [Individual(**i.__dict__)
             for i in (self.individuals + other.individuals)]
        )

    def __repr__(self) -> str:
        return f"<Population d:{self.d} n:{self.n}>"

    def __str__(self) -> str:
        return self.__repr__()
