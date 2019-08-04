import itertools
import numpy as np


class Individual:
    def __init__(self, d, genome=None, last_z=None, mutation_vector=None, fitness=None):
        self.d = d
        self.fitness = fitness or np.inf
        if type(genome) == np.ndarray:
            self.genome = genome.copy()
        else:
            self.genome = np.ones((d, 1))

        if type(last_z) == np.ndarray:
            self.last_z = last_z.copy()
        else:
            self.last_z = np.zeros((d, 1))

        if type(mutation_vector) == np.ndarray:
            self.mutation_vector = mutation_vector.copy()
        else:
            self.mutation_vector = np.zeros((d, 1))

    def mutate(self, parameters):
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

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __repr__(self):
        return f"<Individual d:{self.d} fitness:{self.fitness} x1:{self.genome[0]}>"


class Population:
    def __init__(self, individuals):
        self.individuals = individuals

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.individuals[key]
        elif isinstance(key, slice):
            return Population(
                list(itertools.islice(
                    self.individuals, key.start, key.stop, key.step)))
        else:
            raise KeyError("Key must be non-negative integer or slice, not {}"
                           .format(key))

    def recombine(self, parameters) -> "Population":
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

    @property
    def genomes(self):
        return np.column_stack(
            [ind.genome for ind in self.individuals])

    @property
    def fitnesses(self):
        return np.array([ind.fitness for ind in self.individuals])

    @property
    def mutation_vectors(self):
        return np.column_stack(
            [ind.mutation_vector for ind in self.individuals])

    @property
    def last_zs(self):
        return np.column_stack(
            [ind.last_z for ind in self.individuals])

    @property
    def best_individual(self):
        return self.individuals[np.argmin(self.fitnesses)]

    @staticmethod
    def new_population(n, d, genome=None) -> 'Population':
        return Population(
            [Individual(d, genome) for _ in range(n)]
        )

    @property
    def d(self):
        return self.best_individual.d

    @property
    def n(self):
        return len(self.individuals)

    def copy(self):
        return Population(
            [Individual(**i.__dict__) for i in self.individuals]
        )

    def sort(self):
        self.individuals.sort()

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        return Population(
            [Individual(**i.__dict__)
             for i in (self.individuals + other.individuals)]
        )

    def __repr__(self):
        return f"<Population d:{self.d} n:{self.n}>"

    def __str__(self):
        return self.__repr__()
