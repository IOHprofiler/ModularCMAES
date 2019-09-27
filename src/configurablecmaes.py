import numpy as np
from .optimizer import Optimizer
from .utils import evaluate, _correct_bounds, _scale_with_threshold
from .parameters import Parameters
from .population import Population


class ConfigurableCMAES(Optimizer):
    def __init__(
            self,
            fitness_func,
            *args,
            **kwargs
    ):
        self.parameters = Parameters(*args, **kwargs)
        self._fitness_func = fitness_func

    def tpa_mutation(self, x, y, f):
        yi = ((self.parameters.m - self.parameters.m_old) /
              self.parameters.sigma)
        y.extend([yi, -yi])
        x.extend([
            self.parameters.m + (self.parameters.sigma * yi),
            self.parameters.m + (self.parameters.sigma * -yi)
        ])
        f.extend(list(map(self.fitness_func, x)))
        if f[1] < f[0]:
            self.parameters.rank_tpa = -self.parameters.a_tpa
        else:
            self.parameters.rank_tpa = (
                self.parameters.a_tpa + self.parameters.b_tpa)

    def mutate(self):
        '''Method performing mutation and evaluation of a set
        of individuals.'''
        y, x, f = [], [], []
        n_offspring = self.parameters.lambda_
        if self.parameters.step_size_adaptation == 'tpa' and self.parameters.old_population:
            n_offspring -= 2
            print(len(x))
            self.tpa_mutation(x, y, f)
            print(len(x))

        for i in range(n_offspring):
            zi = next(self.parameters.sampler)
            if self.parameters.threshold_convergence:
                zi = _scale_with_threshold(zi, self.parameters.threshold)

            yi = np.dot(self.parameters.B, self.parameters.D * zi)
            xi = self.parameters.m + (self.parameters.sigma * yi)
            if self.parameters.bound_correction:
                xi = _correct_bounds(
                    xi, self.parameters.ub, self.parameters.lb)
            fi = self.fitness_func(xi)
            [a.append(v) for a, v in ((y, yi), (x, xi), (f, fi),)]

            if self.sequential_break_conditions(i, fi):
                break

        self.parameters.population = Population(
            np.hstack(x),
            np.hstack(y),
            np.array(f))

    def select(self):
        if self.parameters.selection == 'pairwise':
            assert len(self.parameters.population.f) % 2 == 0, (
                'Cannot perform pairwise selection with '
                'an odd number of indivuduals')
            indices = [int(np.argmin(x) + (i * 2))
                       for i, x in enumerate(
                np.split(self.parameters.population.f,
                         len(self.parameters.population.f) // 2))
                       ]
            self.parameters.population = self.parameters.population[indices]

        if self.parameters.elitist and self.parameters.old_population:
            self.parameters.population += self.parameters.old_population[
                : self.parameters.mu]

        self.parameters.population.sort()

        self.parameters.population = self.parameters.population[
            : self.parameters.lambda_]

        self.parameters.fopt = min(
            self.parameters.fopt, self.parameters.population.f[0])

    def recombine(self):
        self.parameters.m_old = self.parameters.m.copy()
        self.parameters.m = self.parameters.m_old + (1 * (
            (self.parameters.population.x[:, :self.parameters.mu] -
                self.parameters.m_old) @
            self.parameters.pweights).reshape(-1, 1)
        )

    def step(self):
        self.mutate()
        self.select()
        self.recombine()
        self.parameters.adapt()
        return not any(self.break_conditions)

    def sequential_break_conditions(self, i: int, f) -> bool:
        '''Add docstring'''
        if self.parameters.sequential:
            return (f < self.parameters.fopt and
                    i > self.parameters.seq_cutoff)
        return False
