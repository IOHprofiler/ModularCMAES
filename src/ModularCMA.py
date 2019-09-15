'''
TODO: Implementing modules
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


import numpy as np
from Optimizer import Optimizer
from Utils import evaluate
from Parameters import Parameters
from Population import Population


def _scale_with_threshold(z, threshold):
    length = np.linalg.norm(z)
    if length < threshold:
        new_length = threshold + (threshold - length)
        z *= (new_length / length)
    return z


def _correct_bounds(x, ub, lb):
    out_of_bounds = np.logical_or(x > ub, x < lb)
    y = (x[out_of_bounds] - lb) / (ub - lb)
    x[out_of_bounds] = lb + (
        ub - lb) * (1. - np.abs(y - np.floor(y)))
    return x


class ModularCMA(Optimizer):
    def __init__(
            self,
            fitness_func,
            absolute_target,
            d,
            rtol,
            **kwargs
    ):
        self.parameters = Parameters(d, absolute_target, rtol, **kwargs)
        self._fitness_func = fitness_func

    def mutate(self):
        y, x, f = [], [], []
        for i in range(self.parameters.lambda_):
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
            assert self.parameters.lambda_ % 2 == 0, (
                'Cannot perform pairwise selection with '
                'an odd number of indivuduals')
            indices = [int(np.argmin(x) + (i * 2))
                       for i, x in enumerate(
                           np.split(self.parameters.population.f,
                                    self.parameters.lambda_ // 2))
                       ]
            self.parameters.population = self.parameters.population[indices]

        if self.parameters.elitist and self.parameters.old_population:
            self.parameters.population += self.parameters.old_population[
                :self.parameters.mu]

        self.parameters.population.sort()

        self.parameters.population = self.parameters.population[
            :self.parameters.lambda_]

        self.parameters.old_population = self.parameters.population.copy()

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


def test_modules():
    from CannonicalCMA import CannonicalCMA
    iterations = 10
    for i in [1, 2]:  # , 5, 6, 8, 9, 10, 11, 12]:
        # print("cannon")
        # np.random.seed(12)
        # evals, fopts = evaluate(
        #     i, 5, CannonicalCMA, iterations=iterations)
        # print("new")
        # np.random.seed(12)
        # evals, fopts = evaluate(
        #     i, 5, ModularCMA, iterations=iterations)

        # print("active")
        # np.random.seed(12)
        # evals, fopts = evaluate(
        #     i, 5, ModularCMA, iterations=iterations, active=True)

        # print("elist")
        # np.random.seed(12)
        # evals, fopts = evaluate(
        #     i, 5, ModularCMA, iterations=iterations, elitist=True)

        print("mirrored")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations, mirrored=True)

        print("Orthogonal")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations, orthogonal=True)

        print("Sequential")
        np.random.seed(12)
        evals, fopts = evaluate(
            i, 5, ModularCMA, iterations=iterations, sequential=True)

        # print("Threshold Convergence")
        # np.random.seed(12)
        # evals, fopts = evaluate(
        #     i, 5, ModularCMA, iterations=iterations, threshold_convergence=True)

        # print("Bound correction")
        # np.random.seed(12)
        # evals, fopts = evaluate(
        #     i, 5, ModularCMA, iterations=iterations, bound_correction=True)

        print('*' * 50)
        # print()


def run_function(fid=1, iterations=20, **kwargs):
    np.random.seed(42)
    evals, fopts = evaluate(
        fid, 5, ModularCMA, iterations=iterations, **kwargs)


if __name__ == "__main__":
    # test_modules()
    fid = 7
    run_function(fid=fid, orthogonal=True)
    run_function(fid=fid, mirrored=True)
    run_function(fid=fid, mirrored=True, selection='pairwise')

    # for fid in (1, 2, 8, 9, 10):
    #     # print("std")
    #     # # print("mirrored old")
    #     # # run_function(fid=fid, mirrored=True, old_samplers=True)
    #     # # print("mirrored new")
    #     # # run_function(fid=fid, mirrored=True, old_samplers=False)
    #     # print()
    #     # print("orth old")
    #     # run_function(fid=fid, orthogonal=True, old_samplers=True)
    #     print("orth new")
    #     run_function(fid=fid, orthogonal=True, old_samplers=False)
    #     print()
    #     # print("mirrored orth old")
    #     # run_function(fid=fid, orthogonal=True, mirrored=True, old_samplers=True)
    #     print("mirrored orth new")
    #     run_function(fid=fid, orthogonal=True,
    #                  mirrored=True, old_samplers=False)
    #     print("*" * 60)
    #     print()
    #     print()
    # run_function(fid=fid, bound_correction=True)
