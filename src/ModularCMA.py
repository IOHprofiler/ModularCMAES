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
            zi = self.parameters.sampler.next()
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


def run_once(fid=1, **kwargs):
    np.random.seed(12)
    evals, fopts = evaluate(
        fid, 5, ModularCMA, iterations=10, **kwargs)


if __name__ == "__main__":
    test_modules()
    fid = 7
    # run_once(fid=fid, bound_correction=False)
    # run_once(fid=fid, bound_correction=True)

    # functions = [20, 22, 23, 24]
    # [3, 4, 16, 17, 18, 19, 21]

    # [1, 2, 5, 6] + list(range(8, 15))  # + [20, 22, 23, 24]
    # d = 20
    # for fid in range(1, 25):
    # evals, fopts = evaluate(
    # fid, d, ModularCMA, logging=True, label="canon")


# mirrored
# Optimizing function 1 in 5D for target 79.48 + 1e-08
# FCE:    79.48000001         0.0000
# ERT:      583.2000         30.0293
# Time elapsed 0.5260462760925293
# Orthogonal
# Optimizing function 1 in 5D for target 79.48 + 1e-08
# FCE:    79.48000001         0.0000
# ERT:      682.4000         42.7907
# Time elapsed 0.7957170009613037
# Sequential
# Optimizing function 1 in 5D for target 79.48 + 1e-08
# FCE:    79.48000001         0.0000
# ERT:      670.7000         20.5234
# Time elapsed 0.6130139827728271
# **************************************************
# mirrored
# Optimizing function 2 in 5D for target -209.88 + 1e-08
# FCE:    -209.87999999       0.0000
# ERT:     2093.6000        153.5521
# Time elapsed 2.687499761581421
# Orthogonal
# Optimizing function 2 in 5D for target -209.88 + 1e-08
# FCE:    -209.87999999       0.0000
# ERT:     2051.2000        131.3657
# Time elapsed 3.267056941986084
# Sequential
# Optimizing function 2 in 5D for target -209.88 + 1e-08
# FCE:    -209.87999999       0.0000
# ERT:     2140.2000        129.0913
# Time elapsed 2.912632465362549
# **************************************************
