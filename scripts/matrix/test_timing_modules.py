from time import perf_counter

import numpy as np
import modcma.c_maes as modcma
import ioh
import pandas as pd 
import matplotlib.pyplot as plt

from pprint import pprint

np.random.seed(12)

def run_modma(problem: ioh.ProblemType, x0: np.ndarray, matrix_adaptation = modcma.options.COVARIANCE, max_generations=1000):
    modules = modcma.parameters.Modules()
    modules.matrix_adaptation = matrix_adaptation
    settings = modcma.Settings(
        problem.meta_data.n_variables, 
        x0=x0,
        modules=modules,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub, 
        verbose=True,
        max_generations=max_generations
    )
    
    cma = modcma.ModularCMAES(settings)

    start = perf_counter()
    cma.run(problem)
    stop = perf_counter()
    elapsed = stop - start
    assert cma.p.stats.t == max_generations
    return elapsed, cma.p.stats.t, problem.state.evaluations, cma.p.stats.n_updates


def collect():
    fid = 2
    dims = 2, 3, 5, 10, 20, 40, 100, 200, 500, 1000
    
    n_repeats = 15
    options = modcma.options.MatrixAdaptationType.__members__
    del options['COVARIANCE_NO_EIGV']

    pprint(options)

    stats = []
    for d in dims:
        for name, option in options.items():
            for _ in range(n_repeats):
                problem = ioh.get_problem(fid, 1, d)
                time, n_gen, n_evals, n_updates = run_modma(problem, np.zeros(d), option)
                stats.append((name, d, time, n_gen, n_evals, n_updates))
                print(stats[-1])

    stats = pd.DataFrame(stats, columns=["method", "dim", "time", "n_gen", "n_evals", "n_updates"])
    stats.to_csv("time_stats.csv")
    print(stats)


if __name__ == "__main__":
    stats = pd.read_csv("time_stats.csv")
    print(stats)