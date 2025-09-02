from time import perf_counter
import warnings

import numpy as np
import modcma.c_maes as modcma
import ioh
import pandas as pd 
import matplotlib.pyplot as plt
import cma as pycma

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


def run_pycma(problem: ioh.ProblemType, x0: np.ndarray, max_generations=1000):
    options = pycma.CMAOptions()
    options['CMA_active'] = False
    # options['maxfevals'] = n_evaluations
    options['conditioncov_alleviate'] = False
    options["verbose"] = 10
    options["CMA_diagonal"] = False
    pprint(options)

    cma = pycma.CMAEvolutionStrategy(x0, 2.0, options=options)
    settings = modcma.Settings(problem.meta_data.n_variables)
    assert settings.lambda0 == cma.sp.popsize
    np.random.seed(1)
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(max_generations):
            X, y = cma.ask_and_eval(problem)
            cma.tell(X, y)
    stop = perf_counter()
    elapsed = stop - start

    return elapsed, cma.countiter, problem.state.evaluations, cma.sm.count_eigen

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
    fid = 2
    dims = 2, 3, 5, 10, 20, 40, 100, 200, 500, 1000
    n_repeats = 15

    stats = []
    for d in dims:
        for _ in range(n_repeats):
            problem = ioh.get_problem(fid, 1, d)
            time, n_gen, n_evals, n_updates = run_pycma(problem, np.zeros(d))
            stats.append(("pycma", d, time, n_gen, n_evals, n_updates))
            print(stats[-1])
    stats = pd.DataFrame(stats, columns=["method", "dim", "time", "n_gen", "n_evals", "n_updates"])
    stats.to_csv("time_stats_pycma.csv")
    print(stats)
    