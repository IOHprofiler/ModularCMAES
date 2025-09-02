import sys
import inspect
import warnings

from time import perf_counter
from pprint import pprint
from dataclasses import dataclass


import numpy as np
from modcma import ModularCMAES
import modcma.c_maes as modcma
import cma as pycma
import ioh
from fcmaes import optimizer, retry

np.random.seed(12)

def timeit(f):
    def inner(*args, **kwargs):
        start = perf_counter()
        result = f(*args, **kwargs)
        stop = perf_counter()
        elapsed = stop - start
        return elapsed
    return inner


# @timeit
# def run_modcmapy(f: ioh.ProblemType, dim: int, n_evaluations, x0: np.ndarray):
#     cma = ModularCMAES(f, dim, budget=n_evaluations, x0=x0)
#     cma.run()
#     assert f.state.evaluations >= n_evaluations
    
    
# @timeit
# def run_fcmaes(f: ioh.ProblemType, dim: int, n_evaluations, x0: np.ndarray):
    
#     lamb = 4 + np.floor(3 * np.log(dim)).astype(int)
#     bounds = np.array([f.bounds.lb, f.bounds.ub])
#     res = optimizer.cmaescpp.minimize(
#         f, x0=x0, max_evaluations=n_evaluations,
#         stop_hist=0, accuracy=1e-10, stop_fitness=-700,
#         popsize=lamb, workers=1, delayed_update=False
#     )
    
        
#     # ret = retry.minimize(f, bounds.T, optimizer=optimizer.Cma_cpp(n_evaluations))
#     assert f.state.evaluations >= n_evaluations
#     print(f.state.current_best_internal.y)


# @timeit
# def run_modma(f: ioh.ProblemType, dim: int, n_evaluations, x0: np.ndarray):
#     modcma.constants.calc_eigv = False
#     modules = modcma.parameters.Modules()
#     # modules.sample_transformation = modcma.options.SCALED_UNIFORM
#     modules.matrix_adaptation = modcma.options.COVARIANCE
#     settings = modcma.Settings(dim, 
#                                budget=n_evaluations, 
#                                x0=x0,
#                                modules=modules,
#                                lb=f.bounds.lb,
#                                ub=f.bounds.ub, 
#                                verbose=True
#                             )
    
#     cma = modcma.ModularCMAES(settings)
    
    
#     maxp = 1/(10 * dim * (cma.p.weights.c1 +cma.p.weights.cmu))
#     # print(dim, max(1, maxp), maxp)
#     # breakpoint()

#     while cma.step(f):
#         pass          
#     # cma.run(f)
#     print(cma.p.stats.t, cma.p.stats.n_updates, f.state.current_best_internal.y)
#     assert f.state.evaluations >= n_evaluations
#     return cma


@timeit
def run_pycma(f: ioh.ProblemType, dim: int, n_evaluations: int, x0: np.ndarray):
    options = pycma.CMAOptions()
    # options['CMA_active'] = False
    # options['maxfevals'] = n_evaluations
    options["verbose"] = -1
    options["CMA_diagonal"] = False
    # pprint(options)

    cma = pycma.CMAEvolutionStrategy(x0, 2.0, options=options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while f.state.evaluations < n_evaluations:
            X, y = cma.ask_and_eval(f)
            cma.tell(X, y)
            breakpoint()
            # cma.disp()
    assert f.state.evaluations >= n_evaluations


if __name__ == "__main__":
    n_iters = 2
    n_evals = 2_000
    fid = 12
    dimensions = [5]
    names, functions = zip(
        *[
            (name, obj)
            for name, obj in inspect.getmembers(sys.modules[__name__])
            if name.startswith("run")
        ]
    )
    data = {name: dict.fromkeys(dimensions) for name in names}

    for d in dimensions:
        x0 = np.random.uniform(size=d)
        for name, function in zip(names, functions):
            data[name][d] = np.array(
                [
                    function(ioh.get_problem(fid, 1, d), d, n_evals * d, x0)
                    for _ in range(n_iters)
                ]
            )

        print(f"fid: {fid} ({d}D) budget: {d * n_evals}")
        for name in names:
            print(name, data[name][d].mean(), data[name][d].std())
