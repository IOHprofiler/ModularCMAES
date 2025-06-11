from time import perf_counter

import numpy as np
import modcma.c_maes as modcma
import ioh
import pandas as pd 

from multiprocessing import Pool


from pprint import pprint

np.random.seed(12)

def run_modma(problem: ioh.ProblemType, 
              x0: np.ndarray, 
              logger_obj,
              matrix_adaptation = modcma.options.COVARIANCE
            ):
    modules = modcma.parameters.Modules()
    modules.matrix_adaptation = matrix_adaptation

    settings = modcma.Settings(
        problem.meta_data.n_variables, 
        x0=x0,
        modules=modules,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub, 
        verbose=True,
        sigma0=2.0,
        target=problem.optimum.y + 1e-8,
        budget=problem.meta_data.n_variables * 100_000
    )
    
    cma = modcma.ModularCMAES(settings)

    start = perf_counter()
    while not cma.break_conditions():
        if cma.p.criteria.any():
            logger_obj.update(cma.p.criteria.items)
        cma.step(problem)

    cma.run(problem)
    stop = perf_counter()
    elapsed = stop - start
    return elapsed, cma.p.stats.t, problem.state.evaluations, cma.p.stats.n_updates


class RestartCollector:
    def __init__(self, strategy = modcma.options.RestartStrategy.NONE):
        modules = modcma.parameters.Modules()
        modules.restart_strategy = strategy
        settings = modcma.Settings(
            2, 
            modules=modules,
        )
        cma = modcma.ModularCMAES(settings)
        self.names = [x.name for x in cma.p.criteria.items]
        self.reset()
    
    def update(self, items):
        for item in items:
            if item.met:
                setattr(self, item.name, getattr(self, item.name) + 1)


    def reset(self):
        for item in self.names:
            setattr(self, item, 0)


if __name__ == "__main__":
    dims = 2, 3, 5, 10, 20, 40, 100
    functions = [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    n_repeats = 100
    options = modcma.options.MatrixAdaptationType.__members__
    del options['COVARIANCE_NO_EIGV']


    def collect(name, option):
        logger = ioh.logger.Analyzer(
            folder_name=name, 
            algorithm_name=name,
            root="data"
        )
        collector = RestartCollector()
        logger.add_run_attributes(collector, collector.names)
        for fid in functions:
            for d in dims:
                problem = ioh.get_problem(fid, 1, d)
                problem.attach_logger(logger)
                for i in range(n_repeats):
                    modcma.utils.set_seed(21 + fid * d * i)
                    collector.reset()
                    run_modma(problem, np.zeros(d), collector, option)
                    print(name, fid, d, problem.state.current_best_internal.y, problem.state.evaluations)
                    problem.reset()

    with Pool(len(options)) as p:
        p.starmap(collect, options.items())

    # problem = ioh.get_problem(fid, 1, d)
    # run_modma(problem, np.zeros(d), modcma.options.CMSA)
    # print(problem.state.evaluations, problem.state.current_best_internal.y)
