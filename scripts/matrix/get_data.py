from time import perf_counter
import warnings
import numpy as np
import modcma.c_maes as modcma
import ioh
import pandas as pd 

from multiprocessing import Pool, Process
import cma as pycma

from pprint import pprint

np.random.seed(12)

DIMS = 2, 3, 5, 10, 20, 40, #100
FUNCTIONS = list(range(1, 25))
N_REPEATS = 100
BUDGET = 100_000
ROOT = "data"


def ert(runs, target = 1e-8):
    total_evals = 0
    n_suc = 0
    for row in runs:
        total_evals += row['evals']
        if row['best_y'] <= target:
            n_suc += 1

    if n_suc <= 0:
        return float("inf")
    return total_evals / n_suc

def run_modma(problem: ioh.ProblemType, 
              x0: np.ndarray, 
              logger_obj,
              matrix_adaptation = modcma.options.COVARIANCE
            ):
    modules = modcma.parameters.Modules()
    modules.matrix_adaptation = matrix_adaptation
    modules.ssa = modcma.options.StepSizeAdaptation.CSA
    modules.restart_strategy = modcma.options.RestartStrategy.STOP

    settings = modcma.Settings(
        problem.meta_data.n_variables, 
        x0=x0,
        modules=modules,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub, 
        verbose=True,
        sigma0=2.0,
        target=problem.optimum.y + 1e-8,
        budget=problem.meta_data.n_variables * BUDGET,
    )

    cma = modcma.ModularCMAES(settings)
    start = perf_counter()
    while not cma.break_conditions():
        if cma.p.criteria.any():
            logger_obj.update(cma.p.criteria.items)
        cma.step(problem)

    if cma.p.criteria.any():
        logger_obj.update(cma.p.criteria.items)

    stop = perf_counter()
    elapsed = stop - start
    return elapsed, cma.p.stats.t, problem.state.evaluations, cma.p.stats.n_updates


class RestartCollector:
    def __init__(self, strategy = modcma.options.RestartStrategy.STOP):
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

def collect(name, option):
    logger = ioh.logger.Analyzer(
        folder_name=name, 
        algorithm_name=name,
        root=ROOT
    )
    collector = RestartCollector()
    logger.add_run_attributes(collector, collector.names)
    for fid in FUNCTIONS:
        for d in DIMS:
            problem = ioh.get_problem(fid, 1, d)
            problem.attach_logger(logger)
            runs = []
            for i in range(N_REPEATS):
                modcma.utils.set_seed(21 + fid * d * i)
                collector.reset()
                run_modma(problem, np.zeros(d), collector, option)
                # print(name, fid, d, problem.state.current_best_internal.y, problem.state.evaluations)
                runs.append(dict(evals=problem.state.evaluations, best_y=problem.state.current_best_internal.y))
                problem.reset()
            print(name, fid, d, "ert:", ert(runs))

def collect_modcma():
    options = modcma.options.MatrixAdaptationType.__members__
    del options['COVARIANCE_NO_EIGV']

    with Pool(len(options)) as p:
        p.starmap(collect, options.items())


def run_pycma(problem: ioh.ProblemType, x0: np.ndarray):
    options = pycma.CMAOptions()
    options['CMA_active'] = False
    options["verbose"] = -1
    options["CMA_diagonal"] = False
    options['conditioncov_alleviate'] = False
    options['ftarget'] = problem.optimum.y + 1e-8
    options['maxfevals'] = problem.meta_data.n_variables * BUDGET

    cma = pycma.CMAEvolutionStrategy(x0, 2.0, options=options)
    settings = modcma.Settings(problem.meta_data.n_variables)
    assert settings.lambda0 == cma.sp.popsize
    start = perf_counter()

    target = problem.optimum.y + 1e-8
    budget = problem.meta_data.n_variables * BUDGET

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while problem.state.evaluations < budget:
            X, y = cma.ask_and_eval(problem)
            cma.tell(X, y)

            if problem.state.current_best.y <= target:
                break
            if cma.stop():
                break


    stop = perf_counter()
    elapsed = stop - start

    return elapsed, cma.countiter, problem.state.evaluations, cma.sm.count_eigen

def collect_pycma():
    logger = ioh.logger.Analyzer(
        folder_name="pycma", 
        algorithm_name="pycma",
        root=ROOT
    )
    for fid in FUNCTIONS:
        for d in DIMS:
            problem = ioh.get_problem(fid, 1, d)
            problem.attach_logger(logger)
            for i in range(N_REPEATS):
                np.random.seed(21 + fid * d * i)
                run_pycma(problem, np.zeros(d))
                print("pycma", fid, d, problem.state.current_best_internal.y, problem.state.evaluations)
                problem.reset()



if __name__ == "__main__":
    # p1 = Process(target=collect_modcma)
    # p2 = Process(target=collect_pycma)

    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
    collect_modcma()