import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import argparse
from functools import partial

import ioh
import numpy as np

from smac import Scenario,AlgorithmConfigurationFacade
from smac.main.config_selector import ConfigSelector
from ConfigSpace import Configuration

from modcma import c_maes


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def get_bbob_performance(
    config: Configuration, seed: int = 0, fid: int = 0,  dim: int = 5
):
    iid = 1 + (seed % 10)
    np.random.seed(seed + iid)
    c_maes.utils.set_seed(seed + iid)
    BUDGET = dim * 10_000

    problem = ioh.get_problem(fid, iid, dim)

    settings = c_maes.settings_from_config(
        dim, 
        config, 
        budget=BUDGET, 
        target=problem.optimum.y + 9e-9,
        ub=problem.bounds.ub,
        lb=problem.bounds.lb
    )
    par = c_maes.Parameters(settings)

    try:
        cma = c_maes.ModularCMAES(par)
        cma.run(problem)
        if not problem.state.final_target_found:
            return BUDGET 
        return problem.state.evaluations
    except Exception as e:
        print(
            f"Found target {problem.state.current_best.y} target, but exception ({e}), so run failed"
        )
        return np.inf


def run_smac(fid, dim, use_learning_rates, add_popsize, add_sigma, n_workers):
    print(f"Running SMAC with fid={fid}, lr={use_learning_rates} and d={dim}")
    cma_cs = c_maes.get_configspace(
        dim, add_learning_rates=use_learning_rates, add_popsize=add_popsize, add_sigma=add_sigma
    )

    scenario = Scenario(
        cma_cs,
        name=str(int(time.time())) + "-" + "CMA",
        deterministic=False,
        n_trials=100_000,
        output_directory=os.path.join(
            DATA_DIR, f"BBOB_F{fid}_{dim}D_LR{use_learning_rates}"
        ),
        n_workers=n_workers,
    )

    eval_func = partial(get_bbob_performance, fid=fid, dim=dim)
    config_selector = ConfigSelector(
        scenario,
        retrain_after=500,
        min_trials=1000,
        retries=16,

    )
    smac = AlgorithmConfigurationFacade(
        scenario, eval_func, 
        intensifier=AlgorithmConfigurationFacade.get_intensifier(
            scenario, max_config_calls=50
        ),
        config_selector=config_selector
    )
    smac.optimize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=int, default=1)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--use_learning_rates", action="store_true")
    parser.add_argument("--add_popsize", action="store_true")
    parser.add_argument("--add_sigma", action="store_true")
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args()
    
    run_smac(
        args.fid, 
        args.dim, 
        args.use_learning_rates, 
        args.add_popsize, 
        args.add_sigma, 
        args.n_workers
    )
