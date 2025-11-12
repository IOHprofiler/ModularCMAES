import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import argparse
from itertools import product
from functools import partial

import ioh
import numpy as np
from smac import Scenario
from smac.acquisition.function import PriorAcquisitionFunction
from smac import AlgorithmConfigurationFacade, HyperparameterOptimizationFacade
from ConfigSpace import Configuration

from modcma import c_maes


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def calc_aoc(logger, budget, fid, iid, dim):
    data = logger.data()
    data1 = data["None"][fid][dim][iid][0]
    fvals = [x["raw_y_best"] for x in data1.values()]
    fvals = np.array(fvals)
    if np.isnan(fvals).any():
        np.nan_to_num(fvals, copy=False, nan=1e8)
    if len(fvals) < budget:
        fvals = np.concatenate([fvals, (budget - len(fvals)) * [np.min(fvals)]])
    parts = np.log10(np.clip(fvals[:budget], 1e-8, 1e2)) + 8
    return np.mean(parts) / 10


def get_bbob_performance(
    config: Configuration, instance: str, seed: int = 0, dim: int = 5
):
    fid, iid = instance.split(",")
    fid = int(fid[1:])
    iid = int(iid[:-1])
    np.random.seed(seed + iid)
    c_maes.utils.set_seed(seed + iid)
    BUDGET = dim * 10_000
    l3 = ioh.logger.Store(
        triggers=[ioh.logger.trigger.ALWAYS], properties=[ioh.logger.property.RAWYBEST]
    )
    problem = ioh.get_problem(fid, iid, dim)
    problem.attach_logger(l3)

    settings = c_maes.settings_from_config(
        dim, 
        config, 
        budget=BUDGET, 
        target= problem.optimum.y + 1e-9,
        ub=problem.bounds.ub,
        lb=problem.bounds.lb
    )
    par = c_maes.Parameters(settings)

    try:
        cma = c_maes.ModularCMAES(par)
        cma.run(problem)
    except Exception as e:
        print(
            f"Found target {problem.state.current_best.y} target, but exception ({e}), so run failed"
        )
        print(config)
        return [np.inf]
    
    auc = calc_aoc(l3, BUDGET, fid, iid, dim)
    return [auc] 


def run_smac(fid, dim, use_learning_rates):
    print(f"Running SMAC with fid={fid}, lr={use_learning_rates} and d={dim}")
    cma_cs = c_maes.get_configspace(dim, add_learning_rates=use_learning_rates)

    iids = range(1, 51)
    if fid == "all":
        fids = range(1, 25)
        max_budget = 240
    else:
        fids = [fid]
        max_budget = 50

    args = list(product(fids, iids))
    np.random.shuffle(args)
    inst_feats = {str(arg): [arg[0]] for idx, arg in enumerate(args)}
    scenario = Scenario(
        cma_cs,
        name=str(int(time.time())) + "-" + "CMA",
        deterministic=False,
        n_trials=50_000,
        instances=args,
        instance_features=inst_feats,
        output_directory=os.path.join(
            DATA_DIR, f"BBOB_F{fid}_{dim}D_LR{use_learning_rates}"
        ),
        n_workers=1,
    )
    pf = PriorAcquisitionFunction(
        acquisition_function=AlgorithmConfigurationFacade.get_acquisition_function(
            scenario
        ),
        decay_beta=scenario.n_trials / 10,
    )
    eval_func = partial(get_bbob_performance, dim=dim)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario, max_config_calls=max_budget
    )
    smac = HyperparameterOptimizationFacade(
        scenario, eval_func, acquisition_function=pf, intensifier=intensifier
    )
    smac.optimize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=int, default=1)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--use_learning_rates", action="store_true")
    args = parser.parse_args()
    run_smac(args.fid, args.dim, args.use_learning_rates)
