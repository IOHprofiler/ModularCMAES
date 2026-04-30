import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import argparse
from functools import partial

import ioh
import numpy as np

from smac import Scenario, AlgorithmConfigurationFacade
from smac.acquisition.maximizer import (
    LocalAndSortedRandomSearch,
)
from smac.main.config_selector import ConfigSelector
from ConfigSpace import Configuration, ConfigurationSpace, ForbiddenGreaterThanRelation
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from modcma import c_maes


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def calc_aoc(problem: ioh.ProblemType, logger: ioh.logger.Store, budget: int) -> float:
    """
    Compute the Area Over the Curve (AOC) for an optimization run.

    The AOC summarizes optimization performance over time by averaging
    the log-scaled best-so-far objective values across a fixed evaluation
    budget. Lower values indicate better and faster convergence.

    Steps:
    - Extract best-so-far objective values ("raw_y_best") from the logger.
    - Replace NaNs with a large penalty value (1e8).
    - Pad the trajectory to the full budget using the best observed value.
    - Clip values to [1e-8, 1e2] and apply log10 scaling.
    - Shift values to [0, 10] and normalize to [0, 1].
    - Return the mean over the budget (the AOC score).

    Parameters
    ----------
    problem : ioh.ProblemType
        The evaluated problem
    logger : ioh.logger.Store
        IOH logger containing experiment data.
    budget : int
        Maximum number of function evaluations to consider.
    Returns
    -------
    float
        AOC score in [0, 1], where lower values indicate better performance.
    """
    
    data = logger.data()
    data1 = data['None'][problem.meta_data.problem_id][problem.meta_data.n_variables][problem.meta_data.instance][0]
    fvals = [x['raw_y_best'] for x in data1.values()]
    fvals = np.array(fvals)
    if np.isnan(fvals).any():
        np.nan_to_num(fvals, copy=False, nan=1e8)
    if len(fvals) < budget:
        fvals = np.concatenate([fvals, (budget-len(fvals))*[np.min(fvals)]])
    parts = np.log10(np.clip(fvals[:budget], 1e-8, 1e2))+8
    return np.mean(parts)/10


def get_bbob_performance(
    config: Configuration, seed: int = 0, fid: int = 0,  dim: int = 5
):
    iid = 1 + (seed % 10)
    np.random.seed(seed + iid)
    c_maes.utils.set_seed(seed + iid)
    BUDGET = dim * 10_000

    problem = ioh.get_problem(fid, iid, dim)
    logger = ioh.logger.Store(
        triggers=[ioh.logger.trigger.ALWAYS], 
        properties=[ioh.logger.property.RAWYBEST]
    )
    problem.attach_logger(logger)
    settings = c_maes.settings_from_config(
        dim, 
        config, 
        budget=BUDGET, 
        target=problem.optimum.y + 9e-9,
        ub=problem.bounds.ub,
        lb=problem.bounds.lb
    )
    settings.modules.center_placement = c_maes.options.CenterPlacement.UNIFORM
    par = c_maes.Parameters(settings)

    try:
        cma = c_maes.ModularCMAES(par)
        cma.run(problem)
        aoc = calc_aoc(problem, logger, BUDGET)
    except Exception as e:
        print(
            f"Found target {problem.state.current_best.y} target, but exception ({e}), so run failed"
        )
        aoc = np.inf
    
    extra = {
        "fid": fid,
        "iid": iid,
        "dim": dim,
        "target": float(problem.optimum.y + 9e-9),
        "final_y": float(problem.state.current_best.y),
        "evals": int(problem.state.evaluations),
        "hit_target": bool(problem.state.current_best.y <= problem.optimum.y + 9e-9),
        "precision": float(abs(problem.state.current_best.y - problem.optimum.y)),
    }
    return aoc, extra

def make_new(hp: CategoricalHyperparameter, filter: list[str]):
    new_choices = [c for c in hp.choices if c not in filter]
    return CategoricalHyperparameter(
        name=hp.name,
        choices=new_choices,
        default_value=hp.default_value
    )


def get_configspace(dim, use_learning_rates, add_popsize, add_sigma):
    cma_cs = c_maes.get_configspace(
        dim, add_learning_rates=use_learning_rates, add_popsize=add_popsize, add_sigma=add_sigma
    )
    cs = ConfigurationSpace()
    for hp in cma_cs.values():
        if hp.name in ("sample_sigma", "bound_correction", "center_placement"):
            continue

        if hp.name == "matrix_adaptation":
            cs.add(make_new(hp, ("COVARIANCE_NO_EIGV", )))
            continue

        if hp.name == "restart_strategy":
            cs.add(make_new(hp, ("STOP", )))
            continue

        if hp.name == "sample_transformation":
            cs.add(make_new(hp, ("NONE", )))
            continue

        cs.add(hp)

    if add_popsize:
        cs.add(ForbiddenGreaterThanRelation(cs["mu0"], cs["lambda0"]))
    return cs

def run_smac(fid, dim, use_learning_rates, add_popsize, add_sigma, n_workers):
    print(f"Running SMAC with fid={fid}, lr={use_learning_rates} and d={dim}")
    cs = get_configspace(dim ,use_learning_rates, add_popsize, add_sigma)
    scenario = Scenario(
        cs,
        name=str(int(time.time())) + "-" + "CMA",
        deterministic=False,
        n_trials=100_000,
        output_directory=os.path.join(
            DATA_DIR, f"BBOB_F{fid}_{dim}D_LR{use_learning_rates}{add_popsize}"
        ),
        n_workers=n_workers,
    )

    eval_func = partial(get_bbob_performance, fid=fid, dim=dim)
    config_selector = ConfigSelector(
        scenario,
        retrain_after=250,
        min_trials=500,
        retries=16,
    )
    
    smac = AlgorithmConfigurationFacade(
        scenario, eval_func, 
        intensifier=AlgorithmConfigurationFacade.get_intensifier(
            scenario, max_config_calls=50
        ),
        config_selector=config_selector,
        initial_design = AlgorithmConfigurationFacade.get_initial_design(scenario), 
        model = AlgorithmConfigurationFacade.get_model(
            scenario,
            n_trees=5,
            ratio_features=0.5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=10,
            bootstrapping=True,
            pca_components=13
        ),
        acquisition_maximizer=LocalAndSortedRandomSearch(
            scenario.configspace,
            seed=scenario.seed,
            challengers=500
        )
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
