import ioh
import numpy as np
from ConfigSpace import Configuration

from modcma import c_maes


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
    settings: c_maes.Settings, seed: int = 0, fid: int = 0
):
    iid = 1 + (seed % 10)
    np.random.seed(seed + iid)
    c_maes.utils.set_seed(seed + iid)

    settings.budget = settings.dim * 10_000
    settings.target = problem.optimum.y + 9e-9

    l3 = ioh.logger.Store(
        triggers=[ioh.logger.trigger.ALWAYS], 
        properties=[ioh.logger.property.RAWYBEST]
    )
    problem = ioh.get_problem(fid, iid, settings.dim )
    problem.attach_logger(l3)

    par = c_maes.Parameters(settings)

    try:
        cma = c_maes.ModularCMAES(par)
        cma.run(problem)
    except Exception as e:
        print(
            f"Found target {problem.state.current_best.y} target, but exception ({e}), so run failed"
        )
        return [np.inf]
    
    auc = calc_aoc(l3, settings.budget, fid, iid, settings.dim)
    return auc



def get_ert(
    settings: c_maes.Settings, 
    seed: int = 0, 
    fid: int = 0,
    n_trials: int = 10,
):
    iid =  1 + (seed % 10)
    np.random.seed(seed + iid)
    c_maes.utils.set_seed(seed + iid)

    problem = ioh.get_problem(fid, iid, settings.dim)
    settings.budget = settings.dim * 10_000
    settings.target = problem.optimum.y + 1e-8

    suc = 0
    rt = 0
    for _ in range(n_trials):
        es = c_maes.ModularCMAES(settings)
        es.run(problem)
        suc += problem.state.final_target_found
        rt  += problem.state.evaluations
        print(problem.state)
        problem.reset()

    if suc == 0:
        return float("inf")
    return rt / suc 


if __name__ == "__main__":
    config = {
       "active": False,
        "bound_correction": "SATURATE",
        "center_placement": "X0",
        "elitist": True,
        "matrix_adaptation": "SEPARABLE",
        "mirrored": "NONE",
        "orthogonal": False,
        "repelling_restart": True,
        "restart_strategy": "RESTART",
        "sample_sigma": True,
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "sequential_selection": True,
        "ssa": "CSA",
        "threshold_convergence": False,
        "weights": "EXPONENTIAL"
    }

    settings = c_maes.settings_from_dict(5, **config)   
    print(get_ert(settings, 1, 21))

