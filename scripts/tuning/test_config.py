from argparse import ArgumentParser

import ioh
import numpy as np

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


def get_bbob_performance(settings: c_maes.Settings, seed: int = 0, fid: int = 0):
    iid = 1 + (seed % 10)
    np.random.seed(seed + iid)
    c_maes.utils.set_seed(seed + iid)

    settings.budget = settings.dim * 10_000
    settings.target = problem.optimum.y + 9e-9

    l3 = ioh.logger.Store(
        triggers=[ioh.logger.trigger.ALWAYS], properties=[ioh.logger.property.RAWYBEST]
    )
    problem = ioh.get_problem(fid, iid, settings.dim)
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
    iid = 1 + (seed % 10)
    np.random.seed(seed + iid)
    c_maes.utils.set_seed(seed + iid)

    problem = ioh.get_problem(fid, iid, settings.dim)
    settings.budget = settings.dim * 10_000
    settings.target = problem.optimum.y + 1e-8
    suc = 0
    rt = 0
    for _ in range(n_trials):
        es = c_maes.ModularCMAES(settings)
        es.p.repelling.coverage = 10
        es.p.criteria.items = es.p.criteria.items[:3]
        # breakpoint()

        l_max = 200
        l_min = settings.lambda0
        ratio = settings.mu0 / settings.lambda0 
        s0 = settings.sigma0

        while not es.break_conditions():

            lambda_adaptive = min(l_max, max(l_min, int(l_max * pow((es.p.mutation.sigma / s0) - 1, 2))))
            mu_adaptive = int(lambda_adaptive * ratio)
            es.p.resize_population(mu_adaptive, lambda_adaptive)

            es.step(problem)


        suc += problem.state.final_target_found
        rt += problem.state.evaluations
        print(problem.state)
        problem.reset()

    if suc == 0:
        return float("inf"), 0
    return rt / suc, suc


configs = {
    1: {
        "lambda0": 2,
        "matrix_adaptation": "SEPARABLE",
        "mirrored": "PAIRWISE",
        "orthogonal": True,
        "restart_strategy": "BIPOP",
        "sample_transformation": "DOUBLE_WEIBULL",
        "sampler": "SOBOL",
        "threshold_convergence": False,
        "active": False,
        "elitist": True,
        "mu0": 2,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    2: {
        "lambda0": 4,
        "matrix_adaptation": "SEPARABLE",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "BIPOP",
        "sample_transformation": "CAUCHY",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "active": True,
        "elitist": True,
        "mu0": 3,
        "repelling_type": "NONE",
        "sequential_selection": True,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    3: {
        "lambda0": 80,
        "matrix_adaptation": "COVARIANCE",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "RESTART",
        "sample_transformation": "LAPLACE",
        "sampler": "HALTON",
        "threshold_convergence": False,
        "active": True,
        "elitist": False,
        "mu0": 40,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "TPA",
        "weights": "DEFAULT",
    },
    4: {
        "lambda0": 167,
        "matrix_adaptation": "SEPARABLE",
        "mirrored": "NONE",
        "orthogonal": True,
        "restart_strategy": "NONE",
        "sample_transformation": "CAUCHY",
        "sampler": "UNIFORM",
        "threshold_convergence": False,
        "active": True,
        "elitist": True,
        "mu0": 1,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "SA",
        "weights": "EXPONENTIAL",
    },
    5: {
        "lambda0": 1,
        "matrix_adaptation": "NONE",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "NONE",
        "sample_transformation": "CAUCHY",
        "sampler": "HALTON",
        "threshold_convergence": False,
    },
    6: {
        "lambda0": 1,
        "matrix_adaptation": "NATURAL_GRADIENT",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "RESTART",
        "sample_transformation": "DOUBLE_WEIBULL",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "repelling_type": "NONE",
    },
    7: {
        "lambda0": 10,
        "matrix_adaptation": "CMSA",
        "mirrored": "MIRRORED",
        "orthogonal": False,
        "restart_strategy": "BIPOP",
        "sample_transformation": "LOGISTIC",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "active": False,
        "elitist": False,
        "mu0": 6,
        "repelling_type": "COVERAGE",
        "sequential_selection": False,
        "ssa": "TPA",
        "weights": "DEFAULT",
    },
    8: {
        "lambda0": 5,
        "matrix_adaptation": "CHOLESKY",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "IPOP",
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "active": False,
        "elitist": True,
        "mu0": 3,
        "repelling_type": "NONE",
        "sequential_selection": True,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    9: {
        "lambda0": 1,
        "matrix_adaptation": "COVARIANCE",
        "mirrored": "MIRRORED",
        "orthogonal": True,
        "restart_strategy": "RESTART",
        "sample_transformation": "SCALED_UNIFORM",
        "sampler": "HALTON",
        "threshold_convergence": False,
        "repelling_type": "COVERAGE",
    },
    10: {
        "lambda0": 3,
        "matrix_adaptation": "COVARIANCE",
        "mirrored": "MIRRORED",
        "orthogonal": True,
        # "restart_strategy": "RESTART",
        "sample_transformation": "LAPLACE",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "active": False,
        "elitist": True,
        "mu0": 3,
        # "repelling_type": "COVERAGE",
        "sequential_selection": False,
        "ssa": "CSA",
        "weights": "EXPONENTIAL",
    },
    11: {
        "lambda0": 12,
        "matrix_adaptation": "NATURAL_GRADIENT",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "RESTART",
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "active": True,
        "elitist": False,
        "mu0": 4,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "TPA",
        "weights": "DEFAULT",
    },
    12: {
        "lambda0": 5,
        "matrix_adaptation": "CHOLESKY",
        "mirrored": "MIRRORED",
        "orthogonal": True,
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "elitist": True,
        "mu0": 5,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    13: {
        "lambda0": 5,
        "matrix_adaptation": "CMSA",
        "mirrored": "MIRRORED",
        "orthogonal": True,
        "restart_strategy": "BIPOP",
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "active": False,
        "elitist": True,
        "mu0": 5,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    14: {
        "lambda0": 3,
        "matrix_adaptation": "NATURAL_GRADIENT",
        "mirrored": "MIRRORED",
        "orthogonal": True,
        "restart_strategy": "IPOP",
        "sample_transformation": "GAUSSIAN",
        "sampler": "SOBOL",
        "threshold_convergence": True,
        "active": False,
        "elitist": True,
        "mu0": 3,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "CSA",
        "weights": "EXPONENTIAL",
    },
    15: {
        "lambda0": 61,
        "matrix_adaptation": "MATRIX",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "BIPOP",
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "threshold_convergence": False,
        "active": True,
        "elitist": False,
        "mu0": 37,
        "repelling_type": "NONE",
        "sequential_selection": True,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    16: {
        "lambda0": 14,
        "matrix_adaptation": "NATURAL_GRADIENT",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "BIPOP",
        "sample_transformation": "SCALED_UNIFORM",
        "sampler": "HALTON",
        "threshold_convergence": False,
        "active": False,
        "elitist": False,
        "mu0": 8,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "PSR",
        "weights": "EXPONENTIAL",
    },
    17: {
        "lambda0": 16,
        "matrix_adaptation": "CMSA",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "IPOP",
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "threshold_convergence": False,
        "active": False,
        "elitist": False,
        "mu0": 13,
        "repelling_type": "NONE",
        "sequential_selection": True,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    18: {
        "lambda0": 24,
        "matrix_adaptation": "COVARIANCE",
        "mirrored": "MIRRORED",
        "orthogonal": False,
        "restart_strategy": "RESTART",
        "sample_transformation": "SCALED_UNIFORM",
        "sampler": "HALTON",
        "threshold_convergence": False,
        "active": True,
        "elitist": False,
        "mu0": 12,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    19: {
        "lambda0": 45,
        "matrix_adaptation": "NATURAL_GRADIENT",
        "mirrored": "MIRRORED",
        "orthogonal": False,
        "restart_strategy": "RESTART",
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "threshold_convergence": False,
        "active": False,
        "elitist": False,
        # "mu0": 48,
        "repelling_type": "NONE",
        "sequential_selection": True,
        "ssa": "XNES",
        "weights": "DEFAULT",
    },
    20: {
        "lambda0": 39,
        "matrix_adaptation": "NATURAL_GRADIENT",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "RESTART",
        "sample_transformation": "GAUSSIAN",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "active": True,
        "elitist": False,
        "mu0": 11,
        "repelling_type": "NONE",
        "sequential_selection": True,
        "ssa": "XNES",
        "weights": "DEFAULT",
    },
    21: {
        "lambda0": 4,
        "matrix_adaptation": "CHOLESKY",
        "mirrored": "MIRRORED",
        "orthogonal": True,
        "restart_strategy": "RESTART",
        "sample_transformation": "GAUSSIAN",
        "sampler": "UNIFORM",
        "threshold_convergence": True,
        "active": False,
        "elitist": True,
        "mu0": 2,
        "repelling_type": "NONE",
        "sequential_selection": True,
        "ssa": "CSA",
        "weights": "DEFAULT",
    },
    22: {
        "lambda0": 2,
        "matrix_adaptation": "COVARIANCE",
        "mirrored": "PAIRWISE",
        "orthogonal": True,
        "restart_strategy": "RESTART",
        "sample_transformation": "DOUBLE_WEIBULL",
        "sampler": "SOBOL",
        "threshold_convergence": True,
        "active": True,
        "elitist": True,
        "mu0": 1,
        "repelling_type": "COVERAGE",
        "sequential_selection": False,
        "ssa": "MSR",
        "weights": "EXPONENTIAL",
    },
    23: {
        "lambda0": 15,
        "matrix_adaptation": "NATURAL_GRADIENT",
        "mirrored": "NONE",
        "orthogonal": False,
        "restart_strategy": "RESTART",
        "sample_transformation": "SCALED_UNIFORM",
        "sampler": "HALTON",
        "threshold_convergence": True,
        "active": True,
        "elitist": False,
        "mu0": 6,
        "repelling_type": "NONE",
        "sequential_selection": False,
        "ssa": "PSR",
        "weights": "DEFAULT",
    },
    24: {
        # "lambda0": 250,
        # "matrix_adaptation": "NATURAL_GRADIENT",
        # "mirrored": "MIRRORED",
        # "orthogonal": False,
        # "restart_strategy": "RESTART",
        # "sample_transformation": "LOGISTIC",
        # "sampler": "HALTON",
        # "threshold_convergence": False,
        # "active": False,
        # "elitist": False,
        # "mu0": 193,
        # "repelling_type": "NONE",
        # "sequential_selection": True,
        # "ssa": "MXNES",
        # "weights": "DEFAULT",

        "lambda0": 14,
        # "matrix_adaptation": "",
        # "mirrored": "MIRRORED",
        # "orthogonal": False,
        "restart_strategy": "RESTART",
        # "sample_transformation": "LOGISTIC",
        # "sampler": "HALTON",
        # "threshold_convergence": False,
        # "active": False,
        # "elitist": False,
        # "mu0": 193,
        # "repelling_type": "NONE",
        # "sequential_selection": True,
        # "ssa": "MXNES",
        # "weights": "DEFAULT",
    },
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fid", type=int, default=1)
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()
    settings = c_maes.settings_from_dict(2, **configs[args.fid])
    print(settings)
    print("ERT/SR:", *get_ert(settings, 0, args.fid, args.reps), "/", args.reps)
