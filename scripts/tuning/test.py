import os

import pandas as pd
import ioh
from modcma import c_maes as modcma

ROOT = os.path.realpath(os.path.dirname(__file__))


def get_sampler(key: str) -> modcma.options.BaseSampler:
    if key == "halton":
        return modcma.options.BaseSampler.HALTON
    if key == "sobol":
        return modcma.options.BaseSampler.SOBOL
    if key == "gaussian":
        return modcma.options.BaseSampler.UNIFORM


def get_boundcorr(key: str) -> modcma.options.CorrectionMethod:
    if key == "saturate":
        return modcma.options.CorrectionMethod.SATURATE
    if pd.isna(key):
        return modcma.options.CorrectionMethod.NONE


def get_matrix(key: str) -> modcma.options.MatrixAdaptationType:
    if key == "matrix":
        return modcma.options.MatrixAdaptationType.MATRIX
    if key == "seperable":
        return modcma.options.MatrixAdaptationType.SEPARABLE
    if key == "covariance":
        return modcma.options.MatrixAdaptationType.COVARIANCE
    if key == "none":
        return modcma.options.MatrixAdaptationType.NONE


def get_restart(key: str) -> modcma.options.RestartStrategy:
    if key == "IPOP":
        return modcma.options.RestartStrategy.IPOP
    if key == "BIPOP":
        return modcma.options.RestartStrategy.BIPOP
    if key == "Restart":
        return modcma.options.RestartStrategy.RESTART
    if pd.isna(key):
        return modcma.options.RestartStrategy.NONE


def get_mirrored(key: str) -> modcma.options.Mirror:
    if key == "mirrored":
        return modcma.options.Mirror.MIRRORED
    if key == "mirrored pairwise":
        return modcma.options.Mirror.PAIRWISE
    if pd.isna(key):
        return modcma.options.Mirror.NONE


def get_sample_transform(key: str) -> modcma.options.SampleTranformerType:
    if key == "Gaussian":
        return modcma.options.SampleTranformerType.GAUSSIAN
    if key == "Cauchy":
        return modcma.options.SampleTranformerType.CAUCHY
    if key == "dWeibull":
        return modcma.options.SampleTranformerType.DOUBLE_WEIBULL
    if key == "Uniform":
        return modcma.options.SampleTranformerType.SCALED_UNIFORM


def get_ssa(key: str) -> modcma.options.StepSizeAdaptation:
    if key == "msr":
        return modcma.options.StepSizeAdaptation.MSR
    if key == "psr":
        return modcma.options.StepSizeAdaptation.PSR
    if key == "tpa":
        return modcma.options.StepSizeAdaptation.TPA
    if key == "csa":
        return modcma.options.StepSizeAdaptation.CSA


def get_weights(key: str) -> modcma.options.RecombinationWeights:
    if key == "default":
        return modcma.options.RecombinationWeights.DEFAULT
    if key == "equal":
        return modcma.options.RecombinationWeights.EQUAL
    if key == "1/2^lambda":
        return modcma.options.RecombinationWeights.HALF_POWER_LAMBDA


def make_modules(record: pd.Series) -> modcma.parameters.Modules:
    mods = modcma.parameters.Modules()
    if record is None:
        return mods
    mods.active = record["active"]
    mods.sampler = get_sampler(record["base_sampler"])
    mods.bound_correction = get_boundcorr(record["bound_correction"])
    mods.matrix_adaptation = get_matrix(record["covariance"])
    mods.elitist = record["elitist"]
    mods.restart_strategy = get_restart(record["local_restart"])
    mods.mirrored = get_mirrored(record["mirrored"])
    mods.orthogonal = record["orthogonal"]
    mods.repelling_restart = record["repelling_restart"]
    mods.sample_transformation = get_sample_transform(record["sample_transform"])
    mods.sequential_selection = record["sequential"]
    mods.ssa = get_ssa(record["step_size_adaptation"])
    mods.threshold_convergence = record["threshold"]
    mods.weights = get_weights(record["weights_option"])
    return mods


def make_settings(record: pd.Series, problem: ioh.ProblemType) -> modcma.Settings:
    modules = make_modules(record)
    if record is None:
        return modcma.Settings(
            dim=problem.meta_data.n_variables,
            modules=modules,
            target=problem.optimum.y + 1e-8,
            budget=problem.meta_data.n_variables * 10_000,
            lb=problem.bounds.lb,
            ub=problem.bounds.ub,
        )

    return modcma.Settings(
        dim=problem.meta_data.n_variables,
        modules=modules,
        target=problem.optimum.y + 1e-8,
        budget=problem.meta_data.n_variables * 10_000,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub,
        cmu=record["cmu"],
        cc=record["cc"],
        c1=record["c1"],
        cs=record["cs"],
        lambda0=record["lambda_"],
        mu0=record["mu"],
        sigma0=record["sigma0"],
        verbose=False,
    )


def run_for_fid(fid, record, n_runs=1, dim=5, n_instances=50):
    total_time = 0
    total_sucs = 0
    total_dfs = 0
    for instance in range(n_instances):
        problem = ioh.get_problem(fid, instance + 1, dim)
        setting = make_settings(record, problem)

        for _ in range(n_runs):
            cma = modcma.ModularCMAES(setting)
            cma.p.repelling.coverage = 2
            while cma.step(problem):
                pass
            print(
                problem.state.evaluations,
                problem.state.final_target_found,
                problem.state.current_best.y,
                problem.optimum.y,
            )
            total_time += problem.state.evaluations
            total_sucs += problem.state.final_target_found
            total_dfs += problem.state.current_best_internal.y
            problem.reset()
            
    # print(cma.p.repelling.archive)
    total_time += setting.budget * ((n_runs * n_instances) - total_sucs)
    print()
    print(record["ID"] if record is not None else "default")
    print(
        "\t",
        problem,
        f"ert: {total_time / total_sucs if total_sucs > 0 else float('inf'):.2f}",
        f"average delta f: {total_dfs / (n_runs * n_instances): .2e}",
        
    )
    print()


problem_f14 = {
    "active": True,
    "base_sampler": "halton",
    "bound_correction": float("nan"),
    "c1": 0.0100996668836,
    "cc": 0.1662296215659,
    "cmu": 0.016628429032,
    "covariance": "matrix",
    "cs": 0.3194387671633,
    "elitist": False,
    "lambda_": 12,
    "local_restart": float("nan"),
    "mirrored": float("nan"),
    "mu": 2,
    "orthogonal": False,
    "repelling_restart": True,
    "sample_transform": "Uniform",
    "sequential": False,
    "sigma0": 2.0308730750449,
    "step_size_adaptation": "psr",
    "threshold": False,
    "weights_option": "default",
    "cost": 0.01904575025093701,
    "ID": "Tuned_14",
}


f3_problem = {
    "active": True,
    "base_sampler": "halton",
    "bound_correction": float("nan"),
    "c1": None,
    "cc": None,
    "cmu": None,
    "cs": None,
    "covariance": "none",
    "elitist": False,
    "lambda_": 32,
    "mu": None,
    "local_restart": "Restart",
    "mirrored": float("nan"),
    "orthogonal": False,
    "repelling_restart": False,
    "sample_transform": "Uniform",
    "sequential": False,
    "sigma0": None,
    "step_size_adaptation": "tpa",
    "threshold": False,
    "weights_option": "default",
    "cost": 0.01904575025093701,
    "ID": "Expert 3",
}

if __name__ == "__main__":
    modcma.utils.set_seed(12)
    data = pd.read_csv(os.path.join(ROOT, "configs_5D.csv"))
    fid = 4
    dim = 5
    # record = data.iloc[fid].copy()
    record = pd.Series(f3_problem)
    run_for_fid(fid, record, dim=dim, n_instances=50)
    print(record)
