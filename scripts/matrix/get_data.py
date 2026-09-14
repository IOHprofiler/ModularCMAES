"""Collect BBOB data for the matrix-adaptation comparison."""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cma as pycma
import ioh
import numpy as np

import modcma.c_maes as ccma

DEFAULT_DIMS = (2, 3, 5, 10, 20, 40)
DEFAULT_FUNCTIONS = tuple(range(1, 25))
DEFAULT_REPEATS = 100
DEFAULT_BUDGET_PER_DIMENSION = 100_000
DEFAULT_TARGET = 1e-8
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Experiment:
    """Configuration shared by all matrix-adaptation benchmark jobs."""

    dims: tuple[int, ...] = DEFAULT_DIMS
    functions: tuple[int, ...] = DEFAULT_FUNCTIONS
    repeats: int = DEFAULT_REPEATS
    budget_per_dimension: int = DEFAULT_BUDGET_PER_DIMENSION
    target: float = DEFAULT_TARGET
    root: Path = DEFAULT_ROOT


def ert(runs: list[dict[str, float]], target: float) -> float:
    """Compute fixed-target expected running time."""
    total_evals = sum(row["evals"] for row in runs)
    n_successes = sum(row["best_y"] <= target for row in runs)
    return float("inf") if n_successes == 0 else total_evals / n_successes


def make_modules(module_name: str, option):
    modules = ccma.parameters.Modules()
    modules.restart_strategy = ccma.options.RestartStrategy.STOP
    setattr(modules, module_name, option)
    return modules


def run_modcma(
    problem: ioh.ProblemType,
    x0: np.ndarray,
    logger_obj,
    module_name: str,
    option,
    experiment: Experiment,
):
    """Run one ModCMA trial and return timing and evaluation statistics."""
    modules = make_modules(module_name, option)
    settings = ccma.Settings(
        problem.meta_data.n_variables,
        x0=x0,
        modules=modules,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub,
        verbose=False,
        sigma0=2.0,
        target=problem.optimum.y + experiment.target,
        budget=problem.meta_data.n_variables * experiment.budget_per_dimension,
    )

    cma = ccma.ModularCMAES(settings)
    start = perf_counter()
    while not cma.break_conditions():
        if cma.p.criteria.any():
            logger_obj.update(cma.p.criteria.items)
        cma.step(problem)

    if cma.p.criteria.any():
        logger_obj.update(cma.p.criteria.items)

    elapsed = perf_counter() - start
    return elapsed, cma.p.stats.t, problem.state.evaluations, cma.p.stats.n_updates


class RestartCollector:
    """Expose restart-criterion counts as IOH run attributes."""

    def __init__(self, strategy=ccma.options.RestartStrategy.STOP):
        modules = ccma.parameters.Modules()
        modules.restart_strategy = strategy
        settings = ccma.Settings(2, modules=modules)
        cma = ccma.ModularCMAES(settings)
        self.names = [criterion.name for criterion in cma.p.criteria.items]
        self.reset()

    def update(self, items):
        for item in items:
            if item.met:
                setattr(self, item.name, getattr(self, item.name) + 1)

    def reset(self):
        for name in self.names:
            setattr(self, name, 0)


def collect(
    name: str,
    module_name: str,
    option,
    experiment: Experiment,
) -> None:
    """Collect all ModCMA trials for one matrix-adaptation option."""
    logger = ioh.logger.Analyzer(
        folder_name=name,
        algorithm_name=name,
        root=str(experiment.root),
    )
    collector = RestartCollector()
    logger.add_run_attributes(collector, collector.names)

    for fid in experiment.functions:
        for dim in experiment.dims:
            problem = ioh.get_problem(fid, 1, dim)
            problem.attach_logger(logger)
            runs = []

            for repetition in range(experiment.repeats):
                ccma.utils.set_seed(21 + fid * dim * repetition)
                collector.reset()
                run_modcma(
                    problem,
                    np.zeros(dim),
                    collector,
                    module_name,
                    option,
                    experiment,
                )
                runs.append(
                    {
                        "evals": problem.state.evaluations,
                        "best_y": problem.state.current_best_internal.y,
                    }
                )
                problem.reset()

            print(name, fid, dim, "ERT:", ert(runs, experiment.target), flush=True)


def collect_modcma(experiment: Experiment, workers: int = 1) -> None:
    """Collect all supported ModCMA matrix-adaptation variants."""
    options = dict(ccma.options.MatrixAdaptationType.__members__)
    options.pop("COVARIANCE_NO_EIGV", None)
    jobs = [
        (name, "matrix_adaptation", option, experiment)
        for name, option in options.items()
    ]

    if workers == 1:
        for job in jobs:
            collect(*job)
        return

    with Pool(processes=workers) as pool:
        pool.starmap(collect, jobs)


def run_pycma(
    problem: ioh.ProblemType,
    x0: np.ndarray,
    experiment: Experiment,
):
    """Run one Pycma baseline trial."""
    options = pycma.CMAOptions()
    options["CMA_active"] = False
    options["verbose"] = -1
    options["CMA_diagonal"] = False
    options["conditioncov_alleviate"] = False
    options["ftarget"] = problem.optimum.y + experiment.target
    options["maxfevals"] = (
        problem.meta_data.n_variables * experiment.budget_per_dimension
    )

    cma = pycma.CMAEvolutionStrategy(x0, 2.0, options=options)
    settings = ccma.Settings(problem.meta_data.n_variables)
    assert settings.lambda0 == cma.sp.popsize

    start = perf_counter()
    target = problem.optimum.y + experiment.target
    budget = problem.meta_data.n_variables * experiment.budget_per_dimension

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while problem.state.evaluations < budget:
            points, values = cma.ask_and_eval(problem)
            cma.tell(points, values)

            if problem.state.current_best.y <= target or cma.stop():
                break

    elapsed = perf_counter() - start
    return elapsed, cma.countiter, problem.state.evaluations, cma.sm.count_eigen


def collect_pycma(experiment: Experiment) -> None:
    """Collect all Pycma baseline trials."""
    logger = ioh.logger.Analyzer(
        folder_name="pycma",
        algorithm_name="pycma",
        root=str(experiment.root),
    )

    for fid in experiment.functions:
        for dim in experiment.dims:
            problem = ioh.get_problem(fid, 1, dim)
            problem.attach_logger(logger)
            runs = []

            for repetition in range(experiment.repeats):
                np.random.seed(21 + fid * dim * repetition)
                run_pycma(problem, np.zeros(dim), experiment)
                runs.append(
                    {
                        "evals": problem.state.evaluations,
                        "best_y": problem.state.current_best_internal.y,
                    }
                )
                problem.reset()

            print("pycma", fid, dim, "ERT:", ert(runs, experiment.target), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect matrix-adaptation BBOB data for ModCMA and Pycma."
    )
    parser.add_argument(
        "--implementation",
        choices=("all", "modcma", "pycma"),
        default="all",
    )
    parser.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--functions", nargs="+", type=int, default=DEFAULT_FUNCTIONS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--budget-per-dimension",
        type=int,
        default=DEFAULT_BUDGET_PER_DIMENSION,
    )
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel ModCMA methods. Pycma is collected sequentially.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.budget_per_dimension < 1:
        raise ValueError("--budget-per-dimension must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    experiment = Experiment(
        dims=tuple(args.dims),
        functions=tuple(args.functions),
        repeats=args.repeats,
        budget_per_dimension=args.budget_per_dimension,
        target=args.target,
        root=args.root.resolve(),
    )
    experiment.root.mkdir(parents=True, exist_ok=True)

    if args.implementation in ("all", "modcma"):
        collect_modcma(experiment, args.workers)
    if args.implementation in ("all", "pycma"):
        collect_pycma(experiment)


if __name__ == "__main__":
    main()
