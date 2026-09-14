"""Measure runtime of the matrix-adaptation implementations."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cma as pycma
import ioh
import numpy as np
import pandas as pd

import modcma.c_maes as ccma

DEFAULT_DIMS = (2, 3, 5, 10, 20, 40, 100, 200, 500, 1000)
DEFAULT_REPEATS = 15
DEFAULT_GENERATIONS = 1_000
DEFAULT_OUTPUT = Path(__file__).resolve().parent


def run_modcma(
    problem: ioh.ProblemType,
    x0: np.ndarray,
    matrix_adaptation=ccma.options.MatrixAdaptationType.COVARIANCE,
    max_generations: int = DEFAULT_GENERATIONS,
):
    """Time one ModCMA run."""
    modules = ccma.parameters.Modules()
    modules.matrix_adaptation = matrix_adaptation
    settings = ccma.Settings(
        problem.meta_data.n_variables,
        x0=x0,
        modules=modules,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub,
        verbose=False,
        max_generations=max_generations,
    )

    cma = ccma.ModularCMAES(settings)
    start = perf_counter()
    cma.run(problem)
    elapsed = perf_counter() - start
    assert cma.p.stats.t == max_generations
    return elapsed, cma.p.stats.t, problem.state.evaluations, cma.p.stats.n_updates


def run_pycma(
    problem: ioh.ProblemType,
    x0: np.ndarray,
    max_generations: int = DEFAULT_GENERATIONS,
):
    """Time one Pycma run with settings aligned to ModCMA."""
    options = pycma.CMAOptions()
    options["CMA_active"] = False
    options["conditioncov_alleviate"] = False
    options["verbose"] = -1
    options["CMA_diagonal"] = False

    cma = pycma.CMAEvolutionStrategy(x0, 2.0, options=options)
    settings = ccma.Settings(problem.meta_data.n_variables)
    assert settings.lambda0 == cma.sp.popsize

    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(max_generations):
            points, values = cma.ask_and_eval(problem)
            cma.tell(points, values)
    elapsed = perf_counter() - start

    return elapsed, cma.countiter, problem.state.evaluations, cma.sm.count_eigen


def collect_modcma(
    dims: tuple[int, ...],
    repeats: int,
    generations: int,
) -> pd.DataFrame:
    """Collect timing data for every supported ModCMA adaptation."""
    options = dict(ccma.options.MatrixAdaptationType.__members__)
    options.pop("COVARIANCE_NO_EIGV", None)

    stats = []
    for dim in dims:
        for name, option in options.items():
            for repetition in range(repeats):
                ccma.utils.set_seed(21 + dim * repetition)
                problem = ioh.get_problem(2, 1, dim)
                result = run_modcma(
                    problem,
                    np.zeros(dim),
                    option,
                    generations,
                )
                stats.append((name, dim, *result))
                print(stats[-1], flush=True)

    return pd.DataFrame(
        stats,
        columns=["method", "dim", "time", "n_gen", "n_evals", "n_updates"],
    )


def collect_pycma(
    dims: tuple[int, ...],
    repeats: int,
    generations: int,
) -> pd.DataFrame:
    """Collect timing data for the Pycma baseline."""
    stats = []
    for dim in dims:
        for repetition in range(repeats):
            np.random.seed(21 + dim * repetition)
            problem = ioh.get_problem(2, 1, dim)
            result = run_pycma(problem, np.zeros(dim), generations)
            stats.append(("pycma", dim, *result))
            print(stats[-1], flush=True)

    return pd.DataFrame(
        stats,
        columns=["method", "dim", "time", "n_gen", "n_evals", "n_updates"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time matrix-adaptation methods for a fixed generation count."
    )
    parser.add_argument(
        "--implementation",
        choices=("all", "modcma", "pycma"),
        default="all",
    )
    parser.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.generations < 1:
        raise ValueError("--generations must be positive")

    dims = tuple(args.dims)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.implementation in ("all", "modcma"):
        stats = collect_modcma(dims, args.repeats, args.generations)
        stats.to_csv(output_dir / "time_stats.csv", index=False)

    if args.implementation in ("all", "pycma"):
        stats = collect_pycma(dims, args.repeats, args.generations)
        stats.to_csv(output_dir / "time_stats_pycma.csv", index=False)


if __name__ == "__main__":
    main()
