"""Runtime of CMA-ES implementations on the ellipsoid, as a function of the dimension.

Every implementation runs its default full-covariance CMA-ES (default population size
4 + floor(3 ln d), initial step size 2, mean 0, no restarts, no active update where it is an option)
for a fixed number of generations on the same separable ellipsoid
f(x) = sum_i 10^(6 (i-1)/(d-1)) x_i^2, evaluated on the whole population at once in the
implementation's own array type. The time per generation therefore measures the algorithm: sampling,
the evaluation call, selection and the updates of the mean, step size and covariance matrix.

Each implementation is driven the way its documentation recommends:
  modcma     ModularCMAES.run(f, vectorized=True)       (C++, one call per generation)
  pycma      CMAEvolutionStrategy.ask() / tell()        (numpy)
  cmaes      CMA.ask() per solution / tell()            (numpy, CyberAgent; used by Optuna)
  evosax     ask / tell compiled with jax.lax.scan      (JAX; compilation excluded)
  evotorch   CMAES.step() on a vectorized Problem       (PyTorch)

Everything runs on one thread: BLAS/OpenMP, XLA and torch are limited to one thread, so the numbers
compare implementations, not thread pools. Run it on an otherwise idle machine.

    python scripts/performance/benchmark.py probe            # a quick estimate of the full run
    python scripts/performance/benchmark.py run --output scripts/performance/results/timing.csv
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import math
import platform
import subprocess
import sys
import time
import warnings
from typing import Callable

import numpy as np

DIMS = (2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)
SIGMA0 = 2.0


def popsize(d: int) -> int:
    return 4 + int(3 * math.log(d))


def ellipsoid_weights(d: int) -> np.ndarray:
    return 10.0 ** (6.0 * np.arange(d) / max(d - 1, 1))


# Each factory returns run(generations) -> None for a fresh optimizer of dimension d.


def modcma_factory(matrix_adaptation: str, fast_sampling: bool = False, vectorized: bool = True):
    def factory(d: int, seed: int) -> Callable[[int], None]:
        from modcma import c_maes

        w = ellipsoid_weights(d)

        def objective(X):
            return (X * X) @ w

        def point_objective(x):
            return float((x * x) @ w)

        def run(generations: int) -> None:
            modules = c_maes.Modules()
            modules.matrix_adaptation = getattr(
                c_maes.options.MatrixAdaptationType, matrix_adaptation
            )
            settings = c_maes.Settings(
                d,
                modules,
                x0=np.zeros(d),
                sigma0=SIGMA0,
                lambda0=popsize(d),
                mu0=popsize(d) // 2,
                max_generations=generations,
                budget=10**15,
                seed=seed,
                fast_sampling=fast_sampling,
            )
            cma = c_maes.ModularCMAES(settings)
            if vectorized:
                cma.run(objective, vectorized=True)
            else:
                cma.run(point_objective)

        return run

    return factory


def pycma_factory(d: int, seed: int) -> Callable[[int], None]:
    import cma

    w = ellipsoid_weights(d)

    def run(generations: int) -> None:
        options = {
            "CMA_active": False,
            "verbose": -9,
            "seed": seed + 1,
            "popsize": popsize(d),
            "conditioncov_alleviate": False,
        }
        es = cma.CMAEvolutionStrategy(np.zeros(d), SIGMA0, options)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(generations):
                X = es.ask()
                A = np.asarray(X)
                es.tell(X, ((A * A) @ w).tolist())

    return run


def cmaes_factory(d: int, seed: int) -> Callable[[int], None]:
    import cmaes

    w = ellipsoid_weights(d)
    lam = popsize(d)

    def run(generations: int) -> None:
        es = cmaes.CMA(mean=np.zeros(d), sigma=SIGMA0, seed=seed, population_size=lam)
        for _ in range(generations):
            X = np.array([es.ask() for _ in range(lam)])
            f = (X * X) @ w
            es.tell(list(zip(X, f)))

    return run


def evosax_factory(d: int, seed: int) -> Callable[[int], None]:
    import jax
    import jax.numpy as jnp
    from evosax.algorithms import CMA_ES

    w = jnp.asarray(ellipsoid_weights(d))
    es = CMA_ES(population_size=popsize(d), solution=jnp.zeros(d))
    params = es.default_params.replace(std_init=SIGMA0)
    compiled = {}

    def make(generations: int):
        def loop(key):
            key, init_key = jax.random.split(key)
            state = es.init(init_key, jnp.zeros(d), params)

            def body(carry, _):
                key, state = carry
                key, ask_key, tell_key = jax.random.split(key, 3)
                X, state = es.ask(ask_key, state, params)
                f = (X * X) @ w
                state, _ = es.tell(tell_key, X, f, state, params)
                return (key, state), None

            (_, state), _ = jax.lax.scan(body, (key, state), None, length=generations)
            return state.mean

        return jax.jit(loop)

    def run(generations: int) -> None:
        if generations not in compiled:
            compiled[generations] = make(generations)
            compiled[generations](jax.random.key(seed)).block_until_ready()  # compile
        compiled[generations](jax.random.key(seed)).block_until_ready()

    return run


def evotorch_factory(d: int, seed: int) -> Callable[[int], None]:
    import torch
    from evotorch import Problem
    from evotorch.algorithms import CMAES

    import logging

    logging.getLogger("evotorch").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    w = torch.as_tensor(ellipsoid_weights(d))

    def f(X):
        return (X * X) @ w

    def run(generations: int) -> None:
        torch.manual_seed(seed)
        problem = Problem(
            "min",
            f,
            solution_length=d,
            initial_bounds=(-1.0, 1.0),
            vectorized=True,
            dtype=torch.float64,
        )
        searcher = CMAES(
            problem,
            stdev_init=SIGMA0,
            popsize=popsize(d),
            center_init=torch.zeros(d, dtype=torch.float64),
            active=False,
        )
        for _ in range(generations):
            searcher.step()

    return run


IMPLEMENTATIONS = {
    "modcma": modcma_factory("COVARIANCE"),
    "modcma (fast sampling)": modcma_factory("COVARIANCE", fast_sampling=True),
    "modcma (per-point callback)": modcma_factory("COVARIANCE", vectorized=False),
    "modcma MATRIX": modcma_factory("MATRIX"),
    "modcma CHOLESKY": modcma_factory("CHOLESKY"),
    "modcma SEPARABLE": modcma_factory("SEPARABLE"),
    "modcma NONE": modcma_factory("NONE"),
    "modcma CMSA": modcma_factory("CMSA"),
    "modcma NATURAL_GRADIENT": modcma_factory("NATURAL_GRADIENT"),
    "modcma COVARIANCE_NO_EIGV": modcma_factory("COVARIANCE_NO_EIGV"),
    "pycma": pycma_factory,
    "cmaes": cmaes_factory,
    "evosax": evosax_factory,
    "evotorch": evotorch_factory,
}


def generations_for(d: int, base: int) -> int:
    """Fewer generations in high dimension (every implementation runs the same number)"""
    return base if d <= 100 else max(base // 5, 50)


def repetitions_for(d: int, base: int) -> int:
    return base if d <= 100 else max(base // 3, 3)


def time_run(run: Callable[[int], None], generations: int) -> float:
    start = time.perf_counter()
    run(generations)
    return time.perf_counter() - start


def environment() -> dict:
    import importlib.metadata as md

    versions = {}
    for package in ("modcma", "cma", "cmaes", "evosax", "jax", "evotorch", "torch", "numpy"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "missing"
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
            **versions}


def probe(implementations, dims, generations: int) -> None:
    """Time a few generations per cell and extrapolate the full run"""
    total = 0.0
    for name in implementations:
        line = []
        for d in dims:
            run = IMPLEMENTATIONS[name](d, 0)
            n = 3 if d >= 500 else 20
            run(n)  # warm-up (and compilation)
            per_generation = time_run(run, n) / n
            cell = per_generation * generations_for(d, generations) * repetitions_for(d, 15)
            total += cell
            line.append(f"{d}:{per_generation * 1e6:.0f}")
        print(f"{name:28s} us/generation {' '.join(line)}", flush=True)
    print(f"estimated full run: {total / 60:.0f} minutes")


def run_all(implementations, dims, generations: int, repetitions: int, output: str,
            max_cell_seconds: float) -> None:
    import pandas as pd

    rows = []
    env = environment()
    print(env, flush=True)
    for d in dims:
        for name in implementations:
            g, r = generations_for(d, generations), repetitions_for(d, repetitions)
            run = IMPLEMENTATIONS[name](d, 0)
            run(min(g, 3))  # warm-up (and compilation)
            first = time_run(run, g)
            if first * r > max_cell_seconds:
                r = max(1, int(max_cell_seconds // max(first, 1e-9)))
            times = [first] + [time_run(IMPLEMENTATIONS[name](d, k)
                                        if not name.startswith("evosax") else run, g)
                               for k in range(1, r)]
            for k, t in enumerate(times):
                rows.append({"implementation": name, "dim": d, "lambda": popsize(d),
                             "generations": g, "repetition": k, "seconds": t,
                             "us_per_generation": t / g * 1e6})
            median = float(np.median([t / g * 1e6 for t in times]))
            print(f"d={d:5d} {name:28s} {median:12.1f} us/generation ({r} x {g})", flush=True)
            pd.DataFrame(rows).assign(**{f"env_{k}": v for k, v in env.items()}).to_csv(
                output, index=False
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("probe", "run"))
    parser.add_argument("--implementations", default=",".join(IMPLEMENTATIONS))
    parser.add_argument("--dims", default=",".join(map(str, DIMS)))
    parser.add_argument("--generations", type=int, default=1000,
                        help="generations per run (d > 100: a fifth)")
    parser.add_argument("--repetitions", type=int, default=15,
                        help="runs per cell (d > 100: a third)")
    parser.add_argument("--max-cell-seconds", type=float, default=600.0,
                        help="fewer repetitions for a cell that would take longer")
    parser.add_argument("--output", default="scripts/performance/results/timing.csv")
    args = parser.parse_args()
    implementations = args.implementations.split(",")
    dims = [int(d) for d in args.dims.split(",")]
    if args.command == "probe":
        probe(implementations, dims, args.generations)
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        run_all(implementations, dims, args.generations, args.repetitions, args.output,
                args.max_cell_seconds)


if __name__ == "__main__":
    main()
