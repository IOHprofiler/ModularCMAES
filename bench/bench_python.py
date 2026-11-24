"""Cost of the Python boundary for the C++ ModularCMAES (TASKS.md, T0.2).

Measures, with the installed ``modcma`` extension:
  * microseconds per evaluation of ``ModularCMAES.run`` with a Python objective (a lambda) and,
    when ``ioh`` is installed, with an ioh problem (C++ objective behind a Python object);
  * the cost of a bare Python call of the same lambda, for reference;
  * attribute access that user code does often (``cma.p.stats.global_best.y``, ``cma.p.pop.X``).

Compare the per-evaluation numbers with ``modcma_bench`` (C++ objective) for the same dimension:
the difference is the boundary. Every number is the minimum over ``--repetitions`` runs.

Usage: python bench/bench_python.py [--dims 2,10,100] [--generations 2000] [--repetitions 5]
"""

import argparse
import time

import numpy as np

from modcma import c_maes

try:
    import ioh
except ImportError:  # pragma: no cover
    ioh = None


def us_per_evaluation(objective, dim: int, generations: int) -> float:
    c_maes.set_seed(1)
    settings = c_maes.parameters.Settings(
        dim, max_generations=generations, budget=10**12, sigma0=2.0, verbose=False
    )
    cma = c_maes.ModularCMAES(settings)
    start = time.perf_counter()
    cma.run(objective)
    elapsed = time.perf_counter() - start
    return elapsed / cma.p.stats.evaluations * 1e6


def us_per_call(function, n: int = 100_000) -> float:
    start = time.perf_counter()
    for _ in range(n):
        function()
    return (time.perf_counter() - start) / n * 1e6


def best_of(repetitions: int, measure) -> float:
    return min(measure() for _ in range(repetitions))


def main():
    parser = argparse.ArgumentParser(description="Cost of the Python boundary for the C++ ModularCMAES")
    parser.add_argument("--dims", default="2,10,100")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    print(f"{'d':>4} {'lambda obj':>12} {'ioh obj':>12} {'bare call':>12}   (us per evaluation)")
    for dim in map(int, args.dims.split(",")):
        sphere = lambda x: float(x.dot(x))  # noqa: E731
        lam = best_of(args.repetitions, lambda: us_per_evaluation(sphere, dim, args.generations))
        if ioh is not None:
            problem = ioh.get_problem(1, 1, dim)
            ioh_cost = best_of(
                args.repetitions, lambda: us_per_evaluation(problem, dim, args.generations)
            )
            ioh_text = f"{ioh_cost:12.3f}"
        else:
            ioh_text = f"{'n/a':>12}"
        x = np.random.default_rng(1).random(dim)
        bare = best_of(args.repetitions, lambda: us_per_call(lambda: sphere(x)))
        print(f"{dim:4d} {lam:12.3f} {ioh_text} {bare:12.3f}")

    cma = c_maes.ModularCMAES(c_maes.parameters.Settings(10, verbose=False))
    cma.step(lambda x: float(x.dot(x)))
    print()
    print("attribute access (us per access):")
    for name, getter in (
        ("cma.p.stats.global_best.y", lambda: cma.p.stats.global_best.y),
        ("cma.p.mutation.sigma", lambda: cma.p.mutation.sigma),
        ("cma.p.pop.X (d=10, lambda=10)", lambda: cma.p.pop.X),
        ("cma.p.adaptation.C (10x10)", lambda: getattr(cma.p.adaptation, "C")),
    ):
        print(f"  {name:32s} {best_of(args.repetitions, lambda: us_per_call(getter)):8.3f}")


if __name__ == "__main__":
    main()
