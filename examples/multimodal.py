"""Multimodal problems: restarts to find the global optimum, repelling restarts to find many.

1. The Rastrigin function has a local optimum at every integer point and the global one at 0. One
   run of CMA-ES converges to a local optimum; restarts with a growing population (IPOP, BIPOP)
   exploit the global structure and find the global optimum.
2. The Himmelblau function has four global optima. Plain restarts often converge to the same one
   again; repelling restarts keep the new runs away from the optima found before, so more of the
   four are found with the same budget.
"""

import os

import numpy as np

import modcma

QUICK = os.environ.get("MODCMA_EXAMPLE_QUICK") == "1"
SEEDS = range(3 if QUICK else 10)


def rastrigin(X):
    """Vectorized: X is (k, d), returns k values; the optimum is 0 at x = 0."""
    return 10 * X.shape[1] + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)


def himmelblau(X):
    return (X[:, 0] ** 2 + X[:, 1] - 11) ** 2 + (X[:, 0] + X[:, 1] ** 2 - 7) ** 2


HIMMELBLAU_OPTIMA = np.array(
    [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]
)


def global_optimum():
    d = 10
    variants = {
        "CMA-ES": {},
        "IPOP-CMA-ES": {"restart_strategy": "IPOP"},
        "BIPOP-CMA-ES": {"restart_strategy": "BIPOP"},
    }
    print(f"Rastrigin, d = {d}, {len(SEEDS)} runs each (solved: f <= 1e-8)")
    for name, modules in variants.items():
        solved = []
        for seed in SEEDS:
            settings = modcma.settings_from_dict(
                d,
                x0=np.full(d, 3.0),
                sigma0=2.0,
                budget=20_000 if QUICK else 200_000,
                target=1e-8,
                seed=seed,
                **modules,
            )
            es = modcma.ModularCMAES(settings)
            es.run(rastrigin, vectorized=True)
            if es.p.stats.global_best.y <= 1e-8:
                solved.append(es.p.stats.evaluations)
        line = f"  {name:14s} solved {len(solved):2d}/{len(SEEDS)}"
        if solved:
            line += f", median {np.median(solved):8.0f} evaluations"
        print(line)


def distinct_optima():
    seeds = range(20)
    print(f"\nHimmelblau, d = 2, restarts from uniform random points, {len(seeds)} runs each")
    for name, repelling in {"restarts": "NONE", "repelling restarts": "COVERAGE"}.items():
        found_per_run = []
        for seed in seeds:
            settings = modcma.settings_from_dict(
                2,
                sigma0=1.0,
                budget=4_000,
                restart_strategy="RESTART",
                center_placement="UNIFORM",
                repelling_type=repelling,
                seed=seed,
            )
            es = modcma.ModularCMAES(settings)
            es.run(himmelblau, vectorized=True)
            # The final mean of every run (es.p.stats.centers): which optimum did it converge to?
            found = set()
            for center in es.p.stats.centers:
                distance = np.linalg.norm(HIMMELBLAU_OPTIMA - np.asarray(center.x), axis=1)
                if distance.min() < 1e-2:
                    found.add(int(distance.argmin()))
            found_per_run.append(len(found))
        print(
            f"  {name:20s} distinct optima found: mean {np.mean(found_per_run):.2f} of 4, "
            f"all four in {sum(n == 4 for n in found_per_run)}/{len(seeds)} runs"
        )


if __name__ == "__main__":
    global_optimum()
    distinct_optima()
