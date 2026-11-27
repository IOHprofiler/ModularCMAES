"""Warm starting: the solutions of a related task give the initial distribution of a new run.

WS-CMA-ES (Nomura et al. 2021) estimates the initial mean, step size and covariance matrix from the
best solutions of a source task (modcma.warm_start). The tasks here are the paper's synthetic
pair, in d = 10: a rotated ellipsoid (condition 1e4) in [0, 1]^d with its optimum at b = 0.6 in
every coordinate (the source task), and the same ellipsoid with its optimum shifted (the target
task); a third target also has another rotation, so its shape is unrelated. The cold start is the
paper's N(0.5, 0.2^2) in every coordinate. Each start runs on the target task until f <= 1e-8;
the table gives the median number of evaluations.

The source solutions are the 1000 points of a random search on the source task, or the 1000
evaluations of a CMA-ES run on it.
"""

import os

import numpy as np

import modcma

QUICK = os.environ.get("MODCMA_EXAMPLE_QUICK") == "1"
SEEDS = range(3 if QUICK else 20)
D = 10
ROTATION, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(D, D)))
WEIGHTS = 1e4 ** (np.arange(D) / (D - 1))


def ellipsoid(b, rotation=ROTATION):
    """The rotated ellipsoid with its optimum (f = 0) at b in every coordinate."""
    return lambda x: float(np.sum(WEIGHTS * (rotation @ (x - b)) ** 2))


def random_search(f, n, seed):
    X = np.random.default_rng(seed).uniform(0, 1, size=(n, D))
    return [(x, f(x)) for x in X]


def cmaes_run(f, n, seed):
    history = []

    def recorded(x):
        history.append((x, f(x)))
        return history[-1][1]

    settings = modcma.Settings(D, x0=np.full(D, 0.5), sigma0=0.2, budget=n, seed=seed)
    modcma.ModularCMAES(settings).run(recorded)
    return history


def evaluations(f, seed, x0, sigma0, C0=None):
    """The evaluations to reach f <= 1e-8 on the target task."""
    settings = modcma.Settings(
        D, x0=x0, sigma0=sigma0, C0=C0, budget=50_000, target=1e-8, seed=seed
    )
    es = modcma.ModularCMAES(settings)
    es.run(f)
    assert es.p.stats.global_best.y <= 1e-8
    return es.p.stats.evaluations


def main():
    source = ellipsoid(0.6)
    other_rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(D, D)))
    targets = {
        "shifted by 0.02": ellipsoid(0.62),
        "shifted by 0.1": ellipsoid(0.7),
        "0.02, rotated": ellipsoid(0.62, other_rotation),
    }
    print(f"Rotated ellipsoid, d = {D}: median evaluations to f <= 1e-8, {len(SEEDS)} seeds")
    print(f"{'source solutions':17s}{'target':16s}{'cold':>6s}{'warm':>6s}{'warm, no C0':>13s}")
    for name, solve in [("random search", random_search), ("CMA-ES run", cmaes_run)]:
        for target_name, target in targets.items():
            cold, warm, without_c0 = [], [], []
            for seed in SEEDS:
                # The source solutions have seeds of their own
                ws = modcma.warm_start(solve(source, 1000, seed=1000 + seed))
                cold.append(evaluations(target, seed, np.full(D, 0.5), 0.2))
                warm.append(evaluations(target, seed, **ws))
                without_c0.append(evaluations(target, seed, ws["x0"], ws["sigma0"]))
            print(
                f"{name:17s}{target_name:16s}{np.median(cold):6.0f}{np.median(warm):6.0f}"
                f"{np.median(without_c0):13.0f}"
            )


if __name__ == "__main__":
    main()
