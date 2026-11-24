"""Evaluate each population in parallel processes with ask-tell."""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import modcma

QUICK = os.environ.get("MODCMA_EXAMPLE_QUICK") == "1"


def expensive_objective(x):
    """A stand-in for a simulation: the Rastrigin function plus a delay."""
    time.sleep(0.0 if QUICK else 0.01)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


if __name__ == "__main__":
    d = 10
    settings = modcma.Settings(d, lambda0=16, budget=1_000 if QUICK else 5_000, seed=1)
    es = modcma.ModularCMAES(settings)

    with ProcessPoolExecutor(max_workers=4) as pool:
        while True:
            X = es.ask()  # (k, d): the points to evaluate
            f = list(pool.map(expensive_objective, X))
            if es.tell(f) and es.break_conditions():
                break

    best = es.p.stats.global_best
    print(f"f = {best.y:.3g} after {es.p.stats.evaluations} evaluations")
