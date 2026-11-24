"""Tune a configuration with a random search over modcma's configuration space.

Every module and the main parameters are hyperparameters of a ConfigSpace
(`modcma.get_configspace`). A random search samples configurations, runs each on a few training
functions and keeps the best; then the best configuration is compared with the default (the
standard CMA-ES) on the same functions. With SMAC or irace instead of the random search, this is the
setup of de Nobel et al. (2021).
"""

import os

import numpy as np

import modcma

QUICK = os.environ.get("MODCMA_EXAMPLE_QUICK") == "1"
DIM = 5
BUDGET = 2_000 * DIM
N_CONFIGURATIONS = 4 if QUICK else 60
SEEDS = range(2 if QUICK else 5)


def sphere(X):
    return np.sum(X**2, axis=1)


def ellipsoid(X):
    return (X * X) @ (10.0 ** (6 * np.arange(X.shape[1]) / (X.shape[1] - 1)))


def rastrigin(X):
    return 10 * X.shape[1] + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)


FUNCTIONS = [sphere, ellipsoid, rastrigin]  # all with the optimum 0 at x = 0


def loss(config):
    """Mean log10 of the best value reached (lower is better), over functions and seeds."""
    values = []
    for function in FUNCTIONS:
        for seed in SEEDS:
            settings = modcma.settings_from_config(
                DIM, config, budget=BUDGET, target=1e-8, seed=seed, x0=np.full(DIM, 2.0)
            )
            es = modcma.ModularCMAES(settings)
            with np.errstate(over="ignore", invalid="ignore"):  # some configurations diverge
                es.run(function, vectorized=True)
            values.append(np.log10(max(es.p.stats.global_best.y, 1e-8)))
    return float(np.mean(values))


if __name__ == "__main__":
    # Only the modules and the population size; sigma0 and the learning rates keep their defaults
    space = modcma.get_configspace(DIM, add_sigma=False, add_learning_rates=False)
    space.seed(1)

    default = space.get_default_configuration()
    best, best_loss = default, loss(default)
    print(f"default configuration: loss {best_loss:.2f}")

    for i, config in enumerate(space.sample_configuration(N_CONFIGURATIONS)):
        value = loss(config)
        if value < best_loss:
            best, best_loss = config, value
            print(f"configuration {i:3d}: loss {value:.2f} (new best)")

    print(f"\nbest loss {best_loss:.2f}; its settings:")
    print(modcma.settings_from_config(DIM, best).modules)
