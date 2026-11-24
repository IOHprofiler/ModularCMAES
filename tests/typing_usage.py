"""Typical uses of the API, checked by a type checker in CI (pyright modcma tests/typing_usage.py).

Not a unittest module: it only has to type-check, so that the stubs and the annotations of the
package stay usable (numpy arrays where vectors are expected, keyword arguments of the helpers).
"""

import numpy as np

import modcma
from modcma import ModularCMAES, Modules, Settings, options


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def typical_use() -> float:
    d = 5
    modules = Modules()
    modules.matrix_adaptation = options.MatrixAdaptationType.MATRIX
    modules.weights = options.RecombinationWeights.UTILITY
    settings = Settings(
        d, modules, x0=np.zeros(d), lb=np.full(d, -5.0), ub=np.full(d, 5.0), sigma0=0.5,
        budget=1000, seed=1,
    )
    es = ModularCMAES(settings)
    es.run(sphere)
    X = es.ask()
    es.tell([sphere(x) for x in X])
    es.run(lambda X: np.sum(X * X, axis=1), vectorized=True)
    xopt, fopt, evaluations, es2 = modcma.fmin(sphere, np.ones(d), 0.5, 1000, active=True)
    other = modcma.settings_from_dict(d, matrix_adaptation="SEPARABLE", budget=1000, x0=np.ones(d))
    space = modcma.get_configspace(d)
    from_config = modcma.settings_from_config(d, space.get_default_configuration(), budget=10)
    return float(fopt + es.xfavorite[0] + other.budget + from_config.budget + evaluations)
