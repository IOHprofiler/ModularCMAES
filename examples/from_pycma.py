"""The modcma equivalents of common pycma calls.

pycma (`import cma`) is the reference implementation of CMA-ES in Python. modcma implements the
same algorithm in C++, with other variants as modules. This script runs the same optimization with
both when pycma is installed (`pip install cma`), and with modcma alone otherwise. (Rosenbrock has
a local optimum in 10-D, where an occasional run of either converges; restarts avoid that.)
"""

import numpy as np

import modcma


def rosenbrock(x):
    x = np.asarray(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


d, x0, sigma0 = 10, np.zeros(10), 0.5

# pycma:  xbest, es = cma.fmin2(rosenbrock, x0, sigma0, {"maxfevals": 20_000, "ftarget": 1e-10})
# pycma uses the active covariance update by default; in modcma it is the module `active`
xopt, fopt, evaluations, es = modcma.fmin(
    rosenbrock, x0, sigma0, budget=20_000, target=1e-10, active=True, seed=3
)
print(f"modcma fmin:     f = {fopt:.2e} after {evaluations} evaluations")

# pycma:  es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": 20, "seed": 1, ...})
#         while not es.stop():
#             X = es.ask()
#             es.tell(X, [rosenbrock(x) for x in X])
settings = modcma.settings_from_dict(
    d, x0=x0, sigma0=sigma0, lambda0=20, budget=20_000, target=1e-10, active=True, seed=3
)
es = modcma.ModularCMAES(settings)
while True:
    X = es.ask()  # (k, d) array; pycma: a list of k arrays
    if es.tell([rosenbrock(x) for x in X]) and es.break_conditions():
        break
print(f"modcma ask-tell: f = {es.p.stats.global_best.y:.2e} after {es.p.stats.evaluations} evaluations")

# Options and their modcma equivalents (Settings arguments, or Modules fields):
#   popsize            -> lambda0                 CMA_mu            -> mu0
#   maxfevals          -> budget                  ftarget           -> target
#   maxiter            -> max_generations (modcma's counts all runs, pycma's one run)
#   seed               -> seed
#   bounds=[lb, ub]    -> lb=..., ub=... (arrays) with a bound_correction module
#   CMA_active         -> active=True (off by default in modcma, on in pycma)
#   CMA_diagonal       -> matrix_adaptation="SEPARABLE"
#   restarts + IPOP    -> restart_strategy="IPOP" / "BIPOP" (cma.fmin2(..., restarts=9, bipop=True))
#   integer_variables  -> integer_variables
#   es.result.xbest    -> es.p.stats.global_best.x   es.result.fbest -> es.p.stats.global_best.y
#   es.sigma           -> es.p.mutation.sigma        es.mean         -> es.p.adaptation.m

try:
    import cma
except ImportError:
    cma = None

if cma is not None:
    xbest, es_cma = cma.fmin2(
        rosenbrock, x0, sigma0, {"maxfevals": 20_000, "ftarget": 1e-10, "seed": 2, "verbose": -9}
    )
    print(f"pycma fmin2:     f = {es_cma.result.fbest:.2e} after {es_cma.result.evaluations} evaluations")
