# Quickstart

## Minimize a function with `fmin`

`fmin` runs one optimization and returns the best point, its value, the number of evaluations and
the optimizer:

```python
import numpy as np
import modcma

def sphere(x):
    return float(np.sum(x**2))

xopt, fopt, evaluations, es = modcma.fmin(sphere, x0=np.ones(5), sigma0=0.5, budget=5_000)
print(xopt, fopt, evaluations)
```

The objective takes a numpy array of shape `(d,)` and returns a float; modcma minimizes. `x0` is the
starting point, `sigma0` the initial step size: a rule of thumb is 0.3 × the width of the region
where you expect the optimum. `budget` is the maximum number of evaluations.

Every other setting and every module can be passed as a keyword argument, modules by the name of
their option:

```python
xopt, fopt, evaluations, es = modcma.fmin(
    sphere,
    x0=np.ones(5),
    sigma0=0.5,
    budget=5_000,
    target=1e-8,                   # stop once f <= 1e-8
    matrix_adaptation="MATRIX",    # MA-ES instead of the full covariance matrix
    active=True,                   # active (negative) covariance update
    seed=42,                       # reproducible
)
print(fopt, evaluations)
```

## The optimizer object

`fmin` is a shortcut for three objects:

* `Modules`: which modules are used (the [user guide](../guide/modules.md) explains each one);
* `Settings`: the problem and the parameters (dimension, budget, bounds, step size, population
  size, ...; see [settings](../guide/settings.md));
* `ModularCMAES`: the optimizer.

```python
modules = modcma.Modules()
modules.matrix_adaptation = modcma.options.MatrixAdaptationType.MATRIX
modules.restart_strategy = modcma.options.RestartStrategy.IPOP

settings = modcma.Settings(
    dim=5,
    modules=modules,
    x0=np.ones(5),
    sigma0=0.5,
    budget=20_000,
    target=1e-8,
    seed=1,
)
es = modcma.ModularCMAES(settings)
es.run(sphere)

best = es.p.stats.global_best
print(best.x, best.y, es.p.stats.evaluations, es.p.stats.t)
```

Everything about the state of the optimizer is in `es.p` (the `Parameters`): `es.p.stats` (the best
solution, evaluations, generations), `es.p.adaptation.m` (the mean), `es.p.mutation.sigma` (the step
size), `es.p.adaptation.C` (the covariance matrix, for the covariance-based adaptations), the
population `es.p.pop`, and the modules themselves.

## One generation at a time

`run` loops `step` until a break condition holds; you can do the same and look at the state in
between:

```python
es = modcma.ModularCMAES(modcma.Settings(5, x0=np.ones(5), sigma0=0.5, budget=2_000, seed=1))
while es.step(sphere):
    if es.p.stats.t % 50 == 0:
        print(es.p.stats.t, es.p.stats.global_best.y, es.p.mutation.sigma)
```

When the objective is evaluated elsewhere (in parallel, on a cluster) use
[ask–tell](ask-tell.md) instead.

## When does it stop?

`run` and `step` stop when one of these holds:

* the budget is used up (`budget`, default 10 000 × d evaluations);
* the target is reached (`target`, none by default);
* `max_generations` generations have run (none by default);
* with `restart_strategy=STOP`: a restart criterion holds (the search has converged or stagnated).

Without a restart strategy the optimizer uses the whole budget. Use `STOP` to stop at convergence,
or `IPOP`/`BIPOP` to restart with a larger population instead: see
[bounds, restarts and stopping](bounds-restarts-stopping.md).
