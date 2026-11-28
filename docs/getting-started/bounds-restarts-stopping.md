# Bounds, restarts and stopping

## Bounds

`Settings(lb=..., ub=...)` sets a box (scalars are not accepted: pass arrays of length d). The
default box is [-5, 5]^d, the domain of the BBOB benchmark. Without `x0`, the search starts at the
center of the box.

What happens to a sampled point outside the box is the `bound_correction` module. The default,
`NONE`, evaluates the point as it is (the objective must accept it); every other method moves it
inside before it is evaluated:

```python
import numpy as np
import modcma

def sphere(x):
    return float(np.sum(x**2))

d = 5
lb, ub = np.full(d, 1.0), np.full(d, 3.0)  # the optimum of the sphere is outside: x = lb is best
xopt, fopt, evaluations, es = modcma.fmin(
    sphere, x0=np.full(d, 2.0), sigma0=0.5, budget=3_000,
    lb=lb, ub=ub, bound_correction="SATURATE", seed=1,
)
print(np.round(xopt, 4))  # all 1.0
```

| `bound_correction` | the point outside is ... |
|---|---|
| `NONE` | evaluated as it is (default) |
| `SATURATE` | moved to the nearest point on the boundary |
| `MIRROR` | reflected at the boundary |
| `TOROIDAL` | wrapped around to the other side |
| `COTN` | moved inside by a random distance from the boundary it crossed |
| `UNIFORM_RESAMPLE` | replaced by a uniform random point in the box (per coordinate) |
| `RESAMPLE` | resampled from the search distribution until it is inside (at most 5 × d tries) |

For an unbounded problem, pass infinite bounds (`lb=np.full(d, -np.inf)`, `ub=np.full(d, np.inf)`).
A single bound can be infinite too, as `lb[i] = 0`, `ub[i] = np.inf` for a positive variable: the
finite bounds are still corrected. More in
[bound correction](../guide/bound-correction.md#infinite-bounds).

## Restarts

On multimodal problems a single run converges to a local optimum. A restart strategy starts a new
run when the search has converged or stagnated, and keeps going until the budget is used. The new
run starts with σ₀ (smaller in the small runs of BIPOP) and the initial covariance matrix again,
by default from `x0` again (`center_placement` chooses another point, below):

| `restart_strategy` | when a restart criterion holds ... |
|---|---|
| `NONE` | nothing: keep going (default); only a failed update of the matrix (`UnableToAdapt`) restarts the run |
| `STOP` | stop the run |
| `RESTART` | restart with the same population size |
| `IPOP` | restart with a population twice as large, up to 2⁹ λ₀ (Auger & Hansen 2005) |
| `BIPOP` | alternate between a growing large population and a varying small one (Hansen 2009) |

```python
def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

xopt, fopt, evaluations, es = modcma.fmin(
    rastrigin, x0=np.full(10, 3.0), sigma0=2.0, budget=100_000,
    restart_strategy="BIPOP", target=1e-8, seed=1,
)
print(fopt, evaluations)
```

The criteria that trigger a restart (no improvement, a tiny step size, a flat fitness, ...) are
objects in `es.p.criteria.items`: you can read their state, change their tolerances, remove them or
add your own; see [restarts](../guide/restarts.md). Where the search restarts is the
`center_placement` module, and repelling restarts keep it away from the optima found before.

## Stopping

A run stops when the first of these holds (`es.stop_conditions()`, checked before every
generation):

| condition | set by |
|---|---|
| the next generation would leave no evaluation for the final center of mass | `budget` (default 10 000 × d) |
| the best value is at or below the target | `target` (default: none) |
| the number of generations is reached | `max_generations` (default: none) |
| a restart criterion holds, with `restart_strategy=STOP` | the restart criteria |

Every run then ends with one evaluation of its center of mass, `es.center_of_mass` (the mean of the
distribution; clipped to the bounds when a bound correction is used), as pycma's `eval_final_mean`:
the budget always keeps that evaluation. After it `es.break_conditions()` holds.

The result is in `es.p.stats`: `global_best` (`.x`, `.y`), `evaluations`, `t` (generations), the
best solution of the current run, `current_best`, and a record of every run in `centers`: its
evaluated center of mass with `sigma`, the `metric` and the `reasons` it ended, and `std`, the
longest semi-axis of its final one-sigma ellipsoid (how far it converged).
