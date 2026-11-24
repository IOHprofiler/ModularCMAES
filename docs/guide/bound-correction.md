# Bound correction

The search space is a box `[lb, ub]` (`Settings`; by default [−5, 5]^d). Sampled points can fall
outside it; `bound_correction` decides what happens to them before they are evaluated. The
corrected point replaces the sample in the update, as if it had been sampled there.

| `bound_correction` | a coordinate outside the box is ... |
|---|---|
| `NONE` | left as it is: the objective is evaluated outside the box (default) |
| `SATURATE` | set to the bound it crossed |
| `MIRROR` | reflected at the bound |
| `TOROIDAL` | wrapped around to the other side |
| `COTN` | moved inside, by a random distance (half-normal) from the bound it crossed |
| `UNIFORM_RESAMPLE` | replaced by a uniform random value between the bounds |
| `RESAMPLE` | the whole point is resampled from the search distribution, at most 5 × d times |

The number of points of the current population that were outside the box is
`es.p.bounds.n_out_of_bounds`.

```python
import numpy as np
import modcma

def sphere_shifted(X):
    return np.sum((X - 4.5) ** 2, axis=1)  # optimum at 4.5, near the upper bound 5

for method in ["NONE", "SATURATE", "MIRROR", "COTN", "TOROIDAL"]:
    es = modcma.ModularCMAES(
        modcma.settings_from_dict(10, bound_correction=method, budget=10_000, target=1e-8, seed=1)
    )
    outside = 0
    while es.step(sphere_shifted, vectorized=True):
        outside += es.p.bounds.n_out_of_bounds
    print(f"{method:9s} evaluations = {es.p.stats.evaluations:6d}  outside the box = {outside}")
```

Which method is best depends on the problem; de Nobel et al. (2021) compared them as modules on
BBOB. With `NONE` the objective must accept any point; for an unconstrained problem, set the
bounds to ±∞.

## The options

<!-- options: CorrectionMethod -->
