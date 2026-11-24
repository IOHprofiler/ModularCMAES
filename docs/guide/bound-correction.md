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
**`modcma.options.CorrectionMethod`**: What happens to a sample outside the box bounds (`Modules.bound_correction`; default NONE). The bounds are `Settings.lb` and `Settings.ub` (default [-5, 5] in every coordinate). The methods other than RESAMPLE move each coordinate that is out of bounds into [lb, ub]; the mutation vector of the sample is then recomputed from the corrected point, so the adaptation learns from the point that was evaluated. The methods are those of Caraffini, Kononova & Corne (2019), "Infeasibility and structural bias in differential evolution".

| value | |
|---|---|
| `NONE` | No correction, the default: points outside the bounds are evaluated as they are (and counted in `Parameters.bounds.n_out_of_bounds`). |
| `MIRROR` | Reflect the coordinate at the violated bound (periodically, for points more than a domain width outside). |
| `COTN` | Complete one-tailed normal correction: place the coordinate inside the violated bound, at a random distance sigma * \|N(0, 1/3^2)\| times the domain width ub - lb. |
| `UNIFORM_RESAMPLE` | Replace the coordinate by a uniform random value in [lb, ub]. |
| `SATURATE` | Set the coordinate to the violated bound (clipping): corrected points lie on the boundary. |
| `TOROIDAL` | Wrap the coordinate around: leaving at one bound enters at the other (a torus). |
| `RESAMPLE` | Sample the whole individual again until it lies inside the bounds, at most 5 d times; after that, the last sample is evaluated as it is. |
<!-- /options -->
