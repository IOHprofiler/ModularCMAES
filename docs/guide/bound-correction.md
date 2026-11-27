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
| `RESAMPLE` | the whole point is resampled from the search distribution, at most 5 × d times; then it is evaluated as it is |

The number of points of the current population that were outside the box is
`es.p.bounds.n_out_of_bounds`. With a correction other than `NONE` the best point that `run()` and
`fmin` return (`es.p.stats.global_best`) is the best point in the box: a point that `RESAMPLE` left
outside after its last try is evaluated and can be selected, but never becomes the best (#168).
Without a correction the box does not constrain the points, and every point counts. The two test points of the step-size rule `TPA` are not corrected
by the method: with any correction other than `NONE` both move toward the mean by the same factor
until both are in the box, which keeps TPA's comparison symmetric (they are not counted).

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

## Infinite bounds

Any bound can be infinite (`-np.inf` in `lb`, `np.inf` in `ub`), for example a variable that
must be positive. Every coordinate and every side is corrected on its own: an infinite bound is
never crossed, and a finite bound is corrected even when the other bound of its coordinate, or of
another coordinate, is infinite.

| the coordinate has | a value that crosses a finite bound is ... |
|---|---|
| two finite bounds | corrected by the method |
| one finite bound | set to that bound, as `SATURATE` does, whatever the method |
| no finite bound | never corrected |

`MIRROR`, `TOROIDAL`, `COTN` and `UNIFORM_RESAMPLE` need the width ub − lb of the coordinate
(the period of the reflection, the wrap-around, the distance from the bound, the interval), which
is infinite with one infinite bound. `RESAMPLE` samples the point again, as with a finite box.
With a correction other than `NONE`, the center of mass evaluated at the end of a run is clipped
to every finite bound.

## Fixed variables and empty boxes

A coordinate with `lb[i] == ub[i]` (finite) is a fixed variable: it holds a parameter constant
while the others are tuned. Every correction puts a value that leaves the box at `lb[i]`, without
dividing by the width, which is zero (as for a coordinate with an infinite bound, the method is
replaced by `SATURATE`).

An empty coordinate is rejected with a `ValueError`: `lb[i] > ub[i]`, `lb[i] = +inf`,
`ub[i] = -inf` (so also `lb[i] = ub[i] = ±inf`, whose width is NaN), or a NaN bound. The
constructor of `Settings` checks it, and so does the creation of an optimizer from the Settings
(`ModularCMAES`). The `lb` and `ub` setters check only their own field (no NaN; no `+inf` in `lb`,
no `-inf` in `ub`), so the two may be set in either order
([changing the settings](settings.md#changing-the-settings)).

```python
lb = np.array([0.0, -np.inf])  # x[0] >= 0
ub = np.array([np.inf, 1.0])   # x[1] <= 1

def outside(x):
    return float((x[0] + 3) ** 2 + (x[1] - 3) ** 2)  # the minimum (-3, 3) is outside

es = modcma.ModularCMAES(
    modcma.settings_from_dict(2, lb=lb, ub=ub, bound_correction="MIRROR", seed=1, budget=1_000)
)
es.run(outside)
print(es.p.stats.global_best.x)  # the corner (0, 1)
```

`Settings.has_bounds` is `False` when the box is not finite; then `diameter` is 10 and `volume`
10^d. The bound correction does not depend on it. The center
placements `UNIFORM`, `NOVELTY_WEIGHTED` and `MAXIMIN_TABOO` sample in the box; in a coordinate
with an infinite bound they sample in x0 ± 2σ0 instead ([restarts](restarts.md#where-a-restart-begins)).

### Where the search starts

Without `x0`, the first mean is `Settings.center`, per coordinate:

| the coordinate has | `center[i]` |
|---|---|
| two finite bounds | the midpoint (lb + ub) / 2 |
| no finite bound | 0 |
| one finite bound | 0 if the half-line holds 0; otherwise the finite bound moved inward by σ0 (lb + σ0, or ub − σ0) |

The start then lies inside the box. With one infinite bound `center` depends on `sigma0`, and
follows it when it is set.

## The options

<!-- options: CorrectionMethod -->
**`modcma.options.CorrectionMethod`**: What happens to a sample outside the box bounds (`Modules.bound_correction`; default NONE). The bounds are `Settings.lb` and `Settings.ub` (default [-5, 5] in every coordinate). The methods other than RESAMPLE move each coordinate that is out of bounds into [lb, ub]; the mutation vector of the sample is then recomputed from the corrected point, so the adaptation learns from the point that was evaluated. The methods are those of Caraffini, Kononova & Corne (2019), "Infeasibility and structural bias in differential evolution". Bounds may be infinite; each coordinate and each side is corrected separately, and an infinite bound is never crossed. MIRROR, COTN, UNIFORM_RESAMPLE and TOROIDAL need the width ub - lb: in a coordinate with one infinite bound they set a coordinate that crosses the finite bound to that bound, as SATURATE does.

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
