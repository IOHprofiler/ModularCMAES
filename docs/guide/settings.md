# Settings

`Settings` holds the problem and the parameters of a run; the algorithm itself is chosen by the
[modules](modules.md) (`Settings(..., modules=...)`; on the difference, see
[modules and features](modules.md#modules-and-features)). Only the dimension is required; every
other argument is a keyword argument with a default:

| argument | default | meaning |
|---|---|---|
| `dim` | (required) | the number of variables d |
| `modules` | `Modules()` | the [modules](modules.md) |
| `x0` | the center of the bounds | the starting point (with `center_placement=X0`) |
| `sigma0` | 2.0 | the initial step size σ₀; rule of thumb: 0.3 × the width of the search region |
| `C0` | the identity | the initial covariance matrix C, (d, d), symmetric positive definite ([warm starting](#the-initial-covariance-matrix-and-warm-starting)) |
| `lb`, `ub` | −5 and 5 in every coordinate | the bounds, arrays of length d ([bounds](bound-correction.md)) |
| `budget` | 10 000 × d | the maximum number of evaluations; at least the first generation (λ, and TPA's two points) and the evaluation of the final center of mass |
| `target` | none | stop when the best value is at or below the target (not NaN) |
| `max_generations` | none | stop after this many generations |
| `lambda0` | 4 + ⌊3 ln d⌋ | the population size λ, at least 1 (1 is the (1+1)-ES; 2 and 3 work poorly with many modules, see [compatibility](compatibility.md)) |
| `mu0` | λ / 2 (λ / 4 with `ssa=SA` or CMSA) | the number of parents µ, at least 1 and at most λ |
| `cs`, `cc`, `cmu`, `c1`, `damps`, `acov` | from d, λ and µ (Hansen 2016) | learning rates and damping of the step-size and covariance updates |
| `integer_variables` | none | indices of integer variables ([integer variables](integer-variables.md)) |
| `seed` | none (copy the global generator) | the seed of the optimizer's random generator ([reproducibility](reproducibility.md)) |
| `fast_sampling` | `False` | a faster random generator and normal sampler, with a different random stream ([performance](performance.md)) |
| `always_compute_eigv` | `False` | keep the eigendecomposition of C even when the step-size adaptation does not need it |
| `verbose` | `False` | print what happens at restarts |

An invalid value raises `ValueError` that names the argument: a dimension or `lambda0` of 0, a
`mu0` of 0 or above `lambda0`, a NaN `target`, a non-positive `sigma0`, vectors that do not have
length d, NaN in the bounds, a lower bound of +∞ or an upper bound of −∞, `lb` above `ub`, a budget
smaller than the first generation and the final center of mass, or a `C0` that is not a symmetric
positive definite (d, d) matrix. With `lambda0=1` (the (1+1)-ES) `mu0` is 1 whatever is given.

```python
import numpy as np
import modcma

settings = modcma.Settings(
    10,
    x0=np.full(10, 1.0),
    sigma0=0.3,
    lambda0=20,
    mu0=5,
    budget=50_000,
    target=1e-8,
    seed=3,
)
es = modcma.ModularCMAES(settings)
print(es.p.lamb, es.p.mu, es.p.weights.mueff)
```

## Changing the settings

The fields can also be set after construction. A setter checks only its own field: a wrong
length, NaN, a lower bound of +∞ or an upper bound of −∞, a non-finite `x0`, a NaN `target`, a
non-positive `sigma0`, `lambda0` or `mu0` of 0, a budget below 2. The checks between fields (`lb` ≤ `ub`, the
integer variables in the box, `mu0` ≤ `lambda0`, the budget against the first generation) run in
the constructor and again when an optimizer is created from the settings (`ModularCMAES(settings)`
or `Parameters(settings)`), so the fields can be set in any order:

```python
settings = modcma.Settings(2)
settings.lb = [6.0, 6.0]    # above ub = 5 until ub is set
settings.ub = [10.0, 10.0]
settings.lambda0 = 12       # mu0 follows lambda0 until it is given: 6
print(settings.center, settings.mu0)
es = modcma.ModularCMAES(settings)  # ValueError here if the fields do not fit together
```

Setting a field recomputes what is derived from it: `lb` and `ub` recompute `db`, `center`,
`diameter`, `volume` and `has_bounds`, which are read-only; `lambda0` recomputes `mu0`, unless
`mu0` was given. The setters do not make the [adjustments](compatibility.md#combinations-that-are-adjusted)
of the constructor. The optimizer rejects what they would have changed: an odd `lambda0` with
mirrored `PAIRWISE`, and a change of `lambda0` to or from 1 (the (1+1)-ES is set up by the
constructor). Changes of `settings.modules` in place are not adjusted either.

The values actually used (after defaults and [adjustments](modules.md#combinations-that-are-adjusted))
are on the optimizer: `es.p.settings` for the settings and modules, `es.p.lamb` and `es.p.mu` for the
current population size (it changes with IPOP and BIPOP), `es.p.weights` for the recombination
weights and the learning rates (`es.p.weights.c1`, `.cmu`, `.cc`, `.cs`, `.damps`, `.mueff`).

The full list, with the formulas of the defaults, is in the [reference](../reference/parameters.md).

## The initial covariance matrix and warm starting

The first generation samples N(x0, σ₀² C0). Without `C0` the matrix C starts as the identity. A
`C0` must be a (d, d) matrix, symmetric (up to rounding: its symmetric part is kept) and positive
definite. Every matrix adaptation starts from it, except `NONE`, which keeps C = I and rejects it;
`SEPARABLE` uses its diagonal (the [compatibility page](compatibility.md#combinations-that-are-adjusted)
and the [deep dive](../deep-dives/matrix-adaptations.md#the-initial-matrix-c0) give the details).
Every restart starts from `C0` again, as it starts from `sigma0` and, with `center_placement=X0`,
from `x0`.

`modcma.warm_start(solutions, gamma=0.1, alpha=0.1)` estimates `x0`, `sigma0` and `C0` from the
solutions of a related task: WS-CMA-ES (Nomura, Watanabe, Akimoto, Ozaki and Onishi, "Warm
starting CMA-ES for hyperparameter optimization", AAAI 2021). It takes the best ⌊γN⌋ of the N pairs
(x, f(x)), with mean m, and returns m as `x0` and the covariance

Σ = α² I + (1/n) ∑ᵢ (xᵢ − m)(xᵢ − m)ᵀ

(the paper's Eqs. 4 and 5) as σ₀² C0 with det(C0) = 1, the split of the `cmaes` package
(`get_warm_start_mgd`, by the paper's authors). The result is a dictionary of `Settings` arguments:

```python
import numpy as np
import modcma

d = 5
rng = np.random.default_rng(1)
rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
weights = 10.0 ** np.arange(d)


def ellipsoid(b):
    """A rotated ellipsoid with its optimum at b in every coordinate."""
    return lambda x: float(np.sum(weights * (rotation @ (x - b)) ** 2))


source, target = ellipsoid(0.6), ellipsoid(0.62)
solutions = [(x, source(x)) for x in rng.uniform(0, 1, size=(500, d))]  # a random search

ws = modcma.warm_start(solutions)
es = modcma.ModularCMAES(modcma.Settings(d, **ws, budget=10_000, target=1e-8, seed=1))
es.run(target)
print(f"sigma0 = {ws['sigma0']:.3f}, evaluations = {es.p.stats.evaluations}")
```

α is the standard deviation of the Gaussian around each selected solution, in the units of x; the
paper sets α = γ = 0.1 for variables in [0, 1] (Appendix A), so scale α with the range of the
variables. The [warm starting example](../examples/warm-start.md) compares warm and cold starts on
such a pair of tasks (d = 10, median of 20 seeds): with the 1000 points of a random search on the
source task, the warm start needs 18% fewer evaluations than the cold start, and without `C0` it
needs as many; with the 1000 evaluations of a CMA-ES run as the source it needs 5% fewer, and on a
target with another rotation 2% more.
