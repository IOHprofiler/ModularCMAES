# Settings

`Settings` holds the problem and the parameters of a run. Only the dimension is required; every
other argument is a keyword argument with a default:

| argument | default | meaning |
|---|---|---|
| `dim` | (required) | the number of variables d |
| `modules` | `Modules()` | the [modules](modules.md) |
| `x0` | the center of the bounds | the starting point (with `center_placement=X0`) |
| `sigma0` | 2.0 | the initial step size σ₀; rule of thumb: 0.3 × the width of the search region |
| `lb`, `ub` | −5 and 5 in every coordinate | the bounds, arrays of length d ([bounds](bound-correction.md)) |
| `budget` | 10 000 × d | the maximum number of evaluations |
| `target` | none | stop when the best value is at or below the target |
| `max_generations` | none | stop after this many generations |
| `lambda0` | 4 + ⌊3 ln d⌋ | the population size λ |
| `mu0` | λ / 2 (λ / 4 with `ssa=SA` or CMSA) | the number of parents µ |
| `cs`, `cc`, `cmu`, `c1`, `damps`, `acov` | from d, λ and µ (Hansen 2016) | learning rates and damping of the step-size and covariance updates |
| `integer_variables` | none | indices of integer variables ([integer variables](integer-variables.md)) |
| `seed` | none (copy the global generator) | the seed of the optimizer's random generator ([reproducibility](reproducibility.md)) |
| `fast_sampling` | `False` | a faster random generator and normal sampler, with a different random stream ([performance](performance.md)) |
| `always_compute_eigv` | `False` | keep the eigendecomposition of C even when the step-size adaptation does not need it |
| `verbose` | `False` | print what happens at restarts |

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

The values actually used (after defaults and [adjustments](modules.md#combinations-that-are-adjusted))
are on the optimizer: `es.p.settings` for the settings and modules, `es.p.lamb` and `es.p.mu` for the
current population size (it changes with IPOP and BIPOP), `es.p.weights` for the recombination
weights and the learning rates (`es.p.weights.c1`, `.cmu`, `.cc`, `.cs`, `.damps`, `.mueff`).

The full list, with the formulas of the defaults, is in the [reference](../reference/parameters.md).
