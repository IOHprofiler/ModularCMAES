# Step-size adaptation

The step size σ scales all mutations; its adaptation decides how fast the search converges and
whether it escapes from a slope. `ssa` chooses the method. The default, cumulative step-size
adaptation (CSA), compares the length of the evolution path (the recent movement of the mean) with
its expected length under random selection, and increases σ when the steps add up.

| option | idea | notes |
|---|---|---|
| `CSA` | cumulative step-size adaptation (Hansen & Ostermeier 2001) | the default |
| `TPA` | two-point adaptation: evaluate the mean moved forward and backward along its last step (Hansen 2008) | evaluates 2 extra points per generation (`ask()` returns λ + 2) |
| `MSR` | median success rule: compare the population with the median of the previous one (Ait Elhara et al. 2013) | |
| `PSR` | population success rule (Loshchilov 2014) | |
| `XNES`, `MXNES`, `LPXNES` | natural-gradient step-size updates in the style of xNES (Glasmachers et al. 2010) and variants | |
| `SR` | the 1/5th success rule, for a (1+1)-ES | chosen automatically with `lambda0=1` |
| `SA` | self-adaptation: each point has its own σ, selected along with it | with `matrix_adaptation=CMSA`: CMSA-ES |

All methods other than CSA work without the eigendecomposition of C, so with the default
`COVARIANCE` they switch it to `COVARIANCE_NO_EIGV` ([adjusted
combinations](modules.md#combinations-that-are-adjusted)).

```python
import numpy as np
import modcma

def sphere(X):
    return np.sum(X**2, axis=1)

for ssa in ["CSA", "TPA", "MSR", "PSR", "XNES"]:
    es = modcma.ModularCMAES(modcma.settings_from_dict(
        10, ssa=ssa, x0=np.full(10, 3.0), budget=10_000, target=1e-8, seed=1
    ))
    es.run(sphere, vectorized=True)
    print(f"{ssa:5s} evaluations = {es.p.stats.evaluations:6d}  final sigma = {es.p.mutation.sigma:.2e}")
```

The current step size is `es.p.mutation.sigma`; the method's state and parameters are on
`es.p.mutation` ([reference](../reference/mutation.md)). The damping and learning rate (`damps`,
`cs`) are [settings](settings.md).

## The options

<!-- options: StepSizeAdaptation -->
