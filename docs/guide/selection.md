# Selection and recombination

Each generation the µ best of the λ points are selected, and the new mean is their weighted
average. The modules of this page change which points are selected and how they are weighted.

* `weights`: the recombination weights of the µ selected points. `DEFAULT`: decreasing with the
  rank, log-shaped (Hansen 2016); `EQUAL`: all 1/µ; `EXPONENTIAL`: halving with the rank.
* `active=True`: the active covariance update (Jastrebski & Arnold 2006): the worst points get
  negative weights in the update of C, which shrinks the variance in unpromising directions. It
  often speeds up the adaptation (pycma uses it by default); not used with
  `matrix_adaptation=NONE`.
* `elitist=True`: (µ + λ) selection, the parents compete with the offspring, so the best point is
  never lost. Faster on unimodal problems, but more easily trapped in a local optimum.
* `sequential_selection=True`: evaluate the points one by one and end the generation at the first
  point (from the µ-th on) that improves on the best so far (Brockhoff et al. 2010); the updates
  use the evaluated points. Saves evaluations when improvements are easy to find, typically with
  mirrored sampling.
* Pairwise selection with mirrored sampling: `mirrored=PAIRWISE` ([sampling](sampling.md)).

The number of parents µ and the population size λ are [settings](settings.md) (`mu0`, `lambda0`);
the weights in use are `es.p.weights` (`positive`, `negative`, `mueff`, ...).

```python
import numpy as np
import modcma

def ellipsoid(X):
    return (X * X) @ (10.0 ** (6 * np.arange(X.shape[1]) / (X.shape[1] - 1)))

for name, modules in {"default": {}, "active": {"active": True}, "elitist": {"elitist": True},
                      "equal weights": {"weights": "EQUAL"}}.items():
    es = modcma.ModularCMAES(modcma.settings_from_dict(
        10, x0=np.full(10, 3.0), budget=20_000, target=1e-8, seed=1, **modules
    ))
    es.run(ellipsoid, vectorized=True)
    print(f"{name:14s} evaluations = {es.p.stats.evaluations}")
```

## The options

<!-- options: RecombinationWeights -->
