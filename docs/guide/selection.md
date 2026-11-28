# Selection and recombination

Each generation the µ best of the λ points are selected, and the new mean is their weighted
average. The modules of this page change which points are selected and how they are weighted.

* `weights`: the recombination weights of the µ selected points. `DEFAULT`: decreasing with the
  rank, log-shaped (Hansen 2016); `EQUAL`: all 1/µ; `EXPONENTIAL`: halving with the rank;
  `UTILITY`: the utilities of NES and xNES, log-shaped for the better half minus a baseline, summing
  to zero. `UTILITY` is more than a set of weights: the mean moves by the utility-weighted sum of
  *all* λ steps (the worse half pulls it away from the worse points), and every matrix update uses
  all of them (it sets `active`). With `NATURAL_GRADIENT` and `ssa=XNES` it is xNES; with the
  CMA-ES updates it needs 20–35 % more evaluations than `DEFAULT` on the sphere, a rotated ellipsoid
  and Rosenbrock (d = 10), and with CMSA (which uses only the positive utilities) 14–22 % fewer on
  the rotated ellipsoid and Rosenbrock, 8 % more on the sphere. Avoid it with outlier-prone
  sampling ([compatibility](compatibility.md#combinations-to-avoid)).
* `active=True`: the active covariance update (Jastrebski & Arnold 2006): the worst points get
  negative weights in the update of C, which shrinks the variance in unpromising directions. It
  often speeds up the adaptation (pycma uses it by default); not used with
  `matrix_adaptation=NONE` or `CMSA`, which has no active variant.
* `elitist=True`: (µ + λ) selection, the parents compete with the offspring, so the best point is
  never lost. Faster on unimodal problems, but more easily trapped in a local optimum. With
  `active`, the negative weights go to the λ − µ worst points of the µ + λ pool (#161); `XNES` adapts σ
  from the offspring alone (#172) ([compatibility](compatibility.md#combinations-that-change-the-meaning-of-an-option)).
* `sequential_selection=True`: evaluate the points one by one and end the generation at the first
  point (from the µ-th on) that improves on the best of the previous generation (Brockhoff et al.
  2010); the updates
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
**`modcma.options.RecombinationWeights`**: The recombination weights (`Modules.weights`; default DEFAULT). The weights of the mu best individuals are positive and sum to 1. With `Modules.active` the other lambda - mu individuals get negative (or zero) weights, scaled to sum to -min(1 + c1 / cmu, 1 + 2 mueff^- / (mueff + 2), (1 - c1 - cmu) / (d cmu)) (Hansen 2016, Eqs. (50) to (53)). UTILITY differs from both rules.

| value | |
|---|---|
| `DEFAULT` | Log-linear weights, the default: w_i proportional to ln((n + 1) / 2) - ln(i) (Hansen 2016, "The CMA Evolution Strategy: A Tutorial", Eq. (49)). For the default mu = floor(lambda / 2), n = lambda: the weights of the standard CMA-ES. For another mu (`Settings.mu0`, or the default mu = lambda / 4 of SA and CMSA), n = 2 mu, as pycma with CMA_mu: ln(mu + 1/2) - ln(i) for the mu best, and the mu negative weights of that population go to the worst ranks (the ranks between get 0 when lambda > 2 mu; only the last lambda - mu of them are used when lambda < 2 mu) (#73). |
| `EQUAL` | Equal weights 1/mu (intermediate recombination): every selected individual counts the same; with `Modules.active` the negative weights are equal too. |
| `EXPONENTIAL` | Exponentially decaying weights, w_i = 2^-i + 1 / (mu 2^mu) for i = 1..mu, which sum to 1 (de Nobel et al. 2026, the framework paper, Sec. 4.2): the first term halves with every rank, more weight on the best individuals than DEFAULT. The negative weights follow the same rule for the lambda - mu worst, the largest on the worst. |
| `UTILITY` | The utilities of NES and xNES (Wierstra et al. 2014; Glasmachers et al. 2010), u_k = max(0, ln(lambda/2 + 1) - ln k) / sum - 1/lambda: log-shaped for the better half, minus a baseline, summing to zero. Unlike the other options they are not only the weights of the mu parents: the mean moves by the utility-weighted sum of all lambda steps, and every matrix update uses all of them (`Settings` sets `Modules.active`, except with matrix_adaptation NONE and CMSA; CMSA uses the positive utilities). The step-size rules SA, LPXNES and XNES without NATURAL_GRADIENT use the positive utilities of the mu best, normalized to sum to 1. With NATURAL_GRADIENT and StepSizeAdaptation.XNES this is xNES. |
<!-- /options -->
