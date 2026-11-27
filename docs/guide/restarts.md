# Restarts

A CMA-ES run converges to one optimum. On multimodal problems, restarting from elsewhere, often with
a larger population, is the most effective way to find a better one. Four modules work together:
the restart strategy (what to do), the restart criteria (when), the center placement (where to
restart) and repelling restarts (where *not* to).

## Restart strategies

| `restart_strategy` | when a criterion holds |
|---|---|
| `NONE` | nothing happens; the run continues until the budget is used (default) |
| `STOP` | the run ends, and the optimizer stops (`stop_conditions()`, then `break_conditions()`) |
| `RESTART` | restart with the same λ |
| `IPOP` | restart with λ doubled, up to 2⁹ λ₀ (Auger & Hansen 2005) |
| `BIPOP` | alternate between a regime of large populations (doubling, up to 2⁹ λ₀ as IPOP) and one of small populations with a random size between λ₀ and the latest large one: each restart uses the regime that has used fewer evaluations so far (Hansen 2009; the population sizes as pycma's `fmin`) |

IPOP and BIPOP are the strongest choices on multimodal functions with a global structure (such as
Rastrigin); see the [multimodal example](../examples/multimodal.md).

## Restart criteria

A restart happens when any criterion in `es.p.criteria.items` is met. With `NONE` only
`UnableToAdapt` is used (the matrix could not be updated); with the other strategies the list
contains the stopping criteria of pycma: `MinSigma`, `MaxSigma`, `ExceededMaxIter`,
`NoImprovement`, `Stagnation`, `MinDSigma`, `MaxDSigma`, `FlatFitness`, `TolX`, `NoEffectCoord`
and, for `COVARIANCE`, `SEPARABLE` and `NONE`, `ConditionC` and `NoEffectAxis`; and
`TooMuchRepelling` with repelling restarts. The other matrix adaptations would need an O(d³)
decomposition per update for `ConditionC` and `NoEffectAxis`, and use the diagonal of C where the
σ criteria need its largest or smallest eigenvalue
([compatibility](compatibility.md#combinations-that-change-the-meaning-of-an-option)). Each
criterion is described in the [reference](../reference/restart.md).

```python
import numpy as np
import modcma

def rastrigin(X):
    return 10 * X.shape[1] + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)

settings = modcma.settings_from_dict(
    5, restart_strategy="IPOP", x0=np.full(5, 3.0), budget=20_000, seed=1
)
es = modcma.ModularCMAES(settings)
print([criterion.name for criterion in es.p.criteria.items])
es.run(rastrigin, vectorized=True)
print("runs:", len(es.p.stats.centers), "final lambda:", es.p.lamb)
```

The tolerances are class attributes, shared by all optimizers (set them before creating one):

```python
modcma.restart.MinSigma.tolerance = 1e-12   # restart once sigma < 1e-12 (default 1e-20)
modcma.restart.MinSigma.tolerance = 1e-20   # back to the default
```

The list can be changed per optimizer: remove criteria, or add your own by subclassing
`Criterion`:

```python
class BudgetPerRun(modcma.restart.Criterion):
    """Restart every 1000 evaluations."""

    def __init__(self):
        super().__init__("BudgetPerRun")
        self.start = 0

    def on_reset(self, parameters):
        self.start = parameters.stats.evaluations

    def update(self, parameters):
        self.met = parameters.stats.evaluations - self.start >= 1000

es = modcma.ModularCMAES(modcma.settings_from_dict(5, restart_strategy="RESTART", budget=5_000, seed=1))
es.p.criteria.items = [BudgetPerRun()]
es.run(rastrigin, vectorized=True)
print("runs:", len(es.p.stats.centers))
```

Every run ends with an evaluation of its center of mass (the mean, clipped to the bounds), at a
restart and when the optimizer stops, as pycma's `eval_final_mean`. `es.p.stats.centers` holds a
record of every run (`Center`): the evaluated point (`x`, `y`), the step size `sigma` and the shape
`metric` the run ended with, the `reasons` it ended, and `std`, σ times the longest axis of the
shape, which tells how far the run converged:

```python
for center in es.p.stats.centers:
    print(f"y = {center.y:9.3g}  std = {center.std:.1e}  reasons = {center.reasons}")
```

* Evaluating the center of mass costs one evaluation per run; the budget always keeps it.
* Repelling restarts archive the better of the center of mass and the best solution of the run.
* The center of the new run, chosen by `center_placement`, is not evaluated.

## Where a restart begins

`center_placement` chooses the mean of each new run: `X0` (the starting point, default),
`ZERO`, `UNIFORM` (a uniform random point in the bounds), `CENTER` (the center of the bounds),
or, with repelling restarts, `NOVELTY_WEIGHTED` and `MAXIMIN_TABOO`, which favour points far from
the optima found so far.

`UNIFORM`, `NOVELTY_WEIGHTED` and `MAXIMIN_TABOO` sample their points in the box `[lb, ub]`. In a
coordinate with an infinite bound there is no uniform distribution; they sample uniformly in
[x0 − 2σ0, x0 + 2σ0] instead, with x0 the starting point (`Settings.x0`, or `Settings.center`,
the [rule for infinite bounds](bound-correction.md#where-the-search-starts)). With one finite bound the interval is shifted so that it
starts at lb (or ends at ub) when it would cross it. `NOVELTY_WEIGHTED` and `MAXIMIN_TABOO`
divide the distances in such a coordinate by the width of this interval, 4σ0, as they divide
the others by ub − lb.

## Repelling restarts

Repelling restarts keep each new run away from the optima found before (de Nobel et al. 2024): the
centers of mass of earlier runs (or their best points, when better) become *taboo points*, with a
region around each, and samples in a taboo region are rejected unless a hill–valley test shows that they are in a different basin of attraction.
This helps to find *different* optima, for example when several good solutions are wanted, and
avoids spending restarts on optima that are already known.

* `repelling_type`: `NONE` (default), `COVERAGE` (the size of the taboo regions follows the
  fraction of the space covered) or `ADAPTIVE` (the size adapts to the rate of rejected samples).
* `repelling_metric`: the distance to a taboo point: `EUCLIDEAN` (default), `CURRENT` (in the
  metric of the current covariance matrix) or `ARCHIVED` (in the metric of the run that found the
  point).

They change what `ask()` returns: the points of the hill–valley tests are asked one at a time.

## The options

<!-- options: RestartStrategy -->
**`modcma.options.RestartStrategy`**: When the optimizer restarts, and with which population size (`Modules.restart_strategy`; default NONE). A restart happens when a restart criterion (`Parameters.criteria`, module `restart`) is met: the mean is placed again (`CenterPlacement`), sigma is reset to sigma0 (BIPOP: possibly smaller) and the adaptation starts over. Except for NONE, the strategies use the full set of criteria (see `restart.Criteria`). With lambda = 1, IPOP and BIPOP become RESTART.

| value | |
|---|---|
| `NONE` | No restart criteria other than `restart.UnableToAdapt`, the default: the optimizer only restarts (with the same population size) when its adaptation fails. |
| `STOP` | Use the restart criteria as stopping criteria: the run ends when one of them is met, without a restart. |
| `RESTART` | Restart with the same population size whenever a restart criterion is met. |
| `IPOP` | IPOP-CMA-ES (Auger & Hansen 2005): every restart multiplies mu and lambda by `restart.IPOP.ipop_factor` (2), up to 2^9 lambda0 (`restart.IPOP.max_lambda_factor`). Larger populations help on multimodal functions with a global structure. |
| `BIPOP` | BIPOP-CMA-ES (Hansen 2009): alternates between a large population, which doubles every time it is used up to 2^9 lambda0 (as IPOP), and a small one with a random size between lambda0 and the latest large population (as pycma) and a smaller initial sigma. Each restart uses the regime that has used fewer evaluations so far, so both get about half of the evaluations; a small run takes at most half of what the large regime has used. See `restart.BIPOP`. |
<!-- /options -->

<!-- options: RepellingType -->
**`modcma.options.RepellingType`**: Repelling restarts (`Modules.repelling_type`; default NONE). After a restart, the result of the run joins an archive of taboo points, unless hill-valley tests (which evaluate the objective) show that it lies in the basin of a point already there. New samples too close to a taboo point are rejected and drawn again, which steers later runs away from the basins already found (paper: doi:10.1007/978-3-031-70068-2_17). Only useful with a restart strategy; adds the restart criterion `restart.TooMuchRepelling`.

| value | |
|---|---|
| `NONE` | No repelling, the default. |
| `COVERAGE` | Radii from a coverage of the search volume: the radius of a taboo point grows with the number of runs that ended in its basin and shrinks as the total number of runs grows. |
| `ADAPTIVE` | Radii adapted from the rejection rate of the samples and the evaluations the runs need, between a minimum and a maximum radius. |
<!-- /options -->
