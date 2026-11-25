# Restarts

A CMA-ES run converges to one optimum. On multimodal problems, restarting from elsewhere, often with
a larger population, is the most effective way to find a better one. Four modules work together:
the restart strategy (what to do), the restart criteria (when), the center placement (where to
restart) and repelling restarts (where *not* to).

## Restart strategies

| `restart_strategy` | when a criterion holds |
|---|---|
| `NONE` | nothing happens; the run continues until the budget is used (default) |
| `STOP` | the run stops (`break_conditions()` is true) |
| `RESTART` | restart with the same λ |
| `IPOP` | restart with λ doubled (Auger & Hansen 2005) |
| `BIPOP` | alternate between a regime of large populations (doubling) and one of small populations with a random size, giving each about the same budget (Hansen 2009) |

IPOP and BIPOP are the strongest choices on multimodal functions with a global structure (such as
Rastrigin); see the [multimodal example](../examples/multimodal.md).

## Restart criteria

A restart happens when any criterion in `es.p.criteria.items` is met. With `NONE` only
`UnableToAdapt` is used (the matrix could not be updated); with the other strategies the list
contains the stopping criteria of pycma: `MinSigma`, `MaxSigma`, `ExceededMaxIter`,
`NoImprovement`, `Stagnation`, `MinDSigma`, `MaxDSigma`, `FlatFitness` and, for
`COVARIANCE`, `TolX`, `ConditionC`, `NoEffectAxis`, `NoEffectCoord`; and `TooMuchRepelling`
with repelling restarts. Each is described in the [reference](../reference/restart.md).

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
print("restarts:", len(es.p.stats.centers), "final lambda:", es.p.lamb)
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
print("restarts:", len(es.p.stats.centers))
```

The final means of the runs that ended are in `es.p.stats.centers`, with their values.

* Evaluating the final mean costs one evaluation per restart.
* The final mean is the candidate that repelling restarts archive.
* The center of the new run, chosen by `center_placement`, is not evaluated.

## Where a restart begins

`center_placement` chooses the mean of each new run: `X0` (the starting point, default),
`ZERO`, `UNIFORM` (a uniform random point in the bounds), `CENTER` (the center of the bounds),
or, with repelling restarts, `NOVELTY_WEIGHTED` and `MAXIMIN_TABOO`, which favour points far from
the optima found so far.

## Repelling restarts

Repelling restarts keep each new run away from the optima found before (de Nobel et al. 2024): the
final means of earlier runs become *taboo points*, with a region around each, and samples in a taboo
region are rejected unless a hill–valley test shows that they are in a different basin of attraction.
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
| `IPOP` | IPOP-CMA-ES (Auger & Hansen 2005): every restart multiplies mu and lambda by `restart.IPOP.ipop_factor` (2). Larger populations help on multimodal functions with a global structure. |
| `BIPOP` | BIPOP-CMA-ES (Hansen 2009): alternates between a large population, which doubles every time it is used, and a small one with a random size and a smaller initial sigma, giving both regimes about half of the budget (`Settings.budget`). See `restart.BIPOP`. |
<!-- /options -->

<!-- options: RepellingType -->
**`modcma.options.RepellingType`**: Repelling restarts (`Modules.repelling_type`; default NONE). After a restart, the result of the run joins an archive of taboo points, unless hill-valley tests (which evaluate the objective) show that it lies in the basin of a point already there. New samples too close to a taboo point are rejected and drawn again, which steers later runs away from the basins already found (paper: doi:10.1007/978-3-031-70068-2_17). Only useful with a restart strategy; adds the restart criterion `restart.TooMuchRepelling`.

| value | |
|---|---|
| `NONE` | No repelling, the default. |
| `COVERAGE` | Radii from a coverage of the search volume: the radius of a taboo point grows with the number of runs that ended in its basin and shrinks as the total number of runs grows. |
| `ADAPTIVE` | Radii adapted from the rejection rate of the samples and the evaluations the runs need, between a minimum and a maximum radius. |
<!-- /options -->
