# Coming from pycma

[pycma](https://github.com/CMA-ES/pycma) is the reference Python implementation of CMA-ES by its
author. modcma implements the same algorithm in C++ (so it is
[faster per generation](../performance.md)), with the variants as modules. The interfaces map
almost one to one:

```python
--8<-- "examples/from_pycma.py"
```

Differences to keep in mind:

* **Defaults.** pycma uses the active covariance update by default; modcma does not
  (`active=True`). pycma's default bounds are none; modcma's default box is [−5, 5]^d, which matters
  only with a `bound_correction` other than `NONE`, or for `center_placement` other than `X0`.
* **Shapes.** `ask()` returns a `(k, d)` numpy array (pycma: a list of arrays), and `tell` takes only
  the values (`tell(f)`), or the values and the points when they differ from the asked ones
  (`tell(f, X)`; pycma: `tell(X, f)`).
* **Stopping.** modcma stops at the budget, the target or `max_generations`; pycma's termination
  criteria (`tolx`, `tolfun`, ...) are modcma's restart criteria, which stop the run with
  `restart_strategy="STOP"` and restart it with `RESTART`, `IPOP` or `BIPOP`
  ([restarts](../guide/restarts.md)).
* **Restarts.** `fmin(..., restarts=n)` stops after n restarts with a larger population; modcma
  restarts until the budget or the target is reached. The populations of IPOP and of BIPOP's
  large runs stop growing at 2⁹ λ₀, the largest population of `fmin` with `restarts=9`, and
  restarts continue at that size. Up to that limit, BIPOP's population sizes are those of
  `fmin(..., bipop=True)` with an integer `popsize`. (pycma's default popsize, 4 + 3 ln d, is not
  rounded there: at d = 5 its large runs have 17, 35, 70, … points, modcma's 16, 32, 64, ….)
* **Results.** The best solution is `es.p.stats.global_best` (`.x`, `.y`), the center of mass (the
  mean, pycma's `xfavorite`) is `es.center_of_mass`; the state is in `es.p`. As with pycma's default
  `eval_final_mean=True`, every run ends with an evaluation of its mean; modcma keeps one
  evaluation of the budget for it (pycma can exceed `maxfevals` by one), and records each run in
  `es.p.stats.centers`.
