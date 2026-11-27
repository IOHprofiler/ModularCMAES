# Coming from pycma

[pycma](https://github.com/CMA-ES/pycma) is the reference Python implementation of CMA-ES by its
author. modcma implements the same algorithm in C++ (so it is
[faster per generation](../performance.md)), with the variants as modules. The interfaces map
almost one to one:

```python
--8<-- "examples/from_pycma.py"
```

Differences to keep in mind:

* **Defaults.** pycma (`CMA_active=True`) and `cmaes` use the active covariance update by default;
  modcma does not, since its default is the baseline of the framework paper. To compare modcma with
  pycma or `cmaes`, set `active=True`. pycma's default bounds are none; modcma's default box is
  [−5, 5]^d, which matters only with a `bound_correction` other than `NONE`, or for
  `center_placement` other than `X0`.
* **Shapes.** `ask()` returns a `(k, d)` numpy array (pycma: a list of arrays), and `tell` takes only
  the values (`tell(f)`), or the values and the points when they differ from the asked ones
  (`tell(f, X)`; pycma: `tell(X, f)`).
* **Stopping.** modcma stops at the budget, the target or `max_generations`; pycma's termination
  criteria (`tolx`, `tolfun`, ...) are modcma's restart criteria, which stop the run with
  `restart_strategy="STOP"` and restart it with `RESTART`, `IPOP` or `BIPOP`
  ([restarts](../guide/restarts.md)).
* **Results.** The best solution is `es.p.stats.global_best` (`.x`, `.y`), the center of mass (the
  mean, pycma's `xfavorite`) is `es.center_of_mass`; the state is in `es.p`. As with pycma's default
  `eval_final_mean=True`, every run ends with an evaluation of its mean; modcma keeps one
  evaluation of the budget for it (pycma can exceed `maxfevals` by one), and records each run in
  `es.p.stats.centers`.
