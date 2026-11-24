# Migrating from 1.x

modcma 1.x contained two implementations: the original **pure-Python** one (`modcma.ModularCMAES`,
`modcma.AskTellCMAES`, ...) and the **C++** one (`modcma.c_maes`). Version 2.0 keeps only the C++
implementation, and makes it the top-level package.

## From the C++ version (`modcma.c_maes`)

Nothing needs to change: the API of `modcma.c_maes` is the same (it is checked against a dump of the
1.x API), and the same names are now also available as `modcma.ModularCMAES`, `modcma.Settings`,
... What is new is additive: [ask–tell](getting-started/ask-tell.md), vectorized objectives,
`Settings(seed=...)`, `fast_sampling`, LAPACK in high dimension; see the [changelog](changelog.md).

Results are not bit-identical to 1.x: the linear algebra is batched, high dimensions use LAPACK,
and several bugs were fixed that changed results (integer variables, the active update, restart
criteria, ...). The differences are statistically indistinguishable on the BBOB functions except
where a bug was fixed. Pin `modcma<2` to reproduce old results exactly.

## From the pure-Python version

| modcma 1.x (Python) | modcma 2.0 |
|---|---|
| `ModularCMAES(func, d, budget=..., active=True, ...)` then `.run()` | `es = modcma.ModularCMAES(modcma.settings_from_dict(d, budget=..., active=True, ...))` then `es.run(func)` |
| `fmin(func, x0, sigma0, budget, **kwargs)` | `modcma.fmin(func, x0, sigma0, budget, **kwargs)` (also returns the optimizer) |
| `AskTellCMAES(d, ...)`, `ask()` one point, `tell(x, f)` | `es.ask()` returns a `(k, d)` array, `es.tell(f)` takes its values ([ask–tell](getting-started/ask-tell.md)) |
| `evaluate_bbob(fid, d, ...)` | `ioh.get_problem(...)` and `es.run(problem)` ([BBOB example](examples/bbob.md)) |
| `BIPOPParameters`, `local_restart="BIPOP"` | `restart_strategy="BIPOP"` |
| `base_sampler="sobol"` / `"halton"` | `sampler="SOBOL"` / `"HALTON"` |
| `weights_option="1/2^lambda"` | `weights="EXPONENTIAL"` |
| `mirrored="mirrored pairwise"` | `mirrored="PAIRWISE"` |
| `step_size_adaptation="csa"` | `ssa="CSA"` |
| `bound_correction="saturate"` | `bound_correction="SATURATE"` |
| `orthogonal=True`, `sequential=True` | `orthogonal=True`, `sequential_selection=True` |
| `parameters.xopt`, `parameters.fopt` | `es.p.stats.global_best.x`, `.y` |
| `parameters.sigma`, `parameters.m`, `parameters.C` | `es.p.mutation.sigma`, `es.p.adaptation.m`, `es.p.adaptation.C` |
| `utils.set_seed(s)` / `np.random.seed(s)` | `modcma.utils.set_seed(s)`, or `Settings(seed=s)` |
| `python -m modcma` | the [examples](examples/index.md) |

Module options are enums (`modcma.options`), or their names as strings in `fmin` and
`settings_from_dict`; the full list is in [modules](guide/modules.md).

Constructing `modcma.ModularCMAES` the 1.x way, or using a name that was removed, raises an error
that points here. The Python implementation remains installable as `pip install "modcma<2"`, and its
source is on the [`legacy-python`](https://github.com/IOHprofiler/ModularCMAES/tree/legacy-python)
branch.
