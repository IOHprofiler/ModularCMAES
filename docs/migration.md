# Migrating from 1.x

modcma 1.x contained two implementations: the original **pure-Python** one (`modcma.ModularCMAES`,
`modcma.AskTellCMAES`, ...) and the **C++** one (`modcma.c_maes`). Version 2.0 keeps only the C++
implementation, and makes it the top-level package.

## From the C++ version (`modcma.c_maes`)

The user-facing API of `modcma.c_maes` (`Settings`, `ModularCMAES`, `fmin`, the options of
`Modules`, the top-level functions) is the same (it is checked against a dump of the 1.x API), and
the same names are now also available as `modcma.ModularCMAES`, `modcma.Settings`, ... What is new is
additive: [ask–tell](getting-started/ask-tell.md), vectorized objectives, `Settings(seed=...)`,
`fast_sampling`, LAPACK in high dimension, presets (`modcma.preset`), warm starting
(`Settings(C0=...)`, `modcma.warm_start`), [gradient injection](guide/gradients.md),
[saving and resuming a run](guide/checkpointing.md) with `pickle`, and the record of every run in
`es.p.stats.centers`; see the [changelog](changelog.md).

The exceptions are `c_maes.utils`, a few misspelled names, the derived fields of `Settings`, the
values that `Settings` now rejects, and the `x0` of `fmin`:

| 1.x | 2.0 |
|---|---|
| `c_maes.utils.set_seed(s)`, `c_maes.utils.build_info()`, `c_maes.utils.compute_ert(...)` | `modcma.set_seed(s)`, `modcma.build_info()`, `modcma.compute_ert(...)` (also in `modcma.c_maes`); the old names still work, with a `DeprecationWarning` |
| `utils.cdf`, `utils.ppf` | `scipy.stats.norm.cdf`, `scipy.stats.norm.ppf` |
| `utils.random_uniform`, `utils.random_normal` | removed: the global generator only seeds new optimizers |
| `utils.i8_sobol`, `utils.Shuffler`, `utils.CachedShuffleSequence`, `sampling.Sobol.cache` | removed: internals of the Sobol sampler (`scipy.stats.qmc.Sobol` for Sobol points) |
| `selection.Elitsm`, `selection.NoElitsm`, `Strategy.elitsm` | `selection.Elitism`, `selection.NoElitism`, `Strategy.elitism` (the old names still work, with a `DeprecationWarning`) |
| `repelling.euclidian`, `repelling.mahanolobis` | `repelling.euclidean`, `repelling.mahalanobis` (likewise) |
| `Adaptation.adapt_evolution_paths_innner` | `Adaptation.adapt_evolution_paths_inner` (likewise) |
| `SequentialSelection(mirror, seq_cuttoff_factor=...)` | `seq_cutoff_factor=` (likewise) |
| `AdaptiveRepelling.target_rejection_rate`, `local_target_rejection_rate` | deprecated: they never had an effect |
| `Settings(mu0=m, lambda0=l)` with m > l: µ = l // 2 | `ValueError` naming both (a (1+1)-ES, `lambda0=1`, still uses µ = 1) |
| `Settings(target=nan)`: never stops | `ValueError` |
| other invalid values were accepted, or crashed the process later: `lambda0=0`, `mu0=0`, `sigma0` ≤ 0, `x0`, `lb` or `ub` of the wrong length, `lb` above `ub`, a NaN bound, a budget below one generation, `max_generations=0`, learning rates out of range (`cs=3`, `damps=-1`, `c1=0.6` with `cmu=0.6`) | `ValueError` naming the field, from the constructor and the setters ([settings](guide/settings.md)) |
| `fmin(func, x0, ...)` used only the length of `x0` and started at the origin | starts at `x0` |
| setting `Settings.db`, `center`, `diameter` or `has_bounds` | read-only: they are derived from `lb` and `ub`, which recompute them; set those (and `x0` for the starting point) |

A removed name raises an `AttributeError` that says what to use instead. (Imported with
`from ... import name`, Python reports only that the name cannot be imported.)

The internals, the methods and helpers of the module objects (samplers, step-size and matrix
adaptations, selection, repelling, `Parameters`), changed without deprecation where 2.0 changed how
they work; the [changelog](changelog.md) lists them. For example, `Parameters.perform_restart` is
gone: to force a restart, append a restart criterion that is met to `es.p.criteria.items`, and the
next generation starts with the restart.

Results are not bit-identical to 1.x: the linear algebra is batched and high dimensions use
LAPACK, which changes the last bits; these differences are statistically indistinguishable on the
BBOB functions. Results also change where a bug was fixed (integer variables, the active update,
restart criteria, ...) and where a module now follows its source (elitism is (µ + λ) selection;
MSR, PSR, SR, TPA's test points, CSA's learning rate, the xNES rates, BIPOP's population sizes,
...; the [changelog](changelog.md) lists each with its effect). Pin `modcma<2` to reproduce old
results exactly.

## From the pure-Python version

| modcma 1.x (Python) | modcma 2.0 |
|---|---|
| `ModularCMAES(func, d, budget=..., active=True, ...)` then `.run()` | `es = modcma.ModularCMAES(modcma.settings_from_dict(d, budget=..., active=True, ...))` then `es.run(func)` |
| `fmin(func, x0, budget=None, **kwargs)`, returns `xopt, fopt, evaluations` | `modcma.fmin(func, x0, sigma0, budget, **kwargs)`: `sigma0` and `budget` are required, and the optimizer is returned as a fourth value |
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
| `utils.set_seed(s)` / `np.random.seed(s)` | `modcma.set_seed(s)`, or `Settings(seed=s)` |
| `python -m modcma` | the [examples](examples/index.md) |

Module options are enums (`modcma.options`), or their names as strings in `fmin` and
`settings_from_dict`; the full list is in [modules](guide/modules.md).

Constructing `modcma.ModularCMAES` the 1.x way, or using a name that was removed, raises an error
that points here. The Python implementation remains installable as `pip install "modcma<2"`, and its
source is on the [`legacy-python`](https://github.com/IOHprofiler/ModularCMAES/tree/legacy-python)
branch.
