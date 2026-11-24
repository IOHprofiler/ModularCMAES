# API reference

Everything is importable from the top-level package: `modcma.fmin`, `modcma.ModularCMAES`,
`modcma.Settings`, `modcma.Modules`, and the submodules `modcma.options`, `modcma.restart`, ... (the
same objects as in `modcma.c_maes`, which stays available under that name).

| page | contents |
|---|---|
| [modcma](modcma.md) | `fmin` and the helpers for configurations (`settings_from_dict`, `get_configspace`, ...) |
| [options](options.md) | the enums that select a module: `MatrixAdaptationType`, `StepSizeAdaptation`, ... |
| [parameters](parameters.md) | `Settings`, `Modules`, `Parameters`, `Stats`, `Weights`, `Solution` |
| [ModularCMAES](modularcmaes.md) | the optimizer: `run`, `step`, `ask`, `tell`; and `Population` |
| [restart](restart.md) | restart strategies and the criteria that trigger a restart |
| [matrix_adaptation](matrix_adaptation.md) | the covariance (or matrix) adaptations |
| [mutation](mutation.md) | step-size adaptation, threshold convergence, sequential selection, sampled step sizes |
| [selection](selection.md) | elitism and pairwise selection |
| [sampling](sampling.md) | samplers (random, Sobol, Halton, mirrored, orthogonal) and sample transformers |
| [bounds](bounds.md) | bound correction methods |
| [repelling](repelling.md) | repelling restarts |
| [center](center.md) | placement of the center at a restart |
| [es](es.md) | the simple (1+1)- and (µ,λ)-ES |
| [constants, set_seed, build_info, compute_ert](constants.md) | library-wide constants (LAPACK thresholds, caches), the global seed, how the extension was built, and the expected running time of a set of runs |

The reference is generated from the docstrings, which are also what `help()` and your editor show.
