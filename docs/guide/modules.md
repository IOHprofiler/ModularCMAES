# Modules

The Modular CMA-ES is one evolution strategy in which its parts can be swapped
independently. Each generation it samples λ points around a mean, evaluates them, selects the best
µ, and updates the mean, the step size σ and the shape of the search distribution (a covariance
matrix, or a factor of it). Each module replaces one of those parts:

| module (`Modules` field) | options (default first) | what it changes | guide |
|---|---|---|---|
| `matrix_adaptation` | `COVARIANCE`, `MATRIX`, `SEPARABLE`, `CHOLESKY`, `CMSA`, `COVARIANCE_NO_EIGV`, `NATURAL_GRADIENT`, `NONE` | how the shape of the distribution is learned | [matrix adaptation](matrix-adaptation.md) |
| `ssa` | `CSA`, `TPA`, `MSR`, `XNES`, `MXNES`, `LPXNES`, `PSR`, `SR`, `SA` | how the step size is adapted | [step-size adaptation](step-size-adaptation.md) |
| `active` | off, on | negative weights for the worst points in the covariance update | [selection](selection.md) |
| `elitist` | off, on | (µ + λ) instead of (µ, λ) selection | [selection](selection.md) |
| `sequential_selection` | off, on | stop a generation early once it has enough improvements | [selection](selection.md) |
| `weights` | `DEFAULT`, `EQUAL`, `EXPONENTIAL` | the recombination weights | [selection](selection.md) |
| `mirrored` | `NONE`, `MIRRORED`, `PAIRWISE` | sample points in mirrored pairs (and select one of each pair) | [sampling](sampling.md) |
| `orthogonal` | off, on | orthogonalize the samples of a generation | [sampling](sampling.md) |
| `sampler` | `UNIFORM`, `SOBOL`, `HALTON` | pseudo-random or quasi-random (low-discrepancy) base samples | [sampling](sampling.md) |
| `sample_transformation` | `GAUSSIAN`, `SCALED_UNIFORM`, `LAPLACE`, `LOGISTIC`, `CAUCHY`, `DOUBLE_WEIBULL`, `NONE` | the distribution of the mutations | [sampling](sampling.md) |
| `sample_sigma` | off, on | a step size per point, sampled around σ | [sampling](sampling.md) |
| `threshold_convergence` | off, on | a minimum length of the mutations, decreasing over time | [sampling](sampling.md) |
| `restart_strategy` | `NONE`, `STOP`, `RESTART`, `IPOP`, `BIPOP` | what happens when the search has converged | [restarts](restarts.md) |
| `center_placement` | `X0`, `ZERO`, `UNIFORM`, `CENTER`, `NOVELTY_WEIGHTED`, `MAXIMIN_TABOO` | where a (re)start begins | [restarts](restarts.md) |
| `repelling_type` | `NONE`, `COVERAGE`, `ADAPTIVE` | keep restarts away from the optima found before | [restarts](restarts.md) |
| `repelling_metric` | `EUCLIDEAN`, `CURRENT`, `ARCHIVED` | the distance used by repelling restarts | [restarts](restarts.md) |
| `bound_correction` | `NONE`, `MIRROR`, `COTN`, `UNIFORM_RESAMPLE`, `SATURATE`, `TOROIDAL`, `RESAMPLE` | what happens to points outside the bounds | [bound correction](bound-correction.md) |

Not every combination is a different algorithm (`active` has no effect without a matrix
adaptation, repelling none without restarts); taking this into account, the fourteen modules of the
framework paper (de Nobel et al. 2026, [citing](../citing.md)) give 14,859,936 distinct variants
with the default parameters.

The defaults are the standard CMA-ES (Hansen & Ostermeier 2001; Hansen 2016): full covariance
matrix, cumulative step-size adaptation, (µ/µ_w, λ) selection with the default weights.

## Modules and features

Not everything you can switch on is a module. The distinction matters for how you use it:

* A **module** is an interchangeable part of the evolution strategy: one of the fields of `Modules`
  above. Each changes how the strategy samples, selects or adapts, so a different module is a
  different algorithm. Modules combine freely, form the configuration space (`get_configspace`),
  and are what you tune or compare in a benchmark.
* A **feature** changes what you can do with the optimizer, not the strategy: what the problem looks
  like, how the run is set up, driven, stopped or inspected. Features are set through `Settings` or
  the API, work with every combination of modules, and are not part of the configuration space.

| features | where |
|---|---|
| the problem: dimension, bounds (the box; what happens to points outside it is the `bound_correction` module), integer variables | `Settings(dim, lb, ub, integer_variables)` |
| the run: starting point and step size, population size, learning rates, budget, target, generations | `Settings(x0, sigma0, lambda0, mu0, cs, ..., budget, target, max_generations)` |
| evaluation: ask–tell, vectorized objectives, injected or repaired points | `ask`, `tell(f, X)`, `run(..., vectorized=True)` |
| stopping and restarting: which restart criteria apply, their tolerances, your own criteria | `es.p.criteria`, `modcma.restart` |
| reproducibility and speed: seeds, threads, the fast random generator, LAPACK and BLAS | `Settings(seed, fast_sampling)`, `modcma.constants` |
| results: the best solution, the favorite solution (the mean) | `es.p.stats`, `es.xfavorite` |

The numeric settings that change the strategy's behaviour (λ, µ, σ₀, the learning rates) are
parameters of the algorithm rather than features: they can be tuned along with the modules
(`get_configspace` includes them, [tuning](tuning.md)), but they do not change which algorithm runs.
A few settings imply a module choice: λ = 1, for example, is the (1+1)-ES ([below](#combinations-that-are-adjusted)).

## Setting modules

The options are enums in `modcma.options`, set on a `Modules` object that goes into the `Settings`:

```python
import numpy as np
import modcma

modules = modcma.Modules()
modules.matrix_adaptation = modcma.options.MatrixAdaptationType.SEPARABLE
modules.ssa = modcma.options.StepSizeAdaptation.TPA
modules.mirrored = modcma.options.Mirror.PAIRWISE
modules.active = True

settings = modcma.Settings(20, modules=modules, budget=20_000)
es = modcma.ModularCMAES(settings)
print(es.p.settings.modules)
```

`fmin` and `settings_from_dict` take the same choices as keyword arguments, enums by name:

```python
settings = modcma.settings_from_dict(
    20, matrix_adaptation="SEPARABLE", ssa="TPA", mirrored="PAIRWISE", active=True, budget=20_000
)
```

All options at once, for example to build a configuration space:

```python
for name, options in modcma.get_all_module_options().items():
    print(f"{name:24s} {options}")
```

## Well-known algorithms

Many published CMA-ES variants are a combination of modules:

| algorithm | modules |
|---|---|
| CMA-ES (Hansen & Ostermeier 2001) | the defaults |
| active CMA-ES (Jastrebski & Arnold 2006) | `active=True` |
| IPOP-CMA-ES (Auger & Hansen 2005) | `restart_strategy=IPOP` |
| BIPOP-CMA-ES (Hansen 2009) | `restart_strategy=BIPOP` |
| sep-CMA-ES (Ros & Hansen 2008) | `matrix_adaptation=SEPARABLE` |
| MA-ES (Beyer & Sendhoff 2017) | `matrix_adaptation=MATRIX` |
| CMSA-ES (Beyer & Sendhoff 2008) | `matrix_adaptation=CMSA`, `ssa=SA` |
| (1+1)-ES with the 1/5th success rule | `lambda0=1` (sets elitism and `ssa=SR`, see below) |
| mirrored sampling with sequential selection (Brockhoff et al. 2010) | `mirrored=MIRRORED`, `sequential_selection=True` |

## Combinations that are adjusted

All modules can be combined, but a few combinations are adjusted when the `Settings` are created
(check `es.p.settings.modules` for what is used):

* `ssa` other than `CSA` with `matrix_adaptation=COVARIANCE`: `COVARIANCE_NO_EIGV` is used, the
  same covariance update sampled through a Cholesky factor, because only CSA needs the
  eigendecomposition (unless `always_compute_eigv=True`).
* `ssa=SA` or `matrix_adaptation=CMSA`: µ defaults to λ/4 instead of λ/2.
* `matrix_adaptation=NONE`: `active` is switched off (there is no matrix to update).
* `mirrored=PAIRWISE`: λ is made even.
* `lambda0=1`: a (1+1)-ES: µ = 1, elitist, equal weights, the 1/5th success rule (`ssa=SR`), no
  active update or sequential selection; IPOP and BIPOP become `RESTART`.

## Configurations from a configuration space

`modcma.get_configspace(dim)` is a [ConfigSpace](https://automl.github.io/ConfigSpace/) with every
module (and, optionally, the population size, σ₀ and the learning rates), for algorithm
configuration tools such as SMAC; `modcma.settings_from_config(dim, config)` turns a sampled
configuration into `Settings`. See [tuning](tuning.md).
