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
| `weights` | `DEFAULT`, `EQUAL`, `EXPONENTIAL`, `UTILITY` | the recombination weights (`UTILITY`: NES's, also for the mean) | [selection](selection.md) |
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

The 17 modules have 2,194,698,240 combinations of options, but not every combination is a
different algorithm: `Settings` converts some into others (`COVARIANCE` becomes
`COVARIANCE_NO_EIGV` unless `ssa` is `CSA`), and some options have no effect in some combinations
(`active` without a matrix adaptation, repelling without restarts; see
[compatibility](compatibility.md)). Leaving these out, 686,685,216 distinct variants remain with
the default parameters. The count comes from the code (`tools/count_variants.py`); it does not
apply equivalences that depend on the problem, such as `X0` and `CENTER` without an `x0`. The
framework paper (de Nobel et al. 2026, [citing](../citing.md)) counted 14,859,936 for its fourteen
modules.

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
| results: the best solution, the center of mass (the mean), the record of every run | `es.p.stats`, `es.center_of_mass`, `es.p.stats.centers` |

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

Many published CMA-ES variants are a combination of modules. `modcma.preset(name, dim, ...)`
(also `Settings.preset(name, dim, ...)`, and `fmin(..., preset=name)`) gives the `Settings` of each
by name:

| preset | algorithm | modules |
|---|---|---|
| `cma-es` | CMA-ES (Hansen & Ostermeier 2001; Hansen 2016) | the defaults |
| `active-cma-es` | active CMA-ES (Jastrebski & Arnold 2006) | `active=True` |
| `sep-cma` | sep-CMA-ES (Ros & Hansen 2008) | `matrix_adaptation=SEPARABLE` |
| `ma-es` | MA-ES (Beyer & Sendhoff 2017) | `matrix_adaptation=MATRIX` |
| `cholesky-cma` | Cholesky-CMA-ES (Krause, Arbonès & Igel 2016) | `matrix_adaptation=CHOLESKY` |
| `cmsa-es` | CMSA-ES (Beyer & Sendhoff 2008) | `matrix_adaptation=CMSA`, `ssa=SA` |
| `xnes` | xNES (Glasmachers et al. 2010) | `matrix_adaptation=NATURAL_GRADIENT`, `ssa=XNES`, `weights=UTILITY` |
| `ipop` | IPOP-CMA-ES (Auger & Hansen 2005) | `restart_strategy=IPOP` |
| `bipop` | BIPOP-CMA-ES (Hansen 2009) | `restart_strategy=BIPOP` |
| `(1+1)-cma-es` | (1+1)-CMA-ES (Igel, Suttorp & Hansen 2006) | `lambda0=1` (elitist, `ssa=SR`, see below) |

```python
import numpy as np
import modcma

settings = modcma.Settings.preset("sep-cma", 20, budget=20_000)
print(settings.modules.matrix_adaptation)

xopt, fopt, evals, es = modcma.fmin(lambda x: float(np.sum(x**2)), np.ones(10), 0.5, 5_000, preset="ma-es")
```

A preset is a selection of modules; the learning rates and the population size are those that
`Settings` computes for them. Keyword arguments after the name override the preset: module options
(`preset("sep-cma", 10, active=True)`) as well as other settings (`budget`, `lambda0`, `seed`, ...).
An unknown name raises a `ValueError` that lists the valid ones (`modcma.preset_names()`).

The `(1+1)-cma-es` is the elitist (1+1) strategy that `lambda0=1` gives, built from the modules of
modcma; it is not a reimplementation of the published algorithm.

Two more combinations that have no preset:

| algorithm | modules |
|---|---|
| (1+1)-ES with the 1/5th success rule | `lambda0=1` (sets elitism and `ssa=SR`, see below) |
| mirrored sampling with sequential selection (Brockhoff et al. 2010) | `mirrored=MIRRORED`, `sequential_selection=True` |

## Combinations that are adjusted

All modules can be combined, but a few combinations are adjusted when the `Settings` are created
(check `es.p.settings.modules` for what is used). The full list, with the options that have no
effect in some combinations and the combinations to avoid, is on the
[compatibility](compatibility.md) page; how each module compares with the published algorithm is in
the [deep dives](../deep-dives/index.md). The main adjustments:

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
