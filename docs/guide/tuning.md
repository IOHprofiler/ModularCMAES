# Tuning

Because every module and parameter is a choice, finding a good configuration for a class of
problems is an algorithm configuration problem. modcma provides the configuration space and the
conversion from a configuration to `Settings`, for tools such as
[SMAC](https://github.com/automl/SMAC3), irace, or a random search.

```python
import modcma

space = modcma.get_configspace(dim=10)   # every module, λ, µ, σ₀ and the learning rates
print(len(list(space.values())), "hyperparameters")

config = space.sample_configuration()
settings = modcma.settings_from_config(10, config, budget=5_000)
print(settings.modules)
```

`get_configspace(dim, add_popsize=True, add_sigma=True, add_learning_rates=True,
remove_duplicates=True)` leaves out the numeric parameters you do not want to tune. The default
configuration of the space is the standard CMA-ES: `space.get_default_configuration()` gives the
`Settings` of the plain constructor.

## The numeric parameters

`lambda0`, `mu0` (integers in [1, 50 d], log scale, with µ ≤ λ), `sigma0` (a normal distribution
around its default), `cs`, `cc`, `cmu`, `c1` (in [0, 1]) and `damps` (in [0, 10]) are absolute
values. The lower bound of `cs`, `cc` and `damps` is 1e-6, because `Settings` rejects a value of 0
or less; `cmu` and `c1` may be 0. Their defaults are those of `Settings(dim)` with the default modules, CSA with `COVARIANCE`.

The defaults of other modules differ (for example `cs` of `XNES`, `SR` and `CSA`, or µ = λ/4 for
`SA` and `CMSA`). So `settings_from_config` passes a numeric value to `Settings` only when it
differs from the default of the space: a value equal to the default stands for the default of the
chosen modules, which `Settings` computes itself. Any other value is passed as it is.
`mu0` is passed whenever `lambda0` is passed and whenever it differs from its default, so a sampled
`mu0` that equals the default is kept (`lambda0 = 20, mu0 = 3` runs with µ = 3, and so does
`mu0 = 10` next to a `lambda0` that differs from the default).

### Configurations that are rejected

The ranges of `c1` and `cmu` are independent, so a sample can have c1 + cmu > 1. `Settings`
rejects it, because the covariance matrix would lose positive definiteness (Hansen 2016). The
`ValueError` is raised by `settings_from_config`, when it builds the `Settings`, and names both
values. A tuner should record the configuration as a crashed run. Every other draw of the space is
valid.

## Conditions

A hyperparameter without an effect is inactive: it is absent from a sampled configuration, so a
tuner does not spend evaluations on it.

| Hyperparameter | Inactive when (otherwise active) |
|---|---|
| `active` | `matrix_adaptation` is `NONE` or `CMSA` (`Settings` switches it off), or `lambda0` is 1 |
| `center_placement`, `repelling_type` | `restart_strategy` is `STOP` |
| `repelling_metric` | `repelling_type` is `NONE` |
| `cs` | `ssa` is `SR` or `SA` (or `lambda0` is 1, which runs `SR`), and `matrix_adaptation` is `NONE`, `CHOLESKY`, `CMSA` or `NATURAL_GRADIENT` |
| `damps` | `ssa` is `TPA`, `XNES`, `MXNES`, `LPXNES` or `SA`, and `lambda0` is above 1 (a (1+1)-ES runs `SR`) |
| `cc` | `matrix_adaptation` is `NONE`, `MATRIX` or `CMSA` |
| `cmu`, `c1` | `matrix_adaptation` is `NONE` or `CMSA`, or `NATURAL_GRADIENT` without `active` |
| `mu0`, `elitist`, `sequential_selection`, `weights`, `ssa` | `lambda0` is 1 (a (1+1)-ES: mu = 1, elitist, equal weights, `SR`; `Settings` overrides them) |

A hyperparameter is inactive only where the C++ code never reads the parameter; the file and line
(or function) of each reason are in the comment of `_INACTIVE_FOR` in `modcma/c_maes/__init__.py`.
Where it is read in some combinations only, it stays active, which costs a tuner some budget but
hides no dimension. For example, `restart_strategy=NONE` still restarts when the adaptation fails,
so `center_placement` stays active; `cmu` and `c1` scale the negative weights, so they matter for
`NATURAL_GRADIENT` with `active`. `tests/test_c_tuning.py` checks that changing an inactive
parameter leaves a run unchanged, for every pair of step-size rule and matrix adaptation.
The thresholds of threshold convergence, `ipop_factor` and the other numeric parameters of modules
are not hyperparameters.

## Configurations left out

* `sample_transformation=NONE` is not in the space. It passes the raw uniform samples (mean 0.5)
  as mutations, which fails everywhere ([compatibility](compatibility.md)). `Settings` accepts it,
  and `get_all_module_options` lists it.
* `Settings` replaces `matrix_adaptation=COVARIANCE` by `COVARIANCE_NO_EIGV` for every `ssa` other
  than `CSA`, and for `lambda0` = 1 (which runs `SR`), so the two are the same configuration
  there. The space forbids `COVARIANCE` with those; `get_configspace(remove_duplicates=False)`
  keeps it.
* `IPOP` and `BIPOP` with `lambda0` = 1 are forbidden: `Settings` runs `RESTART` there; so is
  `mirrored=PAIRWISE` with `lambda0` = 1, which `Settings` turns into `NONE`.
* Not removed, so the space holds the same algorithm more than once:
  * `mirrored=PAIRWISE` makes an odd `lambda0` above 1 even, so `lambda0` = k and k + 1 (k odd,
    k > 1) are one configuration. Forbidding every odd value would add a clause per value.
  * With µ = λ there are no negative weights and no unselected offspring, so `active` and
    `sequential_selection` have nothing to act on. This is a relation of two hyperparameters, not
    a condition.
  * `ssa` stays a hyperparameter for `lambda0` = 1, although `Settings` runs `SR` there: it decides
    whether `COVARIANCE` becomes `COVARIANCE_NO_EIGV`.

The module part of this space is not a count of distinct variants (see
[how many variants](modules.md#how-many-variants)): it leaves out `sample_transformation=NONE`,
and its conditions and forbidden clauses remove only part of the combinations that coincide.

A complete example, a random search for the best configuration on a few functions, is
[tuning a configuration](../examples/tuning.md). The study of de Nobel et al. (2021) used SMAC on
the BBOB functions with this configuration space; the scripts are in
[`scripts/tuning`](https://github.com/IOHprofiler/ModularCMAES/tree/master/scripts/tuning).
