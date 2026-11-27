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

* `lambda0` and `sigma0` are absolute values, with the defaults of `Settings(dim)`. `lambda0` is an
  integer in [1, 50 d] on a log scale.
* `cs_factor`, `cc_factor`, `cmu_factor`, `c1_factor`, `damps_factor` and `mu0_factor` are factors
  of the default of the chosen modules, on a log scale with the default 1: [0.1, 10] for the
  learning rates, [0.1, 2] for µ. The defaults differ per module (for example `cs` of `XNES`,
  `SR` and `CSA`, or µ = λ/4 for `SA` and `CMSA`), so a factor means the same for every
  combination, and the default configuration is the same standard CMA-ES whichever modules are
  chosen. `settings_from_config` gives the factor 1 no value: `Settings` computes the default itself,
  from the population size of the configuration. The factors are relative to the defaults without the other
  factors (`cc_factor` doubles the `cc` that `Settings` gives, also when `cs_factor` changes it).
  µ is rounded, at least 1 and at most λ.

`settings_from_config` passes every value of the configuration to `Settings`, also one that equals
the default. A dict may carry absolute values under the `Settings` names (`cs`, `mu0`, ...);
they are passed as they are.

## Conditions

A hyperparameter without an effect is inactive: it is absent from a sampled configuration, so a
tuner does not spend evaluations on it.

| Hyperparameter | Active when |
|---|---|
| `active` | `matrix_adaptation` is not `NONE` |
| `center_placement`, `repelling_type` | `restart_strategy` is `RESTART`, `IPOP` or `BIPOP` |
| `repelling_metric` | `repelling_type` is not `NONE` |
| `cs_factor` | `ssa` is not `SR` or `SA`, or `matrix_adaptation` is `MATRIX` or `NATURAL_GRADIENT` |
| `damps_factor` | `ssa` is `CSA`, `MSR`, `PSR` or `SR` |
| `cc_factor` | `matrix_adaptation` is not `NONE`, `MATRIX` or `CMSA` |
| `cmu_factor`, `c1_factor` | `matrix_adaptation` is not `NONE`, `CMSA` or `NATURAL_GRADIENT` |
| `mu0_factor` | `lambda0` is above 1 |

The effects of the learning rates were found by running every pair of step-size rule and matrix
adaptation with the parameter changed (d = 5, one seed, the weights `DEFAULT`, `EQUAL` and
`UTILITY`, with and without `active`). A parameter can matter in a rare combination that the table
leaves out: `cmu` and `c1` with `CMSA` or `NATURAL_GRADIENT` when `active` is combined with
`EQUAL` weights, and `cs` with `SA` and `COVARIANCE` with `UTILITY` weights. The thresholds of
threshold convergence, `ipop_factor` and the other numeric parameters of modules are not
hyperparameters.

## Configurations left out

* `sample_transformation=NONE` is not in the space. It passes the raw uniform samples (mean 0.5)
  as mutations, which fails everywhere ([compatibility](compatibility.md)). `Settings` accepts it,
  and `get_all_module_options` lists it.
* `Settings` replaces `matrix_adaptation=COVARIANCE` by `COVARIANCE_NO_EIGV` for every `ssa` other
  than `CSA`, so the two are the same configuration there. The space forbids `COVARIANCE` with
  those; `get_configspace(remove_duplicates=False)` keeps it.
* Other configurations that behave alike are not removed, for example every `active` and
  `elitist` combination where one makes the other void.

The modules are now more than in the framework paper, so its count of variants (14,859,936)
does not describe this space: with the conditions above the module options alone give 736,985,088
configurations (without `sample_transformation=NONE`, with `COVARIANCE` only for `CSA`).

A complete example, a random search for the best configuration on a few functions, is
[tuning a configuration](../examples/tuning.md). The study of de Nobel et al. (2021) used SMAC on
the BBOB functions with this configuration space; the scripts are in
[`scripts/tuning`](https://github.com/IOHprofiler/ModularCMAES/tree/master/scripts/tuning).

The scripts in `scripts/tuning` build on the space of modcma 1.x (the absolute names `cs`, `mu0`,
...); with 2.0 they need the names above.
