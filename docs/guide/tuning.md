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

`get_configspace(dim, add_popsize=True, add_sigma=True, add_learning_rates=True)` leaves out the
numeric parameters you do not want to tune. The defaults of the space are the defaults of modcma,
so `space.get_default_configuration()` is the standard CMA-ES. `settings_from_config` passes only
the values that differ from the defaults to `Settings`, so the defaults that depend on the
dimension (λ, the learning rates) stay correct; keyword arguments (`budget`, bounds, ...) are passed
on.

A complete example, a random search for the best configuration on a few functions, is
[tuning a configuration](../examples/tuning.md). The study of de Nobel et al. (2021) used SMAC on
the BBOB functions with this configuration space; the scripts are in
[`scripts/tuning`](https://github.com/IOHprofiler/ModularCMAES/tree/master/scripts/tuning).
