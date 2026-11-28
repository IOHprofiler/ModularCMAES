# Tuning a configuration

The modules and the population size form a configuration space
([tuning](../guide/tuning.md)). This example searches it at random for the configuration with the
best results on three functions and compares it with the default.

```python
--8<-- "examples/tuning.py"
```

A random search of 60 configurations improved the mean log10 precision (each value clipped at −8,
the target) from −5.13 (the default) to −7.65 here, with ConfigSpace 1.2.0: the sampled
configurations depend on its version. On real problem classes, use an algorithm configurator
(SMAC, irace) with many more evaluations, separate training and test problems, and several seeds
per configuration: modules that help on the training functions may not help on others.
