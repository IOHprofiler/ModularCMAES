# Benchmarking on BBOB

[IOHexperimenter](https://iohprofiler.github.io/IOHexperimenter/) provides the 24 noiseless BBOB
functions (and many other problems) with logging in the format of
[IOHanalyzer](https://iohanalyzer.liacs.nl). An `ioh` problem is a callable, so it can be passed to
`run` directly.

```bash
pip install ioh
```

```python
--8<-- "examples/bbob.py"
```

Output (d = 5, 24 functions × 5 instances, budget 10 000 × d):

```text
CMA-ES                 solved 61/120 (f - f_opt <= 1e-08), median 2177 evaluations when solved
BIPOP active CMA-ES    solved 91/120 (f - f_opt <= 1e-08), median 1897 evaluations when solved
```

Upload the `bbob-data` folder (zipped) to IOHanalyzer to compare the two configurations function by
function: convergence plots, ERT, ECDFs. For many runs, use [threads or
processes](../guide/reproducibility.md#threads); each optimizer is independent.
