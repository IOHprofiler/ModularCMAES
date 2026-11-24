# Parallel evaluation

When one evaluation takes long (a simulation, a model fit), evaluate each population in parallel.
With ask–tell the optimizer does not call the objective: you get the points, evaluate them however
you like, and hand back the values.

```python
--8<-- "examples/parallel.py"
```

* Use processes for objectives written in Python (the GIL), threads for objectives that release it
  (numpy-heavy code, external programs, I/O) or on free-threaded Python 3.14t.
* The batch size `k` of `ask()` is usually the population size λ; set `lambda0` to a multiple of the
  number of workers to keep them busy. Some modules ask for other batch sizes:
  [ask–tell](../getting-started/ask-tell.md#what-ask-returns).
* To run many *independent* optimizations in parallel instead, give each its own seed and run
  them in a pool: [reproducibility and threads](../guide/reproducibility.md#threads).
