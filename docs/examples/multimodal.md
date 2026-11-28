# Multimodal problems

Restarts are the most effective module on multimodal functions. On Rastrigin a single CMA-ES run
converges to one of the many local optima; restarting with a growing population (IPOP, BIPOP)
reaches the global optimum in nearly every run (below: 10 of 10 and 9 of 10). Repelling restarts
solve a different problem: finding *several* optima, by keeping new runs away from the optima found
before.

```python
--8<-- "examples/multimodal.py"
```

Output:

```text
Rastrigin, d = 10, 10 runs each (solved: f <= 1e-8)
  CMA-ES         solved  0/10
  IPOP-CMA-ES    solved 10/10, median    85866 evaluations
  BIPOP-CMA-ES   solved  9/10, median    98762 evaluations

Himmelblau, d = 2, restarts from uniform random points, 20 runs each
  restarts             distinct optima found: mean 3.35 of 4, all four in 9/20 runs
  repelling restarts   distinct optima found: mean 3.45 of 4, all four in 10/20 runs
```

See [restarts](../guide/restarts.md) for the restart criteria, where restarts begin, and the
options of repelling restarts.
