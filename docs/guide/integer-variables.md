# Integer variables

Some coordinates can be integers: list their indices in `Settings(integer_variables=...)`. The search
itself stays continuous; the integer coordinates are rounded to the nearest integer before the
point is evaluated, and their sampled step size has a lower bound, so the search does not get stuck
on one integer value (inspired by
[Marty et al. 2024](http://www.cmap.polytechnique.fr/~nikolaus.hansen/marty2024lb+ic-cma.pdf)):
per coordinate the step σ·sᵢ is at least min(0.2, µ_eff / d) / √Cᵢᵢ.

```python
import numpy as np
import modcma

def mixed(x):
    # x[0], x[1] continuous; x[2], x[3] integers with the optimum at 3 and -2
    return float(x[0] ** 2 + x[1] ** 2 + (x[2] - 3) ** 2 + (x[3] + 2) ** 2)

settings = modcma.Settings(4, integer_variables=[2, 3], budget=4_000, seed=1)
es = modcma.ModularCMAES(settings)
es.run(mixed)
print(es.p.stats.global_best.x)  # x[2] == 3.0, x[3] == -2.0
```

The objective receives the rounded point, and `global_best.x` is the rounded point too. The
same option works with `fmin` and `settings_from_dict`:

```python
xopt, fopt, evaluations, es = modcma.fmin(
    mixed, x0=np.zeros(4), sigma0=1.0, budget=4_000, integer_variables=[2, 3], seed=1
)
print(xopt, fopt)
```

For a problem with only integer variables, use `integer_variables=list(range(d))`. Integer
variables combine with every module. A complete example is
[mixed-integer problems](../examples/mixed-integer.md).

## Integer variables and bounds

The bound correction applies to the continuous point; the rounding comes after it. An integer
coordinate inside `[lb, ub]` is rounded to the nearest integer in [⌈lb⌉, ⌊ub⌋], so it stays in
the box also when a bound is not an integer: with lb = 0.3, the value 0.4 becomes 1, not 0. A
coordinate outside the box (only with `bound_correction="NONE"`, after the last try of
`RESAMPLE`, or for the two test points of TPA, which are not corrected) is rounded to the nearest
integer. With integer bounds this is plain rounding.

An integer variable needs an integer in `[lb, ub]`: `Settings` raise a `ValueError` when
[⌈lb⌉, ⌊ub⌋] is empty, as for lb = 0.6, ub = 0.9.

```python
settings = modcma.settings_from_dict(
    2, lb=[0.3, -1.0], ub=[2.7, 1.0], integer_variables=[0], bound_correction="SATURATE",
    budget=2_000, seed=1,
)
es = modcma.ModularCMAES(settings)
es.run(lambda x: float(x[0] ** 2 + x[1] ** 2))
print(es.p.stats.global_best.x)  # x[0] == 1.0: the nearest integer in [0.3, 2.7] to 0
```
