# Coming from gradient-based optimization

CMA-ES needs only the values of the objective. When you also have its gradient (analytic,
automatic differentiation, an adjoint simulation), modcma can use it: **gradient injection**.
Every generation, one point along the negative gradient from the mean replaces one sample of the
population. CMA-ES stays a search with a population; the gradient point competes with the
samples, and moves the mean only when it ranks among the best.

```python
import numpy as np
import modcma

d = 10
weights = 10.0 ** (6.0 * np.arange(d) / (d - 1))  # an ellipsoid with condition 1e6


def ellipsoid(x):
    return float(weights @ x**2)


def gradient(x):
    return 2.0 * weights * x


_, _, with_gradient, _ = modcma.fmin(
    ellipsoid, np.full(d, 3.0), 2.0, 50_000, target=1e-8, seed=1, gradient=gradient
)
_, _, without, _ = modcma.fmin(ellipsoid, np.full(d, 3.0), 2.0, 50_000, target=1e-8, seed=1)
print(with_gradient, without)  # evaluations to reach 1e-8
```

`run(f, gradient=...)`, `step(f, gradient=...)` and `mutate(f, gradient=...)` take the same
argument. The gradient function is called once per generation, with the center of mass
(`es.center_of_mass`, a (d,) array), and returns d numbers. Its calls are not counted in the budget.

## With ask–tell

Between the `ask()` that starts a population and its `tell()`, `es.accepts_gradient` is true;
`es.inject_gradient(g)` then replaces a point of that population, and the next `ask()` returns it.
The other batches (a restart's center of mass, the points of a hill–valley test, the later batches
of sequential selection) take no gradient. The results are identical to `run(f, gradient=...)`
with the same seed, bit for bit:

```python
es = modcma.ModularCMAES(modcma.Settings(d, x0=np.full(d, 3.0), sigma0=2.0, budget=50_000,
                                         target=1e-8, seed=1))
while True:
    X = es.ask()
    if es.accepts_gradient:                      # a new population
        es.inject_gradient(gradient(es.center_of_mass))
        X = es.ask()                             # with the gradient point
    if es.tell([ellipsoid(x) for x in X]) and es.break_conditions():
        break

es_run = modcma.ModularCMAES(modcma.Settings(d, x0=np.full(d, 3.0), sigma0=2.0, budget=50_000,
                                             target=1e-8, seed=1))
es_run.run(ellipsoid, gradient=gradient)
assert es.p.stats.evaluations == es_run.p.stats.evaluations
```

`inject_gradient` raises `RuntimeError` when no gradient is accepted (also a second time for the
same population), and `ValueError` for a gradient that does not have d finite entries. A zero
gradient changes nothing.

## The gradient point

With the mean m, the step size σ and the matrix C of the samples (a sample is m + σ A z with
z ~ N(0, I) and A Aᵀ = C), the gradient g at the center of mass gives the point

x = m − σ c C g / √(gᵀ C g),

with c = E‖N(0, I)‖ ≈ √d (1 − 1/(4d) + 1/(21d²)), the length of a typical sample z. In the
coordinates of z it is the step z = −c Aᵀ g / ‖Aᵀ g‖: the steepest descent in the metric of the
distribution, as long as a typical sample. Once C has adapted to the inverse Hessian of a convex
quadratic, C g points from m to the optimum (the Newton direction).

This is Eq. (14) of Hansen (2011), [Injecting external solutions into
CMA-ES](https://arxiv.org/abs/1110.4181), which libcmaes implements (its gradient injection,
`set_gradient`, in `src/cmastrategy.cc`). Both start from the mean, as modcma does. Both use the
length √d, where modcma uses E‖N(0, I)‖, which is 0.25 % (d = 100) to 2.5 % (d = 10) shorter.
The point then enters the update as an injected solution: its step is clipped when it is longer
than √d + 2d/(d + 2) in the metric of the distribution (Hansen 2011, Eq. 3), which a gradient
point reaches only when a bound correction or a heavy-tailed sampler lengthens it.

## When it helps

Measured on three BBOB functions, with the default modules, x0 uniform in [−4, 4]^d, σ₀ = 2, target
f − f_opt ≤ 1e-8. The functions follow ioh's definitions (with T_osz, and T_asy for f12); the
optimum and the rotation are drawn by the experiment, since ioh does not expose its rotations.
Median evaluations to the target (all runs reached it), and the ratio to no injection:

| function | d | runs | no gradient | exact gradient | ratio |
|---|---|---|---|---|---|
| f2, separable ellipsoid | 10 | 15 | 6,041 | 5,381 | 0.89 |
| | 40 | 15 | 72,886 | 61,951 | 0.85 |
| | 100 | 10 | 424,992 | 361,982 | 0.85 |
| f10, rotated ellipsoid | 10 | 15 | 6,041 | 5,431 | 0.90 |
| | 40 | 15 | 71,956 | 62,026 | 0.86 |
| | 100 | 10 | 429,251 | 357,307 | 0.83 |
| f12, bent cigar | 10 | 15 | 11,311 | 10,420 | 0.92 |
| | 40 | 15 | 45,255 | 35,625 | 0.79 |
| | 100 | 15 | 125,801 | 67,320 | 0.54 |

Every difference is significant (two-sided Mann–Whitney U, p ≤ 0.0012) except f12 at d = 10
(p = 0.14). On the sphere the gain is larger: 1.5× fewer evaluations at d = 10 and 2.2× at
d = 40 (10 runs), close to the factor 2 that Hansen (2011) reports for injected solutions of very
good quality. On the ellipsoids most of the evaluations go into learning C, which the gradient
does not speed up much; on the bent cigar, whose one long axis C learns quickly, the gradient
saves more.

So the gradient is a hint, not a gradient method. On a smooth, unimodal function with a cheap
gradient, a quasi-Newton method is far faster: L-BFGS-B (scipy) reaches 1e-8 on f10 in 1,732
function-and-gradient calls at d = 10 and 18,658 at d = 40 (median of 15). Gradient injection is
for the problems where you use CMA-ES for its own reasons (several local optima, noise, steps in
the function, a gradient that is not reliable everywhere) and have a gradient as well.

## What a bad gradient costs

A gradient point that is bad ranks low, so it hardly moves the mean: a bad gradient costs the
sample it replaced. Hansen (2011): "injecting a single bad solution essentially corresponds to
decreasing the population size by one". Measured on f10 (the setup above; median evaluations,
ratio to no injection):

| gradient | d = 10 | d = 40 |
|---|---|---|
| none | 6,041 | 71,956 |
| exact | 5,431 (0.90) | 62,026 (0.86) |
| random (a standard normal vector at every call) | 6,281 (1.04) | 77,011 (1.07) |
| exact plus noise of the same norm (45° off) | 6,231 (1.03) | 77,566 (1.08) |
| negated (the ascent direction) | 6,571 (1.09) | 75,331 (1.05) |

Every run reached the target (15 per cell). The default population has λ = 10 points at d = 10
and 15 at d = 40, so one sample is 10 % and 6.7 % of a generation: a bad gradient costs about
that, and a direction 45° off is already no better than a random one. A gradient helps when it is
close to the true one.

With λ = 1 (the (1+1)-ES), and with λ = 2 and mirrored sampling, the gradient point (and its
mirror) is the whole population: no sample is left, and the matrix learns only the gradient
steps. This is allowed, but avoid it: on f10 with λ = 1, none of 15 runs reached 1e-8 at d = 10 or
40, not even with the exact gradient (median best f 1.5e5 and 3.9e5), while all 15 did without a
gradient (median 4,898 and 71,003 evaluations).

## How it combines with the modules

The samples are drawn as without a gradient; then the point replaces individual 0. Details are
on the [compatibility page](compatibility.md#gradient-injection):

* **Mirrored sampling**: individuals 0 and 1 are a pair, so both are replaced, by the gradient
  point and its mirror (the step +z). Pairwise selection then keeps the better of the two.
* **Bound correction**: the point is moved into the box as TPA's points are (`RESAMPLE`, which
  cannot sample it again, clips it); integer variables are rounded.
* **Sampled step sizes** (`ssa=SA`, `LPXNES`): the point gets the step size of the individual it
  replaces. Integer variables keep their lower bound on the step size.
* **Sequential selection**: individual 0 is in the first batch, so the gradient point is always
  evaluated.
* **TPA**: its two points are extra points of the generation, not samples; they stay.
* **Elitism**: the gradient point is an offspring like the others.
* **Restarts**: every population of every run gets a gradient point, at the mean of its run.
* **Repelling restarts**: a gradient point in a taboo region of the archive is not injected (the
  sample stays), as a sample there is rejected.

## Without an analytic gradient

modcma does not compute a gradient for you. libcmaes falls back to forward finite differences,
d + 1 evaluations per generation, which it counts in its budget. A finite-difference gradient
computed in your `gradient` function works, but its evaluations are not counted in `budget`.
