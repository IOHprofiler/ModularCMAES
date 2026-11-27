# Warm starting from a related task

When a related problem has been solved before, its solutions can start the new run:
`modcma.warm_start` estimates the initial mean, step size and covariance matrix from them
(WS-CMA-ES, Nomura et al. 2021), and `Settings(..., **warm_start(solutions))` starts the run from
that distribution ([settings](../guide/settings.md#the-initial-covariance-matrix-and-warm-starting)).
The tasks below are the paper's synthetic pair, a rotated ellipsoid whose optimum moves, in
d = 10; the cold start is the paper's N(0.5, 0.2²) in every coordinate. The
[second part](#the-papers-synthetic-experiment) repeats the paper's own experiment, in d = 2.

```python
--8<-- "examples/warm_start.py"
```

Output (`warm_start` with its defaults, γ = α = 0.1; the target runs use the seeds 0 to 19, the
source solutions the seeds 1000 to 1019):

```text
Rotated ellipsoid, d = 10: median evaluations to f <= 1e-8, 20 seeds
source solutions target            cold  warm  warm, no C0
random search    shifted by 0.02   3461  2841         3481
random search    shifted by 0.1    3546  2931         3496
random search    0.02, rotated     3451  3526         3461
CMA-ES run       shifted by 0.02   3461  3286         3296
CMA-ES run       shifted by 0.1    3546  3411         3401
CMA-ES run       0.02, rotated     3451  3436         3431
```

* **The shape transfers.** With the solutions of a random search, the best 10% lie in a level set
  of the source task, an ellipsoid with the shape of its inverse Hessian. `C0` starts the run with
  part of that shape (its condition is 9 to 11 over the 20 seeds, against 10⁴ of the Hessian: the
  box [0, 1]^d cuts the level set, whose spread has a condition of 39 for the first seed, and
  α² I is added),
  and the warm start needs 18% fewer evaluations. Without `C0`, the warm `x0` and `sigma0` alone
  save nothing: most of a run is spent learning the shape, not reaching the region.
* **The source matters.** The best 10% of a CMA-ES run that converges are close together, and α
  dominates their spread: `C0` is nearly the identity (median condition 1.07), `sigma0` is about α
  (median 0.101), and only the location transfers (5% fewer evaluations).
* **Related tasks only.** When the target has another rotation, the shape of the source is wrong
  for it and the warm start needs 2% more evaluations than the cold start. The paper measures the
  similarity of two tasks (its γ-similarity); the warm start helps when it is positive.

## The paper's synthetic experiment

The paper measures the warm start on two 2-D problems (Nomura et al. 2021, Section 4.1, Appendices
A and B, Table 1): the sphere (x₁ − b)² + (x₂ − b)², and the rotated ellipsoid f_ell(R x) with
f_ell(y) = (y₁ − b)² + 25 (y₂ − b)² and R the rotation by π/6. The target task has b = 0.6, the
source tasks b = 0.4 to 0.8. λ = 8, α = γ = 0.1, the cold start N(0.5, 0.2²), and the measure is
the best value of the first 50 evaluations, over 20 runs. The paper does not say how many source
solutions it used here; the script assumes a random search of 100 points in [0, 1]², the number of
its hyperparameter experiments. Every run uses the active update (`active=True`): the paper's
CMA-ES is pycma with its default options (Appendix A), which include the active update
(`CMA_active=True` in pycma 4.6.0), while modcma's default does not.

```python
--8<-- "examples/warm_start_paper.py"
```

Output (run i = 0, ..., 19 uses the seed 1000 + i for the source points and i + 1 for the
optimizer):

```text
Best f of the first 50 evaluations, mean ± standard error of 20 runs
                                cold       b = 0.4       b = 0.5       b = 0.6       b = 0.7       b = 0.8
sphere (x 1e3)
  modcma                0.42 ± 0.077   0.93 ± 0.19   0.45 ± 0.15    0.3 ± 0.17   0.48 ± 0.15    1.3 ± 0.39
  paper, Table 1          0.43 ± 0.1     1.3 ± 0.3   0.26 ± 0.06  0.073 ± 0.02   0.27 ± 0.07    0.82 ± 0.2
rotated ellipsoid (x 1e2)
  modcma                  1.3 ± 0.47    0.5 ± 0.11   0.3 ± 0.074 0.077 ± 0.021  0.19 ± 0.067   0.81 ± 0.21
  paper, Table 1          0.25 ± 0.1   0.26 ± 0.06   0.21 ± 0.06   0.14 ± 0.05   0.15 ± 0.06    0.38 ± 0.2
```

* **The pattern of the paper holds**, within the noise of 20 runs: the warm start is best with the
  source task equal to the target (b = 0.6) and gets worse as b moves away. On the sphere the
  sources b = 0.4 and 0.8 are worse than the cold start, as in the paper.
* **Means of 20 runs are noisy here.** The best values have a heavy tail: over 500 runs (seeds 1 to
  500, source seeds 1000 to 1499, b = 0.6) the means are 1.9 to 2.6 times the medians, and the
  sphere's warm start with b = 0.6 has the median 0.063 and the mean 0.14 (×1e3), against the mean
  0.3 of the 20 runs above. The same holds for the paper's 20-run means.
* **The cold start on the ellipsoid is not reproduced**: 1.3 ± 0.47 against the paper's
  0.25 ± 0.1 (over the 500 runs: mean 1.4, median 0.69). The cold start does not use
  `warm_start`, and `cmaes` 0.13.1, which always uses the active update, gives 1.78 ± 0.55 with the
  same protocol and seeds; the difference most likely comes from a detail of the protocol that the
  paper does not state.
