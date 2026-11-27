# Warm starting from a related task

When a related problem has been solved before, its solutions can start the new run:
`modcma.warm_start` estimates the initial mean, step size and covariance matrix from them
(WS-CMA-ES, Nomura et al. 2021), and `Settings(..., **warm_start(solutions))` starts the run from
that distribution ([settings](../guide/settings.md#the-initial-covariance-matrix-and-warm-starting)).
The tasks below are the paper's synthetic pair, a rotated ellipsoid whose optimum moves, in
d = 10; the cold start is the paper's N(0.5, 0.2²) in every coordinate.

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
