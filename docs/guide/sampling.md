# Sampling

Each generation draws λ vectors z, transforms them to the mutations y = A z (the
[matrix adaptation](matrix-adaptation.md)) and the points x = m + σ y. Several modules change how the
z are drawn.

## Base sampler and sample transformation

The base sampler produces points in [0, 1)^d; the sample transformation maps them to the
distribution of z, by default the standard normal.

* `sampler`: `UNIFORM` (pseudo-random, the default), or the low-discrepancy sequences `SOBOL`
  and `HALTON`, which cover the space more evenly (quasi-random sampling).
* `sample_transformation`: `GAUSSIAN` (the default), or heavier- or lighter-tailed
  distributions: `CAUCHY`, `LAPLACE`, `LOGISTIC`, `DOUBLE_WEIBULL`, `SCALED_UNIFORM`; see the
  [study of these distributions](https://dl.acm.org/doi/10.1145/3712256.3726479).

## Mirrored and orthogonal sampling

* `mirrored=MIRRORED`: half of the z are the negatives of the other half, which removes a bias in
  the recombination (Brockhoff et al. 2010). `PAIRWISE` also selects only the better point of each
  pair (Auger et al. 2011), so mirrored points do not cancel each other; it makes λ even.
* `orthogonal=True`: the samples are orthonormalized in groups of at most d (a Householder QR
  decomposition) and given the lengths of the original samples (Wang et al. 2014); with λ < d a
  group spans generations ([deep dive](../deep-dives/selection-sampling-restarts.md#orthogonal-sampling)).

## Step sizes per point and threshold convergence

* `sample_sigma=True`: each point gets its own step size per coordinate, σ·exp(βn) with n
  standard normal (log-normal around σ); always on with the step-size adaptations `SA` and
  `LPXNES`.
* `threshold_convergence=True`: the mutations must be at least a minimum length, which decreases
  over the run; it prolongs the exploration of the beginning (Piad-Morffis et al. 2015).

## The random generator

`Settings(fast_sampling=True)` replaces the generator and the normal transformation by faster ones
for the default `UNIFORM` + `GAUSSIAN` sampling; see [performance](performance.md#fast-sampling).

```python
import numpy as np
import modcma

def rosenbrock(X):
    return np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)

configs = {
    "default": {},
    "Sobol": {"sampler": "SOBOL"},
    "mirrored, pairwise": {"mirrored": "PAIRWISE"},
    "orthogonal": {"orthogonal": True},
}
for name, modules in configs.items():
    evaluations = []
    for seed in range(5):
        settings = modcma.settings_from_dict(5, budget=20_000, target=1e-8, seed=seed, **modules)
        es = modcma.ModularCMAES(settings)
        es.run(rosenbrock, vectorized=True)
        evaluations.append(es.p.stats.evaluations)
    print(f"{name:20s} median evaluations {np.median(evaluations):7.0f}")
```

## The options

<!-- options: BaseSampler -->
**`modcma.options.BaseSampler`**: The source of the uniform numbers in (0, 1)^d that become the samples (`Modules.sampler`; default UNIFORM). The `SampleTranformerType` (`Modules.sample_transformation`) maps them to the distribution of the mutations, by default the standard normal one. SOBOL and HALTON are quasi-random (low-discrepancy) sequences: their points cover the space more evenly than random ones (de Nobel et al. 2024, "Sampling in CMA-ES: low numbers of low discrepancy points").

| value | |
|---|---|
| `UNIFORM` | Pseudo-random uniform numbers (mt19937, the optimizer's own generator), the default. With the GAUSSIAN transformation: ordinary random normal samples, which `Settings.fast_sampling` replaces by a faster generator. |
| `SOBOL` | Points of the Sobol sequence (Sobol' 1967). A cache of points (see `constants.cache_max_doubles`) is computed when the sampler is created and drawn in a random order. |
| `HALTON` | Points of the Halton sequence (Halton 1960), scrambled with random digit permutations. |
<!-- /options -->

<!-- options: Mirror -->
**`modcma.options.Mirror`**: Mirrored sampling (`Modules.mirrored`; default NONE). With mirroring, every second mutation vector is the negation of the one before it, which spreads the samples more evenly around the mean.

| value | |
|---|---|
| `NONE` | Independent samples, the default. |
| `MIRRORED` | Mirrored sampling (Brockhoff et al. 2010): the samples come in pairs z, -z. |
| `PAIRWISE` | Mirrored sampling with pairwise selection (Auger et al. 2011): of each mirrored pair only the better one can be selected, so that the two do not cancel out in the recombination. Makes lambda even; with lambda0 = 1 (the (1+1)-ES) Settings turn it into NONE. |
<!-- /options -->
