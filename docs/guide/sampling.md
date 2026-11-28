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
  [study of these distributions](https://dl.acm.org/doi/10.1145/3712256.3726479). `NONE` passes
  the raw base samples on, for testing: their mean is 0.5, so they fail as mutations.

## Mirrored and orthogonal sampling

* `mirrored=MIRRORED`: the individuals of a generation come in pairs z, −z (Brockhoff et al.
  2010); with an odd λ the last one is unpaired. `PAIRWISE` also ranks only the better point of
  each pair (Auger et al. 2011), so mirrored points do not cancel each other in the recombination;
  it makes λ even.
* `orthogonal=True`: the samples of a generation are orthonormalized in blocks of at most d and
  given the lengths of the original samples (Wang et al. 2019); with mirroring, the first of each
  pair ([deep dive](../deep-dives/selection-sampling-restarts.md#orthogonal-sampling)).

## Step sizes per point and threshold convergence

* `sample_sigma=True`: each point gets its own step size, σ·exp(βn) with n standard normal
  (log-normal around σ), the same in every coordinate, so the point stays along its y; always on
  with the step-size adaptations `SA` (β = 1/√(2d)) and `LPXNES` (β = ln 2/(√d ln d)); otherwise
  β = 1/√(2µ_eff).
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
| `SOBOL` | Points of the Sobol sequence (Sobol' 1967). A cache of points (see `constants.cache_max_doubles`) is computed when the sampler is created and drawn in a random order, which repeats after all of them: N = max(128, the largest power of 2 <= 2e6 / d) points with the default constants; at d = 1000, N = 1024: after 42 generations of the default lambda = 24. Only for d <= 1111, the size of the table of direction numbers: with a larger d, creating the sampler ends the process. |
| `HALTON` | Points of the Halton sequence (Halton 1960), scrambled with random digit permutations. |
<!-- /options -->

<!-- options: Mirror -->
**`modcma.options.Mirror`**: Mirrored sampling (`Modules.mirrored`; default NONE). With mirroring, individuals 2i and 2i + 1 of every generation are a pair z, -z, which spreads the samples more evenly around the mean: ceil(lambda / 2) independent samples per generation. With an odd lambda the last individual is unpaired, so with lambda = 1 mirroring has no effect. A pair that bound correction (RESAMPLE) or repelling rejects is sampled again as a pair.

| value | |
|---|---|
| `NONE` | Independent samples, the default. |
| `MIRRORED` | Mirrored sampling (Brockhoff et al. 2010): the samples come in pairs z, -z. All the points take part in the selection. |
| `PAIRWISE` | Mirrored sampling with pairwise selection (Auger et al. 2011): the worse point of each mirrored pair is ranked after all the better ones (and after the parents of elitism), so that with mu <= lambda / 2 only the better one can be selected and the two do not cancel out in the recombination. The worse points keep their objective values and get no negative weights in the active update (as pycma). Makes lambda even (the default mu is half the even lambda); with lambda0 = 1 (the (1+1)-ES) Settings turn it into NONE. |
<!-- /options -->
