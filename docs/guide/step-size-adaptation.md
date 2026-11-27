# Step-size adaptation

The step size σ scales all mutations; its adaptation decides how fast the search converges and
whether it escapes from a slope. `ssa` chooses the method. The default, cumulative step-size
adaptation (CSA), compares the length of the evolution path (the recent movement of the mean) with
its expected length under random selection, and increases σ when the steps add up.

| option | idea | notes |
|---|---|---|
| `CSA` | cumulative step-size adaptation (Hansen & Ostermeier 2001) | the default |
| `TPA` | two-point adaptation: evaluate the mean moved forward and backward along its last step (Hansen 2008) | evaluates 2 extra points per generation (`ask()` returns λ + 2) |
| `MSR` | median success rule: does the median point beat the 0.3λ-th best of the previous generation? (Ait ElHara et al. 2013) | |
| `PSR` | population success rule: rank the current and the previous generation together (Loshchilov 2014) | |
| `XNES`, `MXNES`, `LPXNES` | natural-gradient step-size updates in the style of xNES (Glasmachers et al. 2010) and variants | |
| `SR` | the 1/5th success rule, for a (1+1)-ES | chosen automatically with `lambda0=1` |
| `SA` | self-adaptation: each point has its own σ, selected along with it | with `matrix_adaptation=CMSA`: CMSA-ES |

All methods other than CSA work without the eigendecomposition of C, so with the default
`COVARIANCE` they switch it to `COVARIANCE_NO_EIGV` ([adjusted
combinations](modules.md#combinations-that-are-adjusted)).

```python
import numpy as np
import modcma

def sphere(X):
    return np.sum(X**2, axis=1)

for ssa in ["CSA", "TPA", "MSR", "PSR", "XNES"]:
    es = modcma.ModularCMAES(modcma.settings_from_dict(
        10, ssa=ssa, x0=np.full(10, 3.0), budget=10_000, target=1e-8, seed=1
    ))
    es.run(sphere, vectorized=True)
    print(f"{ssa:5s} evaluations = {es.p.stats.evaluations:6d}  final sigma = {es.p.mutation.sigma:.2e}")
```

The current step size is `es.p.mutation.sigma`; the method's state and parameters are on
`es.p.mutation` ([reference](../reference/mutation.md)). The damping and learning rate (`damps`,
`cs`) are [settings](settings.md).

## The options

<!-- options: StepSizeAdaptation -->
**`modcma.options.StepSizeAdaptation`**: The adaptation of the global step size sigma (`Modules.ssa`; default CSA). With the matrix adaptation COVARIANCE, any method other than CSA switches it to COVARIANCE_NO_EIGV (no eigendecomposition), unless `Settings(always_compute_eigv=True)`. A population of one (lambda = 1, the (1+1)-ES) always uses SR (and so COVARIANCE_NO_EIGV, whatever ssa is given). For a comparison of the methods see de Nobel et al. 2021 (doi:10.1145/3449726.3463167).

| value | |
|---|---|
| `CSA` | Cumulative step-size adaptation (Hansen & Ostermeier 2001), the default: sigma grows when the evolution path p_sigma is longer than expected under random selection and shrinks when it is shorter. |
| `TPA` | Two-point step-size adaptation (Hansen 2008): two extra points m +- sigma * dm along the last step dm of the mean are evaluated every generation (two more evaluations); sigma grows when the point ahead is the better one. The bound correction moves the two points into the box (RESAMPLE moves both toward m, symmetrically). |
| `MSR` | Median success rule (Ait ElHara et al. 2013): counts the individuals that are better than the 0.3 lambda-th best of the previous generation; sigma grows when more than half are. |
| `XNES` | The step-size update of xNES (Glasmachers et al. 2010), from the weighted squared lengths of the samples; with NATURAL_GRADIENT, from its natural gradient. |
| `MXNES` | A variant of the xNES update from the length of the weighted mean of the selected samples, mueff \|\|sum w_i z_i\|\|^2 - d, instead of their individual lengths (m-xNES, Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules"). |
| `LPXNES` | An xNES-like update from the logarithms of the individual step sizes of the samples, weighted by rank; samples one step size per individual (as `Modules.sample_sigma`) (lp-xNES, Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules"). |
| `PSR` | Population success rule (Loshchilov 2014, LM-CMA-ES): ranks the current and the previous generation together and adapts sigma towards a target success ratio of 0.25. |
| `SR` | Success rule of the (1+1)-ES: sigma grows when the success ratio is above the target 2/11 and shrinks when it is below (Igel et al. 2006). Used automatically when lambda = 1. Designed for elitist selection; without elitism a success is measured against the best of the run, which plays the parent's role. |
| `SA` | Self-adaptation: every individual gets its own step size, sampled log-normally around sigma (learning rate 1 / sqrt(2 d), CMSA's), and sigma moves to the weighted geometric mean of the step sizes of the selected individuals, divided by that of the population (the CMSA-ES operator of Beyer & Sendhoff 2008, weighted, with the correction for elitism of Ahrari et al. 2017). Meant for CMSA; makes the default mu lambda / 4. |
<!-- /options -->
