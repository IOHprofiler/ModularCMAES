# Step-size adaptation

The step size σ scales all mutations; its adaptation decides how fast the search converges and
whether it escapes from a slope. `ssa` chooses the method. The default, cumulative step-size
adaptation (CSA), compares the length of the evolution path (the recent movement of the mean) with
its expected length under random selection, and increases σ when the steps add up.

| option | idea | notes |
|---|---|---|
| `CSA` | cumulative step-size adaptation (Hansen & Ostermeier 2001) | the default |
| `TPA` | two-point adaptation: evaluate the mean moved forward and backward by half its last step (Hansen 2008) | evaluates 2 extra points per generation (`ask()` returns λ + 2; with `elitist`, none after a generation that kept every parent) |
| `TPA_INPLACE` | two-point adaptation within the population: offspring 1 and 2 along the last mean step, compared by their ranks (pycma; Hansen, Atamna & Auger 2014) | no extra points |
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

for ssa in ["CSA", "TPA", "TPA_INPLACE", "MSR", "PSR", "XNES"]:
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
**`modcma.options.StepSizeAdaptation`**: The adaptation of the global step size sigma (`Modules.ssa`; default CSA). With the matrix adaptation COVARIANCE, any method other than CSA switches it to COVARIANCE_NO_EIGV (no eigendecomposition), unless `Settings(always_compute_eigv=True)`. A population of one (lambda0 = 1, the (1+1)-ES) always uses SR, whatever ssa is given (and so COVARIANCE becomes COVARIANCE_NO_EIGV). The values of cs and damps given below are the defaults; `Settings.cs` and `Settings.damps` replace them. For a comparison of the methods see de Nobel et al. 2021 (doi:10.1145/3449726.3463167).

| value | |
|---|---|
| `CSA` | Cumulative step-size adaptation (Hansen & Ostermeier 2001), the default: sigma grows when the evolution path p_sigma is longer than expected under random selection and shrinks when it is shorter: sigma <- sigma exp(cs / damps (\|\|p_sigma\|\| / E\|\|z\|\| - 1)), the exponent clipped to [-1, 1] (as pycma). cs = (mueff + 2) / (d + mueff + 5) and damps = 1 + 2 max(0, sqrt((mueff - 1) / (d + 1)) - 1) + cs (Hansen 2016, Table 1, Eq. (55); #138). |
| `TPA` | Two-point step-size adaptation (Hansen 2008, "CMA-ES with Two-Point Step-Size Adaptation"): two extra points, m +- 0.5 sigma_old dm, half the last step of the mean with the step size that sampled it (Eqs. (8), (9)), are evaluated every generation (two more evaluations). r = 0.5 when the point ahead is the better one, -0.5 when the point behind is, 0 on a tie (as pycma; the paper counts a tie as a success); s <- (1 - cs) s + cs r and sigma <- sigma exp(s), cs = 0.3 (Eqs. (10) to (13)); damps is not used. With a bound correction both points move toward m, symmetrically, until both are in the box. With elitism, a generation whose selection kept every parent is a failure: the next generation evaluates no test points (the mean did not move; the budget still reserves them) and takes r = -0.5, as in the success rules of elitist ESs (#179; no TPA source covers elitism). Before 2.0 the points were m +- sigma dm, about one step of the mean (#167). `TPA_INPLACE` is the version within the population. |
| `MSR` | Median success rule (Ait ElHara et al. 2013, as pycma): K is the number of offspring better than the 0.3 lambda-th best of the previous generation (interpolated between two ranks, as pycma), z = 2 K / lambda - 1, s <- (1 - cs) s + cs z and sigma <- sigma exp(s / damps), with cs = 0.3 and damps = 2 - 2/d (at least 1). sigma grows while more than half of the offspring are better than that value (the framework paper compares the medians, #83). With pairwise selection only the ranked offspring count. |
| `XNES` | The step-size update of xNES. Without NATURAL_GRADIENT, xNES SA (Krause, Glasmachers & Igel (2017), Sec. 4.2): sigma <- sigma exp(cs / sqrt(d) sum_i w_i (\|\|z_i\|\|^2 - d)) over the mu best, with their positive weights and cs = mueff / (2 ln(max(2, d)) sqrt(d)) (their Table 3; #72). With NATURAL_GRADIENT, xNES's own update (Glasmachers et al. 2010): sigma <- sigma exp(cs / 2 tr(G) / d), G the natural gradient, with xNES's cs = eta_sigma = (9 + 3 ln d) / (5 d sqrt(d)). With elitism both read the offspring alone (#172). With `Modules.active` (and weights other than UTILITY) a generation with h_sigma = 0 has no negative weights in the matrix update of COVARIANCE, COVARIANCE_NO_EIGV, SEPARABLE, MATRIX and CHOLESKY (#175; modcma's rule: Hansen 2016 and pycma keep them). xNES SA with weights EQUAL fails on the ellipsoid (#180, see the compatibility page). |
| `MXNES` | A variant of the xNES update from the length of the weighted mean of the selected samples instead of their individual lengths: sigma <- sigma exp(cs / d (mueff \|\|sum_i w_i z_i\|\|^2 - d)), with cs = 1 (m-xNES, Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules", Sec. 4.3, Table 3). |
| `LPXNES` | An xNES-like update from the logarithms of the individual step sizes of the samples (lp-xNES, Krause, Glasmachers & Igel (2017), "Qualitative and quantitative assessment of step size adaptation rules", Sec. 4.4). Every individual samples its own step size, as with `Modules.sample_sigma` but with another tau: sigma_i = sigma exp(tau n_i), tau = ln(2) / (sqrt(d) ln(max(2, d))). Then sigma <- sigma^(1 - cs) exp(cs sum_i w_i ln(sigma_i)) over the mu best with their positive weights; cs = 9 mueff / (10 sqrt(d)) (their Table 3), capped at 1 (modcma's cap, #72). |
| `PSR` | Population success rule (Loshchilov 2014, LM-CMA-ES): the n best of the offspring and of the previous generation are ranked together, n the smaller number of ranked individuals of the two (equal values share their mean rank, #120); z = (sum of the ranks of the previous generation - sum of the ranks of the offspring) / n^2 - 0.25, the target success ratio; s <- (1 - cs) s + cs z and sigma <- sigma exp(s / damps), with cs = 0.3 and damps = 1 (Loshchilov 2014; #83). With pairwise selection only the ranked offspring count. |
| `SR` | Success rule of the (1+1)-ES (Igel et al. 2006): a generation is a success when its best point beats the best of the run so far; the success ratio p is averaged with rate 1/12, and sigma <- sigma exp((p - 2/11) / ((1 - 2/11) damps)), damps = 1 + d/2: sigma grows when p is above the target 2/11. Used automatically when lambda0 = 1. Designed for elitist selection, where the best of the run is the parent; without elitism the best of the run plays the parent's role (the framework paper compares with the previous generation's best, which made sigma diverge, #83). |
| `SA` | Self-adaptation: every individual samples its own step size sigma_i = sigma exp(tau n_i), with CMSA's tau = 1 / sqrt(2 d) (Spettel, Beyer & Hellwig 2018, Table I; #71); sigma is multiplied by the weighted geometric mean of the step sizes of the mu selected individuals divided by the geometric mean of those of the population (the CMSA-ES operator of Beyer & Sendhoff 2008, weighted, with the correction for elitism of Ahrari et al. 2017, as the framework paper describes it). Meant for CMSA; makes the default mu lambda / 4. |
| `TPA_INPLACE` | Two-point step-size adaptation within the population, as pycma 4.6.0's CMAAdaptSigmaTPA (after Hansen, Atamna & Auger 2014, Eq. (6) and Algorithm 3). From the second generation of a run on, offspring 1 and 2 are a mirrored pair along the last mean shift, z_1 = \|\|N(0, I)\|\| A^-1 dm / \|\|A^-1 dm\|\| and z_2 = -z_1, so no extra evaluations. With z = (rank of offspring 2 - rank of offspring 1) / (n - 1) among the n offspring, positive when offspring 1 is the better one, s <- (1 - cs) s + cs sign(z) \|z\|^0.5 and sigma <- sigma exp(s / damps); cs = 0.3, damps = 0.7 + 2 ln(d) + 2 ln(max(1, lambda - d)). pycma departs from the paper in the \|z\|^0.5 (the paper uses z), the damping (the paper: sqrt(d)), and the update in the first generation of a run, from two ordinary samples (the paper updates from the second generation on); modcma follows pycma. A generation with fewer than two offspring is not updated. Works with mirrored sampling (the test points are the first pair), pairwise selection (the ranks are of all offspring), sequential selection (which cannot stop before both are evaluated; n is then the evaluated offspring) and elitism (the ranks are of the offspring; after a generation whose selection kept every parent, the generation samples ordinary offspring and z = -1, #179); with orthogonal sampling the test points are not orthogonal to the other samples. #173. |
<!-- /options -->
