#include "common.hpp"

#include <nanobind/ndarray.h>

namespace modcma_python
{
namespace
{
//! The compiler that built the extension, e.g. "gcc 13.3.0" or "appleclang 21.0.0"
std::string compiler_version()
{
#if defined(__clang__)
#if defined(__apple_build_version__)
    return "appleclang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) +
           "." + std::to_string(__clang_patchlevel__);
#else
    return "clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
           std::to_string(__clang_patchlevel__);
#endif
#elif defined(__GNUC__)
    return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

struct constants_w
{
};

//! The LAPACK helpers take square matrices: ValueError otherwise (#113)
void require_square(const char *name, const Matrix &M)
{
    if (M.rows() != M.cols())
        throw nb::value_error((std::string(name) + ": the matrix must be square, got " +
                               std::to_string(M.rows()) + " x " + std::to_string(M.cols()))
                                  .c_str());
}
} // namespace

void define_constants(nb::module_ &m)
{
    nb::class_<constants_w>(
        m,
        "constants",
        "Global switches of the library, as static properties (`constants.eigen_solver = 0`).\n\n"
        "They are process-wide: a change applies to every optimizer, also to ones that exist "
        "already, except for the cache settings, which are read when a sampler is created. Set "
        "them before creating optimizers, and not while optimizers run in other threads. They "
        "are not part of the saved state of an optimizer (pickle): a loaded run continues with "
        "the values of the process that loads it.")
        .def_prop_rw_static(
            "cache_max_doubles",
            [](nb::handle) { return constants::cache_max_doubles; },
            [](nb::handle, size_t a) { constants::cache_max_doubles = a; },
            "The number of floats a sample cache may hold (default 2000000).\n\n"
            "A cache (the Sobol sampler, and `cache_samples`) holds max(cache_min_samples, the "
            "largest power of 2 <= cache_max_doubles / d) points of dimension d. Read when a "
            "sampler is created.")
        .def_prop_rw_static(
            "cache_min_samples",
            [](nb::handle) { return constants::cache_min_samples; },
            [](nb::handle, size_t a) { constants::cache_min_samples = a; },
            "The least number of points a sample cache holds (default 128); see "
            "`cache_max_doubles`.")
        .def_prop_rw_static(
            "cache_samples",
            [](nb::handle) { return constants::cache_samples; },
            [](nb::handle, nb::handle a) { constants::cache_samples = to_bool(a); },
            "Whether new optimizers cache their samples (default False).\n\n"
            "When True, the sampler of an optimizer created afterwards is wrapped in a "
            "`sampling.CachedSampler`: the first n samples (n from `cache_max_doubles` and "
            "`cache_min_samples`) are drawn and then repeated in the same cyclic order. Faster "
            "sampling, at the cost of a finite set of samples. Not applied to the SOBOL sampler, "
            "which caches its points itself.")
        .def_prop_rw_static(
            "clip_sigma",
            [](nb::handle) { return constants::clip_sigma; },
            [](nb::handle, nb::handle a) { constants::clip_sigma = to_bool(a); },
            "Whether the step size is clipped after every adaptation (default False).\n\n"
            "When True, sigma is kept in [restart.MinSigma.tolerance, "
            "restart.MaxSigma.tolerance] (default [1e-20, 1e4]), as step sizes (the criterion "
            "MaxSigma reads its tolerance as a growth factor).")
        .def_prop_rw_static(
            "use_box_muller",
            [](nb::handle) { return constants::use_box_muller; },
            [](nb::handle, nb::handle a) { constants::use_box_muller = to_bool(a); },
            "Whether the GAUSSIAN sample transformation uses the Box-Muller transform (default "
            "False).\n\n"
            "False: every uniform number is mapped to a normal one with the inverse normal CDF. "
            "True: pairs of uniform numbers are mapped with Box-Muller, a different "
            "random stream. No effect with `Settings.fast_sampling`, whose sampler (UNIFORM with "
            "GAUSSIAN) draws normal numbers with a ziggurat.")
        .def_prop_rw_static(
            "eigen_solver",
            [](nb::handle) { return constants::eigen_solver; },
            [](nb::handle, int a) {
                if (a < -1 || a > 2)
                    throw nb::value_error("eigen_solver: -1 (automatic), 0 (Eigen), 1 (LAPACK "
                                          "dsyevd) or 2 (LAPACK dsyevr)");
                if (a > 0 && !lapack::available())
                    throw nb::value_error("eigen_solver: no LAPACK available (not linked, and "
                                          "scipy's was not found)");
                constants::eigen_solver = a;
            },
            "The eigendecomposition of the covariance matrix (COVARIANCE) (default -1).\n\n"
            "Also used for `Adaptation.metric()` of the matrix adaptations that compute it when "
            "asked (all but COVARIANCE, SEPARABLE and NONE), and for the square root of "
            "`Settings.C0` of MATRIX and NATURAL_GRADIENT. "
            "-1: automatic, LAPACK dsyevd from `lapack_min_dim` on when LAPACK is available, "
            "Eigen below; 0: Eigen's SelfAdjointEigenSolver; 1: LAPACK dsyevd (divide and "
            "conquer); 2: LAPACK dsyevr (MRRR). 1 and 2 raise ValueError without LAPACK "
            "(`build_info()['lapack'] == 'none'`). LAPACK is several times faster in high "
            "dimension (d = 1000: 5.5 ms against 18.6 ms per generation, docs/performance.md); the "
            "results are not bit-identical to Eigen's.")
        .def_prop_rw_static(
            "cholesky_solver",
            [](nb::handle) { return constants::cholesky_solver; },
            [](nb::handle, int a) {
                if (a < -1 || a > 1)
                    throw nb::value_error(
                        "cholesky_solver: -1 (automatic), 0 (Eigen) or 1 (LAPACK dpotrf)");
                if (a == 1 && !lapack::available())
                    throw nb::value_error("cholesky_solver: no LAPACK available (not linked, and "
                                          "scipy's was not found)");
                constants::cholesky_solver = a;
            },
            "The Cholesky factorization of CMSA and COVARIANCE_NO_EIGV (default -1).\n\n"
            "Also used for the factor of `Settings.C0` of CHOLESKY and CMSA. "
            "-1: automatic, LAPACK dpotrf from `lapack_min_dim` on when LAPACK is available, "
            "Eigen below; 0: Eigen's LLT; 1: LAPACK dpotrf (raises ValueError without LAPACK).")
        .def_prop_rw_static(
            "blas",
            [](nb::handle) { return constants::blas; },
            [](nb::handle, int a) {
                if (a < -1 || a > 1)
                    throw nb::value_error("blas: -1 (automatic), 0 (Eigen) or 1 (BLAS/LAPACK)");
                constants::blas = a;
            },
            "The matrix products and the inverse of the matrix adaptations (default -1).\n\n"
            "Covers sampling a batch (Y = A Z), the updates of MATRIX (MA-ES) and "
            "NATURAL_GRADIENT (xNES), and the LU inverse of their matrix (dgetrf/dgetri). -1: "
            "automatic, BLAS/LAPACK from `blas_min_dim` on when available, Eigen below; 0: always "
            "Eigen; 1: always BLAS/LAPACK when available (Eigen otherwise).")
        .def_prop_rw_static(
            "blas_min_dim",
            [](nb::handle) { return constants::blas_min_dim; },
            [](nb::handle, size_t a) { constants::blas_min_dim = a; },
            "The dimension from which `blas` = -1 uses BLAS and LAPACK (default 100).\n\n"
            "Below it, Eigen's products are faster (docs/performance.md, \"LAPACK and BLAS in high "
            "dimension\").")
        .def_prop_rw_static(
            "lapack_min_dim",
            [](nb::handle) { return constants::lapack_min_dim; },
            [](nb::handle, size_t a) { constants::lapack_min_dim = a; },
            "The dimension from which `eigen_solver` = -1 and `cholesky_solver` = -1 use LAPACK "
            "(default 120).\n\n"
            "Below it, Eigen's decompositions are faster (docs/performance.md, \"LAPACK and BLAS "
            "in high dimension\").");
}

void define_options(nb::module_ &main)
{
    auto m = main.def_submodule(
        "options",
        "The options of the modules: one enum per module family.\n\n"
        "A `parameters.Modules` object holds one value of each (and the on/off modules as "
        "bools). The options combine freely; `parameters.Settings` adjusts the few "
        "combinations that cannot run as given (the compatibility page of the guide). The "
        "default of every enum gives the CMA-ES with cumulative step-size adaptation.");
    using namespace parameters;
    nb::enum_<RecombinationWeights>(
        m,
        "RecombinationWeights",
        "The recombination weights (`Modules.weights`; default DEFAULT).\n\n"
        "The weights of the mu best individuals are positive and sum to 1. With "
        "`Modules.active` the other lambda - mu individuals get negative (or zero) weights, "
        "scaled to sum to -min(1 + c1 / cmu, 1 + 2 mueff^- / (mueff + 2), (1 - c1 - cmu) / "
        "(d cmu)) (Hansen 2016, Eqs. (50) to (53)). UTILITY differs from both rules.")
        .value("DEFAULT",
               RecombinationWeights::DEFAULT,
               "Log-linear weights, the default: w_i proportional to ln((n + 1) / 2) - ln(i) "
               "(Hansen 2016, \"The CMA Evolution Strategy: A Tutorial\", Eq. (49)). For the "
               "default mu = floor(lambda / 2), n = lambda: the weights of the standard CMA-ES. "
               "For another mu (`Settings.mu0`, or the default mu = lambda / 4 of SA and CMSA), "
               "n = 2 mu, as pycma with CMA_mu: ln(mu + 1/2) - ln(i) for the mu best, and the "
               "mu negative weights of that population go to the worst ranks (the ranks between "
               "get 0 when lambda > 2 mu; only the last lambda - mu of them are used when "
               "lambda < 2 mu) (#73).")
        .value("EQUAL",
               RecombinationWeights::EQUAL,
               "Equal weights 1/mu (intermediate recombination): every selected individual "
               "counts the same; with `Modules.active` the negative weights are equal too.")
        .value("EXPONENTIAL",
               RecombinationWeights::EXPONENTIAL,
               "Exponentially decaying weights, w_i = 2^-i + 1 / (mu 2^mu) for i = 1..mu, which "
               "sum to 1 (de Nobel et al. 2026, the framework paper, Sec. 4.2): the first term "
               "halves with every rank, more weight on the best individuals than DEFAULT. The "
               "negative weights follow the same rule for the lambda - mu worst, the largest on "
               "the worst.")
        .value("UTILITY",
               RecombinationWeights::UTILITY,
               "The utilities of NES and xNES (Wierstra et al. 2014; Glasmachers et al. 2010), "
               "u_k = max(0, ln(lambda/2 + 1) - ln k) / sum - 1/lambda: log-shaped for the better "
               "half, minus a baseline, summing to zero. Unlike the other options they are not "
               "only the weights of the mu parents: the mean moves by the utility-weighted sum of "
               "all lambda steps, and every matrix update uses all of them (`Settings` sets "
               "`Modules.active`, except with matrix_adaptation NONE and CMSA; CMSA uses the "
               "positive utilities). The step-size rules SA, LPXNES and XNES without "
               "NATURAL_GRADIENT use the positive utilities of the mu best, normalized to sum to "
               "1. With NATURAL_GRADIENT and StepSizeAdaptation.XNES this is xNES.");

    nb::enum_<BaseSampler>(
        m,
        "BaseSampler",
        "The source of the uniform numbers in (0, 1)^d that become the samples "
        "(`Modules.sampler`; default UNIFORM).\n\n"
        "The `SampleTranformerType` (`Modules.sample_transformation`) maps them to the "
        "distribution of the mutations, by default the standard normal one. SOBOL and HALTON "
        "are quasi-random (low-discrepancy) sequences: their points cover the space more evenly "
        "than random ones (de Nobel et al. 2024, \"Sampling in CMA-ES: low numbers of low "
        "discrepancy points\").")
        .value("UNIFORM",
               BaseSampler::UNIFORM,
               "Pseudo-random uniform numbers (mt19937, the optimizer's own generator), the "
               "default. With the GAUSSIAN transformation: ordinary random normal samples, which "
               "`Settings.fast_sampling` replaces by a faster generator.")
        .value("SOBOL",
               BaseSampler::SOBOL,
               "Points of the Sobol sequence (Sobol' 1967). A cache of points (see "
               "`constants.cache_max_doubles`) is computed when the sampler is created and drawn "
               "in a random order, which repeats after all of them: N = max(128, the largest "
               "power of 2 <= 2e6 / d) points with the default constants; at d = 1000, N = 1024: "
               "after 42 generations of the default lambda = 24. Only for d <= 1111, the size of "
               "the table of direction numbers: with a larger d, creating the sampler ends the "
               "process.")
        .value("HALTON",
               BaseSampler::HALTON,
               "Points of the Halton sequence (Halton 1960), scrambled with random digit "
               "permutations.");

    nb::enum_<SampleTranformerType>(
        m,
        "SampleTranformerType",
        "The distribution of the mutation vectors (`Modules.sample_transformation`; default "
        "GAUSSIAN).\n\n"
        "Each coordinate of a uniform sample of the `BaseSampler` is mapped through the inverse "
        "CDF of a symmetric distribution. Except for CAUCHY (and NONE), the distributions have "
        "unit variance. de Nobel et al. (2025), \"Abnormal mutations: evolution strategies don't "
        "require Gaussianity\" (doi:10.1145/3712256.3726479).")
        .value("NONE",
               SampleTranformerType::NONE,
               "No transformation: the raw base sampler, for testing. The uniform numbers of the "
               "base sampler, in (0, 1) with mean 0.5, are not centred at 0, so they fail as "
               "mutations (SCALED_UNIFORM is the centred uniform distribution). "
               "`get_configspace` leaves it out.")
        .value("GAUSSIAN",
               SampleTranformerType::GAUSSIAN,
               "Standard normal samples (the inverse normal CDF, or Box-Muller with "
               "`constants.use_box_muller`; with the UNIFORM sampler and "
               "`Settings.fast_sampling`, a ziggurat), the default: the CMA-ES.")
        .value("SCALED_UNIFORM",
               SampleTranformerType::SCALED_UNIFORM,
               "Uniform in (-sqrt(3), sqrt(3)): bounded mutations with unit variance.")
        .value("LAPLACE",
               SampleTranformerType::LAPLACE,
               "Laplace distribution with scale 1/sqrt(2) (unit variance): heavier tails than "
               "the normal distribution.")
        .value("LOGISTIC",
               SampleTranformerType::LOGISTIC,
               "Logistic distribution with scale sqrt(3)/pi (unit variance): slightly heavier "
               "tails than the normal distribution.")
        .value("CAUCHY",
               SampleTranformerType::CAUCHY,
               "Standard Cauchy distribution (scale 1): very heavy tails and no finite variance; "
               "occasional long jumps. E||z|| is infinite: CSA and threshold convergence use an "
               "empirical approximation of the median of ||z||, d (1 + 0.18 tanh(d / 4.07)), "
               "instead.")
        .value("DOUBLE_WEIBULL",
               SampleTranformerType::DOUBLE_WEIBULL,
               "Double Weibull distribution with shape 2 and scale 1 (unit variance), "
               "+-sqrt(-ln(2u)): bimodal, with little mass near 0.");

    nb::enum_<Mirror>(m,
                      "Mirror",
                      "Mirrored sampling (`Modules.mirrored`; default NONE).\n\n"
                      "With mirroring, individuals 2i and 2i + 1 of every generation are a "
                      "pair z, -z, which spreads the samples more evenly around the mean: "
                      "ceil(lambda / 2) independent samples per generation. With an odd lambda "
                      "the last individual is unpaired, so with lambda = 1 mirroring has no "
                      "effect. A pair that bound correction (RESAMPLE) or repelling rejects is "
                      "sampled again as a pair.")
        .value("NONE", Mirror::NONE, "Independent samples, the default.")
        .value("MIRRORED",
               Mirror::MIRRORED,
               "Mirrored sampling (Brockhoff et al. 2010): the samples come in pairs z, -z. All "
               "the points take part in the selection.")
        .value("PAIRWISE",
               Mirror::PAIRWISE,
               "Mirrored sampling with pairwise selection (Auger et al. 2011): the worse point of "
               "each mirrored pair is ranked after all the better ones (and after the parents of "
               "elitism), so that with mu <= lambda / 2 only the better one can be selected and "
               "the two do not cancel out in the recombination. The worse points keep their "
               "objective values and get no negative weights in the active update (as pycma). "
               "Makes lambda even (the default mu is half the even lambda); with lambda0 = 1 (the "
               "(1+1)-ES) Settings turn it into NONE.");

    nb::enum_<StepSizeAdaptation>(
        m,
        "StepSizeAdaptation",
        "The adaptation of the global step size sigma (`Modules.ssa`; default CSA).\n\n"
        "With the matrix adaptation COVARIANCE, any method other than CSA switches it to "
        "COVARIANCE_NO_EIGV (no eigendecomposition), unless `Settings(always_compute_eigv=True)`. "
        "A population of one (lambda0 = 1, the (1+1)-ES) always uses SR, whatever ssa is given "
        "(and so COVARIANCE becomes COVARIANCE_NO_EIGV). The learning rate cs and the damping "
        "damps of each method are its defaults below; `Settings.cs` and `Settings.damps` "
        "replace them. For a comparison of the methods see de Nobel et al. 2021 "
        "(doi:10.1145/3449726.3463167).")
        .value("CSA",
               StepSizeAdaptation::CSA,
               "Cumulative step-size adaptation (Hansen & Ostermeier 2001), the default: sigma "
               "grows when the evolution path p_sigma is longer than expected under random "
               "selection and shrinks when it is shorter, sigma <- sigma exp(cs / damps "
               "(||p_sigma|| / E||z|| - 1)), the exponent clipped to [-1, 1] (as pycma). cs = "
               "(mueff + 2) / (d + mueff + 5) and damps = 1 + 2 max(0, sqrt((mueff - 1) / (d + "
               "1)) - 1) + cs (Hansen 2016, Table 1, Eq. (55); #138).")
        .value("TPA",
               StepSizeAdaptation::TPA,
               "Two-point step-size adaptation (Hansen 2008, \"CMA-ES with Two-Point Step-Size "
               "Adaptation\"): two extra points, m +- 0.5 sigma_old dm, half the last step of the "
               "mean with the step size that sampled it (Eqs. (8), (9)), are evaluated every "
               "generation (two more evaluations). r = 0.5 when the point ahead is the better "
               "one, -0.5 when the point behind is, 0 on a tie (as pycma; the paper counts a tie "
               "as a success); s <- (1 - cs) s + cs r and sigma <- sigma exp(s), cs = 0.3 (Eqs. "
               "(10) to (13)); damps is not used. With a bound correction both points move "
               "toward m, symmetrically, until both are in the box. With elitism, a generation "
               "whose selection kept every parent is a failure: the next generation evaluates no "
               "test points (the mean did not move; the budget still reserves them) and takes r "
               "= -0.5, as in the success rules of elitist ESs (#179; no TPA source covers "
               "elitism). Before 2.0 the points were m +- sigma dm, about one step of the mean "
               "(#167). `TPA_INPLACE` is the version within the population.")
        .value("MSR",
               StepSizeAdaptation::MSR,
               "Median success rule (Ait ElHara et al. 2013, as pycma): K is the number of "
               "offspring better than the 0.3 lambda-th best of the previous generation "
               "(interpolated between two ranks, as pycma), z = 2 K / lambda - 1, s <- (1 - cs) s "
               "+ cs z and sigma <- sigma exp(s / damps), with cs = 0.3 and damps = 2 - 2/d (at "
               "least 1). sigma grows when more than half of the offspring are better (#83). "
               "With pairwise selection only the ranked offspring count.")
        .value("XNES",
               StepSizeAdaptation::XNES,
               "The step-size update of xNES. Without NATURAL_GRADIENT, xNES SA (Krause, "
               "Glasmachers & Igel (2017), Sec. 4.2): sigma <- sigma exp(cs / sqrt(d) sum_i w_i "
               "(||z_i||^2 - d)) over the mu best, with their positive weights and cs = mueff / "
               "(2 ln(max(2, d)) sqrt(d)) (their Table 3; #72). With NATURAL_GRADIENT, xNES's own "
               "update (Glasmachers et al. 2010): sigma <- sigma exp(cs / 2 tr(G) / d), G the "
               "natural gradient, with xNES's cs = eta_sigma = (9 + 3 ln d) / (5 d sqrt(d)). With "
               "elitism both read the offspring alone (#172). With `Modules.active` (and weights "
               "other than UTILITY) a generation with h_sigma = 0 has no negative weights in the "
               "matrix update of COVARIANCE, COVARIANCE_NO_EIGV, SEPARABLE, MATRIX and CHOLESKY "
               "(#175; modcma's rule: Hansen 2016 and pycma keep them). xNES SA with weights "
               "EQUAL fails on the ellipsoid (#180, see the compatibility page).")
        .value("MXNES",
               StepSizeAdaptation::MXNES,
               "A variant of the xNES update from the length of the weighted mean of the "
               "selected samples instead of their individual lengths: sigma <- sigma exp(cs / d "
               "(mueff ||sum_i w_i z_i||^2 - d)), with cs = 1 (m-xNES, Krause, Glasmachers & "
               "Igel (2017), \"Qualitative and quantitative assessment of step size adaptation "
               "rules\", Sec. 4.3, Table 3).")
        .value("LPXNES",
               StepSizeAdaptation::LPXNES,
               "An xNES-like update from the logarithms of the individual step sizes of the "
               "samples (lp-xNES, Krause, Glasmachers & Igel (2017), \"Qualitative and "
               "quantitative assessment of step size adaptation rules\", Sec. 4.4). Every "
               "individual samples its own step size, as with `Modules.sample_sigma` but with "
               "another tau: sigma_i = sigma exp(tau n_i), tau = ln(2) / (sqrt(d) ln(max(2, d))). "
               "Then sigma <- sigma^(1 - cs) exp(cs sum_i w_i ln(sigma_i)) over the mu best with "
               "their positive weights; cs = 9 mueff / (10 sqrt(d)) (their Table 3), capped at 1 "
               "(modcma's cap, #72).")
        .value("PSR",
               StepSizeAdaptation::PSR,
               "Population success rule (Loshchilov 2014, LM-CMA-ES): the offspring and the "
               "previous generation are ranked together (equal values share their mean rank, "
               "#120); z = (sum of the ranks of the previous generation - sum of the ranks of the "
               "offspring) / n^2 - 0.25, the target success ratio, with n the size of each; s <- "
               "(1 - cs) s + cs z and sigma <- sigma exp(s / damps), with cs = 0.3 and damps = 1 "
               "(Loshchilov 2014; #83). With pairwise selection only the ranked offspring count.")
        .value("SR",
               StepSizeAdaptation::SR,
               "Success rule of the (1+1)-ES (Igel et al. 2006): a generation is a success when "
               "its best point beats the best of the run so far; the success ratio p is averaged "
               "with rate 1/12, and sigma <- sigma exp((p - 2/11) / ((1 - 2/11) damps)), damps = "
               "1 + d/2: sigma grows when p is above the target 2/11. Used automatically when "
               "lambda0 = 1. Designed for elitist selection, where the best of the run is the "
               "parent; without elitism it plays the parent's role (the framework paper compares "
               "with the previous generation's best, which made sigma diverge, #83).")
        .value("SA",
               StepSizeAdaptation::SA,
               "Self-adaptation: every individual samples its own step size sigma_i = sigma "
               "exp(tau n_i), with CMSA's tau = 1 / sqrt(2 d) (Spettel, Beyer & Hellwig 2018, "
               "Table I; #71); sigma is multiplied by the weighted geometric mean of the step "
               "sizes of the mu selected individuals divided by the geometric mean of those of "
               "the population (the CMSA-ES operator of Beyer & Sendhoff 2008, weighted, with the "
               "correction for elitism of Ahrari et al. 2017, as the framework paper describes "
               "it). Meant for CMSA; makes the default mu lambda / 4.")
        .value("TPA_INPLACE",
               StepSizeAdaptation::TPA_INPLACE,
               "Two-point step-size adaptation within the population, as pycma 4.6.0's "
               "CMAAdaptSigmaTPA (after Hansen, Atamna & Auger 2014, Eq. (6) and Algorithm 3). "
               "From the second generation of a run on, offspring 1 and 2 are a mirrored pair "
               "along the last mean shift, z_1 = ||N(0, I)|| A^-1 dm / ||A^-1 dm|| and z_2 = "
               "-z_1, so no extra evaluations. With z = (rank of offspring 2 - rank of offspring "
               "1) / (n - 1) among the n offspring, positive when offspring 1 is the better one, "
               "s <- (1 - cs) s + cs sign(z) |z|^0.5 and sigma <- sigma exp(s / damps); cs = 0.3, "
               "damps = 0.7 + 2 ln(d) + 2 ln(max(1, lambda - d)). pycma departs from the paper in "
               "the |z|^0.5 (the paper uses z), the damping (the paper: sqrt(d)), and the update "
               "in the first generation of a run, from two ordinary samples (the paper updates "
               "from the second generation on); modcma follows pycma. A generation with fewer "
               "than two offspring is not updated. Works with mirrored sampling (the test points "
               "are the first pair), pairwise selection (the ranks are of all offspring), "
               "sequential selection (which cannot stop before both are evaluated; n is then the "
               "evaluated offspring) and elitism (the ranks are of the offspring; after a "
               "generation whose selection kept every parent, the generation samples ordinary "
               "offspring and z = -1, #179); with orthogonal sampling the test points are not "
               "orthogonal to the other samples. #173.");

    nb::enum_<CorrectionMethod>(
        m,
        "CorrectionMethod",
        "What happens to a sample outside the box bounds (`Modules.bound_correction`; default "
        "NONE).\n\n"
        "The bounds are `Settings.lb` and `Settings.ub` (default [-5, 5] in every coordinate). "
        "The methods other than RESAMPLE move each coordinate that is out of bounds into "
        "[lb, ub]; the mutation vector of the sample is then recomputed from the corrected "
        "point, so the adaptation learns from the point that was evaluated. The methods are "
        "those of Caraffini, Kononova & Corne (2019), \"Infeasibility and structural bias in "
        "differential evolution\".\n\n"
        "Bounds may be infinite; each coordinate and each side is corrected separately, and an "
        "infinite bound is never crossed. MIRROR, COTN, UNIFORM_RESAMPLE and TOROIDAL need the "
        "width ub - lb: in a coordinate with one infinite bound they set a coordinate that "
        "crosses the finite bound to that bound, as SATURATE does.")
        .value("NONE",
               CorrectionMethod::NONE,
               "No correction, the default: points outside the bounds are evaluated as they are "
               "(and counted in `Parameters.bounds.n_out_of_bounds`).")
        .value("MIRROR",
               CorrectionMethod::MIRROR,
               "Reflect the coordinate at the violated bound (periodically, for points more than "
               "a domain width outside).")
        .value("COTN",
               CorrectionMethod::COTN,
               "Complete one-tailed normal correction: place the coordinate inside the violated "
               "bound, at a random distance sigma * |N(0, 1/3^2)| times the domain width ub - lb. "
               "A distance above the width is drawn again (up to 10 times), then set to the "
               "width: the point stays in the box.")
        .value("UNIFORM_RESAMPLE",
               CorrectionMethod::UNIFORM_RESAMPLE,
               "Replace the coordinate by a uniform random value in [lb, ub].")
        .value("SATURATE",
               CorrectionMethod::SATURATE,
               "Set the coordinate to the violated bound (clipping): corrected points lie on the "
               "boundary.")
        .value("TOROIDAL",
               CorrectionMethod::TOROIDAL,
               "Wrap the coordinate around: leaving at one bound enters at the other (a torus).")
        .value("RESAMPLE",
               CorrectionMethod::RESAMPLE,
               "Sample the whole individual again until it lies inside the bounds, at most 5 d "
               "times; after that, the last sample is evaluated as it is (such a point does not "
               "become `stats.global_best`, #168). A point outside the box that is not a sample "
               "(an injected gradient point; TPA's test points when the mean is outside the box) "
               "is clipped to it instead.");

    nb::enum_<RestartStrategyType>(
        m,
        "RestartStrategy",
        "When the optimizer restarts, and with which population size "
        "(`Modules.restart_strategy`; default NONE).\n\n"
        "A restart happens when a restart criterion (`Parameters.criteria`, module `restart`) "
        "is met: the run ends with the evaluation of its center of mass, the mean is placed "
        "again (`CenterPlacement`), sigma is reset to sigma0 (BIPOP: possibly smaller) and the "
        "adaptation starts over. Except for NONE, the strategies use the full set of criteria "
        "(see `restart.Criteria`). With lambda0 = 1, IPOP and BIPOP become RESTART.")
        .value("NONE",
               RestartStrategyType::NONE,
               "No restart criteria other than `restart.UnableToAdapt`, the default: the "
               "optimizer only restarts (with the same population size) when its adaptation "
               "fails.")
        .value("STOP",
               RestartStrategyType::STOP,
               "Use the restart criteria as stopping criteria: the run ends when one of them is "
               "met, without a restart.")
        .value("RESTART",
               RestartStrategyType::RESTART,
               "Restart with the same population size whenever a restart criterion is met.")
        .value("IPOP",
               RestartStrategyType::IPOP,
               "IPOP-CMA-ES (Auger & Hansen 2005): every restart multiplies lambda by "
               "`restart.IPOP.ipop_factor` (2), up to 2^9 lambda0 "
               "(`restart.IPOP.max_lambda_factor`); mu is the default of the new lambda, or the "
               "ratio of a given mu0. Larger populations help on multimodal functions with a "
               "global structure.")
        .value("BIPOP",
               RestartStrategyType::BIPOP,
               "BIPOP-CMA-ES (Hansen 2009): alternates between a large population, which doubles "
               "every time it is used up to 2^9 lambda0 (as IPOP), and a small one with a random "
               "size between lambda0 and the latest large population (as pycma; Hansen 2009 has "
               "half of it, #140) and a smaller initial sigma, sigma0 10^(-2 U) with U uniform in "
               "(0, 1). Each restart uses the regime that has used fewer evaluations so far (the "
               "first run counts as small), so both get about half of the evaluations; a small "
               "run takes at most half of what the large regime has used. mu as with IPOP. See "
               "`restart.BIPOP`.");

    nb::enum_<MatrixAdaptationType>(
        m,
        "MatrixAdaptationType",
        "The adaptation of the shape of the search distribution (`Modules.matrix_adaptation`; "
        "default COVARIANCE).\n\n"
        "The mutations are y = A z, with z from the sampler and A a factor of the covariance "
        "matrix, C = A A^T. The options differ in how A is adapted and in their cost per "
        "generation (docs/performance.md, \"ModularCMAES matrix adaptations\"). C starts as "
        "the identity, or as `Settings.C0` (warm starting).")
        .value("COVARIANCE",
               MatrixAdaptationType::COVARIANCE,
               "The covariance matrix adaptation of the CMA-ES (Hansen & Ostermeier 2001), with "
               "rank-one and rank-mu update, the default. C is updated every generation and "
               "eigendecomposed when 1 / (10 d (c1 + cmu)) generations have passed since the last "
               "decomposition (every generation in low dimension; O(d^3), with LAPACK from "
               "`constants.lapack_min_dim` on). Keeps its metric (`Adaptation.metric()`), so the "
               "restart criteria ConditionC and NoEffectAxis are used, as with SEPARABLE and "
               "NONE.")
        .value("NONE",
               MatrixAdaptationType::NONE,
               "No matrix adaptation (C = I): only the step size is adapted. Disables "
               "`Modules.active`, rejects `Settings.C0`. O(d lambda) per generation.")
        .value("MATRIX",
               MatrixAdaptationType::MATRIX,
               "MA-ES (Beyer & Sendhoff 2017): adapts a matrix M with C = M M^T directly, "
               "without the evolution path p_c (its rank-one term reads p_sigma) and without "
               "eigendecomposition; O(d^2 lambda) per generation, about 6 times faster than "
               "COVARIANCE at d = 1000 (docs/performance.md). With a step-size rule other than "
               "CSA the rank-one term is skipped in a generation with h_sigma = 0 (#160). With "
               "`Modules.active` it uses the active update of COVARIANCE, written in M (MA-ES has "
               "none; #73).")
        .value("SEPARABLE",
               MatrixAdaptationType::SEPARABLE,
               "sep-CMA-ES (Ros & Hansen 2008): adapts only the diagonal of C, O(d lambda) per "
               "generation, with pycma's separable learning rates c1 and cmu. For separable "
               "problems and very high dimension; it cannot learn correlations between "
               "variables. Of `Settings.C0` it uses the diagonal.")
        .value("CHOLESKY",
               MatrixAdaptationType::CHOLESKY,
               "Cholesky-CMA-ES (Krause et al. 2016): adapts the lower triangular Cholesky factor "
               "of C with rank-one updates of the factor, without decomposing C; O(d^2 lambda) "
               "per generation.")
        .value("CMSA",
               MatrixAdaptationType::CMSA,
               "CMSA-ES (Beyer & Sendhoff 2008), covariance matrix self-adaptation: C is an "
               "exponential average of the selected y y^T with time constant 1 + d (d + 1) / "
               "(2 mueff), factorized with a Cholesky decomposition every generation. Meant for "
               "StepSizeAdaptation.SA; makes the default mu lambda / 4. Disables "
               "`Modules.active`: CMSA has no active variant.")
        .value("COVARIANCE_NO_EIGV",
               MatrixAdaptationType::COVARIANCE_NO_EIGV,
               "The covariance update of COVARIANCE, with C factorized by a Cholesky "
               "decomposition instead of an eigendecomposition (cheaper), at the same interval. "
               "The path p_sigma uses the inverse Cholesky factor instead of C^-1/2 (the same up "
               "to a rotation); with CSA it gave the same results as COVARIANCE on the sphere and "
               "a rotated ellipsoid (d = 5 to 20; the deep dive on the matrix adaptations). There "
               "is no published counterpart. Chosen automatically when COVARIANCE is combined "
               "with a step-size adaptation other than CSA, and for lambda0 = 1.")
        .value("NATURAL_GRADIENT",
               MatrixAdaptationType::NATURAL_GRADIENT,
               "xNES (Glasmachers et al. 2010), exponential natural evolution strategies: A is "
               "updated with the matrix exponential of the natural gradient, A <- A expm(cc G) "
               "with G = sum_i w_i (z_i z_i^T - I) made trace-free (its trace goes to sigma) and "
               "xNES's cc = eta_B / 2 = (9 + 3 ln d) / (10 d sqrt(d)); a low-rank form keeps it "
               "O(d^2 lambda) for lambda < d. With StepSizeAdaptation.XNES also xNES's update of "
               "sigma, with xNES's learning rate cs = eta_sigma (the other step-size adaptations "
               "keep their own cs; #72), and with RecombinationWeights.UTILITY (also selected by "
               "`Modules.active` with the default weights) xNES's utilities for the mean and the "
               "matrix: all three together are xNES.");

    nb::enum_<CenterPlacement>(
        m,
        "CenterPlacement",
        "Where the mean of the search distribution is placed at every restart "
        "(`Modules.center_placement`; default X0). The first run always starts at "
        "`Settings.x0`, or at the center of the bounds.")
        .value("X0",
               CenterPlacement::X0,
               "At `Settings.x0`, or at the center of the bounds (`Settings.center`) when x0 is "
               "not given (0 with the default bounds), the default.")
        .value("ZERO", CenterPlacement::ZERO, "At the origin.")
        .value("UNIFORM",
               CenterPlacement::UNIFORM,
               "At a uniform random point in the bounds: every restart starts elsewhere. In a "
               "coordinate with an infinite bound, uniform in x0 +- 2 sigma0 (moved inside a "
               "finite bound; x0 is `Settings.x0` or the center).")
        .value("CENTER",
               CenterPlacement::CENTER,
               "At the center of the bounds, `Settings.center`: (lb + ub) / 2 in a coordinate "
               "with two finite bounds; with one infinite bound, 0 if 0 lies between the bounds, "
               "else the finite bound moved inward by sigma0; 0 with two infinite bounds (#134).")
        .value("NOVELTY_WEIGHTED",
               CenterPlacement::NOVELTY_WEIGHTED,
               "At one of 512 uniform random candidates in the bounds, drawn with a probability "
               "proportional to its squared distance to the nearest point of the repelling "
               "archive (distances relative to the domain width). Uniform while the archive is "
               "empty; the archive needs `Modules.repelling_type`. In a coordinate with an "
               "infinite bound the candidates are drawn as by UNIFORM.")
        .value("MAXIMIN_TABOO",
               CenterPlacement::MAXIMIN_TABOO,
               "At the one of 512 uniform random candidates in the bounds that is farthest from "
               "its nearest point of the repelling archive (maximin; distances relative to the "
               "domain width). Uniform while the archive is empty; the archive needs "
               "`Modules.repelling_type`. In a coordinate with an infinite bound the candidates "
               "are drawn as by UNIFORM.");

    nb::enum_<RepellingType>(
        m,
        "RepellingType",
        "Repelling restarts (`Modules.repelling_type`; default NONE): de Nobel et al. 2024, "
        "\"Avoiding redundant restarts in multimodal global optimization\" "
        "(doi:10.1007/978-3-031-70068-2_17).\n\n"
        "After a restart, the result of the run (the better of its best point and its final "
        "center of mass) joins an archive of taboo points, unless hill-valley tests (which "
        "evaluate the objective at up to 10 points per taboo point) show that it lies in the "
        "basin of a point already there. A new sample whose distance to a taboo point "
        "(`RepellingMetric`, divided by the current sigma) is below the point's radius is "
        "rejected and drawn again; the radius shrinks by 0.95^(1/d) with every rejection in a "
        "generation, and only the taboo points near the mean are checked. This steers later "
        "runs away from the basins already found. Only useful with restarts; with a restart "
        "strategy other than NONE it adds the restart criterion `restart.TooMuchRepelling`.")
        .value("NONE", RepellingType::NONE, "No repelling, the default.")
        .value("COVERAGE",
               RepellingType::COVERAGE,
               "Radii from a coverage of the search volume (de Nobel et al. 2024, Eqs. (10), "
               "(11)): r = (n_T S / (c sigma0^d N))^(1/d) Gamma(d/2 + 1)^(1/d) / sqrt(pi), in "
               "units of the current sigma, with S the volume of the box (10^d with an infinite "
               "bound), c = `Repelling.coverage` (2), N the runs so far and n_T the runs that "
               "ended in the basin of the point. sigma0 is the problem's, also in BIPOP's small "
               "runs; the paper divides by sigma0 instead of sigma0^d (#158). See "
               "`repelling.CoverageRepelling`.")
        .value("ADAPTIVE",
               RepellingType::ADAPTIVE,
               "Experimental (modcma's own, without a published source): radii adapted from the "
               "rejection rate of the samples and the evaluations the runs need, between a "
               "minimum and a maximum radius. In the experiment of #158 (BBOB f21 and f22, d = 5 "
               "to 40, RESTART and IPOP) it solved 99 of 240 runs, 107 without repelling and 144 "
               "with COVERAGE (the compatibility page).");

    nb::enum_<RepellingMetric>(
        m,
        "RepellingMetric",
        "The distance between a sample and a taboo point of repelling "
        "(`Modules.repelling_metric`; default CURRENT).\n\n"
        "Every metric divides the distance by the current sigma before comparing it with the "
        "radius of the taboo point. The default was EUCLIDEAN until #158; the repelling paper "
        "(de Nobel et al. 2024, Eq. (8)) measures the Mahalanobis distance.")
        .value("EUCLIDEAN",
               RepellingMetric::EUCLIDEAN,
               "||x - x_t||, the Euclidean distance (the default before #158).")
        .value("CURRENT",
               RepellingMetric::CURRENT,
               "||A^-1 (x - x_t)||, with A of the current run: the Mahalanobis distance in the "
               "current search distribution (de Nobel et al. 2024, Eq. (8)), the default. With "
               "matrix_adaptation NONE it is the Euclidean distance.")
        .value("ARCHIVED",
               RepellingMetric::ARCHIVED,
               "||A_t^-1 (x - x_t)||, with A_t of the run that found x_t, scaled to unit "
               "determinant: the shape of the basin that run had learned (the Euclidean distance "
               "when that matrix is degenerate).");
}

void define_utils(nb::module_ &main)
{
    // The public helpers live in the main module (modcma.set_seed, build_info, compute_ert); the
    // deprecated modcma.c_maes.utils (Python) forwards to them
    main.def(
        "build_info",
        []() {
            return std::map<std::string, std::string>{
                {"compiler", compiler_version()},
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
                {"fma", "yes"},
#else
                {"fma", "no"},
#endif
                // "linked" (a build with MODCMA_LAPACK: Accelerate on macOS), "scipy" (scipy's
                // LAPACK, given at import), or "none"
                {"lapack", lapack::source()},
            };
        },
        "How the extension was built.\n\n"
        "Returns\n"
        "-------\n"
        "dict[str, str]\n"
        "    'compiler': the compiler and its version, e.g. 'gcc 13.3.0';\n"
        "    'fma': 'yes' when the code may use fused multiply-add instructions (which change\n"
        "    the last bits of results), 'no' otherwise;\n"
        "    'lapack': where LAPACK and BLAS come from: 'linked' (built with them, Accelerate\n"
        "    on macOS), 'scipy' (scipy's, handed over at import) or 'none' (Eigen only).");
    main.def("set_seed",
             &rng::set_seed,
             nb::arg("seed"),
             "Seed the global random generator (mt19937).\n\n"
             "An optimizer without `Settings.seed` copies the global generator when it is "
             "created and then reseeds the global one from its next number, so that after "
             "set_seed(s) the optimizers created in the same order draw the same numbers. "
             "Samplers and bound corrections that are used on their own draw from the global "
             "generator directly. The global generator is not part of the saved state of an "
             "optimizer (pickle).\n\n"
             "Parameters\n"
             "----------\n"
             "seed : int\n"
             "    The seed.");
    main.def("compute_ert",
             &utils::compute_ert,
             nb::arg("running_times"),
             nb::arg("budget"),
             "The expected running time (ERT) of a set of runs.\n\n"
             "Parameters\n"
             "----------\n"
             "running_times : list[int]\n"
             "    The evaluations each run used; a run succeeded when it used fewer than budget.\n"
             "budget : int\n"
             "    The budget of each run.\n\n"
             "Returns\n"
             "-------\n"
             "tuple[float, int]\n"
             "    The sum of all running times divided by the number of successful runs (inf when "
             "no run succeeded, nan for an empty list), and the number of successful runs.");

    auto m = main.def_submodule(
        "_internal",
        "Internals of the library, exposed for the tests: the LAPACK/BLAS routines, the Sobol "
        "sequence and the permutation generators of the samplers. Not part of the API: they may "
        "change without notice.");
    m.def(
        "i8_sobol",
        [](int dim_num, long long seed, nb::ndarray<double, nb::ndim<1>, nb::c_contig> quasi) {
            if (quasi.shape(0) < static_cast<size_t>(dim_num))
                throw nb::value_error("quasi must hold at least dim_num numbers");
            i8_sobol(dim_num, &seed, quasi.data());
            return seed;
        },
        nb::arg("dim_num"),
        nb::arg("seed"),
        nb::arg("quasi"),
        "Write the next point of the Sobol sequence into quasi and return the next seed.\n\n"
        "The routine keeps its tables in static variables: calls from several threads at once "
        "are not safe.\n\n"
        "Parameters\n"
        "----------\n"
        "dim_num : int\n"
        "    The dimension of the point, 1 to 1111 (the size of the table of direction numbers; "
        "outside this range the routine ends the process).\n"
        "seed : int\n"
        "    The position in the sequence (the `sampling.Sobol` sampler starts at 2).\n"
        "quasi : numpy.ndarray\n"
        "    A contiguous float64 array of at least dim_num numbers, overwritten with the "
        "point.\n\n"
        "Returns\n"
        "-------\n"
        "int\n"
        "    The seed of the point after this one.");
    m.def(
        "set_lapack",
        [](nb::dict capsules, const std::string &source) {
            // PyCapsules of scipy.linalg.cython_lapack.__pyx_capi__ (the name is the C signature)
            const auto pointer = [](nb::handle capsule) -> void * {
                if (!PyCapsule_CheckExact(capsule.ptr()))
                    throw nb::type_error("expected a PyCapsule");
                void *p = PyCapsule_GetPointer(capsule.ptr(), PyCapsule_GetName(capsule.ptr()));
                if (p == nullptr)
                    throw nb::python_error();
                return p;
            };
            const auto get = [&](const char *name) { return pointer(capsules[name]); };
            lapack::Routines r;
            r.syevd = reinterpret_cast<lapack::syevd_fn>(get("dsyevd"));
            r.syevr = reinterpret_cast<lapack::syevr_fn>(get("dsyevr"));
            r.potrf = reinterpret_cast<lapack::potrf_fn>(get("dpotrf"));
            r.getrf = reinterpret_cast<lapack::getrf_fn>(get("dgetrf"));
            r.getri = reinterpret_cast<lapack::getri_fn>(get("dgetri"));
            r.gemm = reinterpret_cast<lapack::gemm_fn>(get("dgemm"));
            lapack::set_routines(r, source);
        },
        nb::arg("capsules"),
        nb::arg("source"),
        "Use these LAPACK/BLAS routines (internal).\n\n"
        "capsules maps the names dsyevd, dsyevr, dpotrf, dgetrf, dgetri and dgemm to the "
        "PyCapsules of scipy.linalg.cython_lapack and cython_blas; source is what "
        "build_info()['lapack'] reports afterwards. Ignored when the library has its own "
        "(build_info()['lapack'] == 'linked'). modcma.c_maes calls it at import.");
    m.def(
        "lapack_eigh",
        [](const Matrix &C, int method) {
            require_square("lapack_eigh", C);
            Vector w;
            Matrix V;
            lapack::Workspace ws;
            if (!lapack::symmetric_eigen(
                    C, w, V, method == 2 ? lapack::Method::SYEVR : lapack::Method::SYEVD, ws))
                throw std::runtime_error("LAPACK eigendecomposition failed or not available");
            return std::make_pair(w, V);
        },
        nb::arg("C"),
        nb::arg("method") = 1,
        "Eigenvalues and eigenvectors of the symmetric matrix C with LAPACK, as the covariance "
        "adaptation computes them.\n\n"
        "Parameters\n"
        "----------\n"
        "C : numpy.ndarray\n"
        "    A symmetric matrix (its lower triangle is read).\n"
        "method : int\n"
        "    1: dsyevd (divide and conquer, default), 2: dsyevr (MRRR); any other value: "
        "dsyevd.\n\n"
        "Returns\n"
        "-------\n"
        "tuple[numpy.ndarray, numpy.ndarray]\n"
        "    The eigenvalues in ascending order and the orthonormal eigenvectors (columns).\n\n"
        "Raises\n"
        "------\n"
        "ValueError\n"
        "    When C is not square.\n"
        "RuntimeError\n"
        "    When LAPACK fails or is not available.");
    m.def(
        "lapack_cholesky",
        [](const Matrix &C) {
            require_square("lapack_cholesky", C);
            Matrix L;
            if (!lapack::cholesky(C, L))
                throw std::runtime_error("LAPACK Cholesky failed or not available");
            return L;
        },
        nb::arg("C"),
        "The lower triangular Cholesky factor L of C (C = L L^T) with LAPACK dpotrf, as CMSA "
        "and COVARIANCE_NO_EIGV compute it.\n\n"
        "Raises ValueError when C is not square, RuntimeError when C is not positive definite "
        "or LAPACK is not available.");
    m.def(
        "lapack_inverse",
        [](const Matrix &M) {
            require_square("lapack_inverse", M);
            Matrix inverse;
            if (!lapack::inverse(M, inverse))
                throw std::runtime_error("LAPACK inverse failed (singular) or not available");
            return inverse;
        },
        nb::arg("M"),
        "The inverse of the square matrix M with an LU factorization (LAPACK dgetrf, dgetri), "
        "as MA-ES and xNES compute it.\n\n"
        "Raises ValueError when M is not square, RuntimeError when M is singular or LAPACK is "
        "not available.");
    m.def(
        "blas_gemm",
        [](const Matrix &A, const Matrix &B, bool transpose_a, bool transpose_b) {
            if (!lapack::available())
                throw std::runtime_error("BLAS not available");
            if ((transpose_a ? A.rows() : A.cols()) != (transpose_b ? B.cols() : B.rows()))
                throw nb::value_error("blas_gemm: the inner dimensions of op(A) and op(B) differ");
            Matrix C = Matrix::Zero(transpose_a ? A.cols() : A.rows(),
                                    transpose_b ? B.rows() : B.cols());
            lapack::gemm(1.0, A, transpose_a, B, transpose_b, 0.0, C);
            return C;
        },
        nb::arg("A"),
        nb::arg("B"),
        nb::arg("transpose_a") = false,
        nb::arg("transpose_b") = false,
        "The product op(A) op(B) with BLAS dgemm, as the matrix adaptations compute it (for "
        "tests).\n\n"
        "op(X) is X, or X^T when transpose_a (transpose_b) is True. Raises ValueError when the "
        "inner dimensions of op(A) and op(B) differ, RuntimeError when BLAS is not available.");
    nb::class_<rng::Shuffler>(
        m,
        "Shuffler",
        "A random permutation of the integers in [start, stop), generated one at a time.\n\n"
        "A linear congruential generator, x <- (multiplier x + offset) mod modulus, with a "
        "random seed, offset and multiplier; values >= n are skipped. The first n values of "
        "`next` are a permutation of the range, after which it repeats. Used to draw cached "
        "samples (Sobol) in a random order.")
        .def(nb::init<size_t, size_t>(),
             nb::arg("start"),
             nb::arg("stop"),
             "A permutation of [start, stop); seed, offset and multiplier are drawn from the "
             "global generator.")
        .def(nb::init<size_t>(), nb::arg("stop"), "A permutation of [0, stop).")
        .def("next", &rng::Shuffler::next, "The next integer of the permutation.")
        .def_rw("start", &rng::Shuffler::start, "The first integer of the range.")
        .def_rw("stop", &rng::Shuffler::stop, "The end of the range (exclusive).")
        .def_rw("n", &rng::Shuffler::n, "The length of the range, stop - start.")
        .def_rw("seed",
                &rng::Shuffler::seed,
                "The state of the generator: start + seed is the value returned last (before "
                "the first call: the first value).")
        .def_rw("offset", &rng::Shuffler::offset, "The increment of the generator (odd).")
        .def_rw("multiplier",
                &rng::Shuffler::multiplier,
                "The multiplier of the generator, drawn at random with multiplier = 5 (mod 8) "
                "(1 when modulus <= 4): the full period, and not a constant stride.")
        .def_rw("modulus",
                &rng::Shuffler::modulus,
                "The modulus of the generator, the smallest power of 2 >= n.")
        .def_rw("found", &rng::Shuffler::found, "The number of integers returned so far.");

    nb::class_<rng::CachedShuffleSequence>(
        m,
        "CachedShuffleSequence",
        "A cache of points, returned in a random order (a `Shuffler` over the points).\n\n"
        "Holds n points of dimension dim, with n = max(constants.cache_min_samples, the "
        "largest power of 2 <= constants.cache_max_doubles / dim). The Sobol sampler keeps its "
        "points in one.")
        .def(nb::init<size_t>(),
             nb::arg("dim"),
             "A cache of points of dimension dim, all zero until `fill`.")
        .def("fill",
             &rng::CachedShuffleSequence::fill,
             nb::arg("c"),
             "Copy the numbers c into the cache, point after point (dim numbers each).\n\n"
             "c must not hold more than n * dim numbers.")
        .def("get_index",
             &rng::CachedShuffleSequence::get_index,
             nb::arg("index"),
             "A copy of point index of the cache (index < n, otherwise IndexError).")
        .def("next",
             static_cast<Vector (rng::CachedShuffleSequence::*)()>(
                 &rng::CachedShuffleSequence::next),
             "The next point in the random order.");
}

} // namespace modcma_python

NB_MODULE(cmaescpp, m)
{
    using namespace modcma_python;
    // The registries of Python-defined restart criteria and samplers keep Python objects alive
    // for the life of the process.
    nb::set_leak_warnings(false);

    m.doc() =
        "The C++ ModularCMAES library: a modular CMA-ES in which every module (matrix "
        "adaptation, step-size adaptation, sampling, selection, restarts, bound correction, "
        "...) is an option that combines freely with the others.\n\n"
        "Use it through `modcma.c_maes`, which re-exports these names. The classes you need "
        "most are `ModularCMAES`, `parameters.Settings` and `parameters.Modules`, with the "
        "options in `options`. Submodules: `options` (the module options), `parameters` "
        "(settings, modules, weights, statistics), `sampling`, `mutation` (step-size "
        "adaptation), `matrix_adaptation`, `selection`, `bounds`, `center`, `repelling`, "
        "`restart` (criteria and strategies) and `es` (a (1+1)-ES and a (mu, lambda)-ES); the "
        "class `constants` (global switches); `set_seed`, `build_info` and `compute_ert`.";

    define_constants(m);
    define_options(m);
    define_utils(m);
    define_population(m);
    define_samplers(m);
    define_mutation(m);
    define_restart_criteria(m);
    define_restart_strategy(m);
    define_matrix_adaptation(m);
    define_center_placement(m);
    define_repelling(m);
    define_parameters(m);
    define_bounds(m);
    define_selection(m);
    define_cmaes(m);
    define_es(m);
}
