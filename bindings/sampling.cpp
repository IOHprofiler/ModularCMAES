#include "common.hpp"

namespace modcma_python
{
namespace
{
//! A sampler that draws each coordinate from a Python function
struct PySampler : sampling::Sampler
{
    std::function<Float()> func;

    PySampler(size_t d, std::function<Float()> f) :
        Sampler::Sampler(d),
        func(std::move(f))
    {
    }

    void sample(Eigen::Ref<Vector> out) override
    {
        for (size_t j = 0; j < d; ++j)
            out(j) = func();
    }
};

//! The members every concrete transformer binds
template <typename T, typename Class> void transformer_members(Class &cls)
{
    cls.def(nb::init<const std::shared_ptr<sampling::Sampler>>(),
            nb::arg("sampler"),
            "Transform the samples of sampler, which should give uniform numbers in (0, 1).")
        .def("transform",
             &T::transform,
             nb::arg("x"),
             "The transformation of x, a vector of uniform numbers in (0, 1), as a new vector.")
        .def("__call__", &T::operator(), "One transformed sample of the sampler.")
        .def("expected_length",
             &T::expected_length,
             "The expected length E||z|| of a transformed sample (see the class), which CSA, "
             "h_sigma, threshold convergence and gradient injection use.");
}
} // namespace

void define_samplers(nb::module_ &main)
{
    using namespace sampling;

    auto m = main.def_submodule(
        "sampling",
        "The samplers that generate the vectors z of the mutations y = A z.\n\n"
        "The optimizer builds its sampler from the modules: a base sampler of uniform numbers "
        "in (0, 1)^d (`Uniform`, `Sobol` or `Halton`; `options.BaseSampler`), a transformer to "
        "the distribution of the mutations (`GaussianTransformer` and the others; "
        "`options.SampleTranformerType`), then `Orthogonal` around it with "
        "`Modules.orthogonal`, and `CachedSampler` with `constants.cache_samples` (not with "
        "SOBOL). With `Settings.fast_sampling`, `FastGaussian` replaces the default UNIFORM + "
        "GAUSSIAN. Mirrored sampling (`Modules.mirrored`) is not a sampler of the optimizer: "
        "it samples the first of each pair and negates it for the second (`Mirrored` is a "
        "standalone version). Calling a sampler returns one sample, a vector of length d. The "
        "samplers created here draw from the global generator (`set_seed`); the ones of an "
        "optimizer draw from its own.");

    nb::class_<Sampler>(m,
                        "Sampler",
                        "The base class of the samplers: generates vectors of d random numbers.")
        .def_ro("d", &Sampler::d, "The dimension of the samples.")
        .def("reset",
             &Sampler::reset,
             nb::arg("modules"),
             nb::arg("lamb"),
             "Prepare for populations of lamb individuals with these modules; called at the "
             "start and at every restart. `Orthogonal` adapts its number of samples per "
             "generation and `Mirrored` starts a new pair; most samplers do nothing.")
        .def("begin_generation",
             &Sampler::begin_generation,
             "A new generation starts; the optimizer calls it before it samples a population.\n\n"
             "`Orthogonal` starts a new block and `Mirrored` a new pair; most samplers do "
             "nothing.")
        .def("expected_length",
             &Sampler::expected_length,
             "The expected length E||z|| of a sample. The optimizer reads it when it is created "
             "(`Weights.expected_length_z`: CSA, h_sigma, gradient injection) and threshold "
             "convergence at every generation. sqrt(d) for `Uniform`, `Sobol`, `Halton`, "
             "`Gaussian` and `PySampler`; the transformers and `FastGaussian` give their own, "
             "and `Orthogonal`, `CachedSampler` and `Mirrored` that of the sampler they wrap.");

    nb::class_<PySampler, Sampler>(
        m,
        "PySampler",
        "A sampler that calls a Python function for every coordinate. It cannot be pickled "
        "or copied.")
        .def(nb::init<size_t, std::function<Float()>>(),
             nb::arg("d"),
             nb::arg("function"),
             "Samples of dimension d, each coordinate a call function() -> float.")
        .def("__call__", &Sampler::operator(), "One sample: d calls of the function.");

    nb::class_<Gaussian, Sampler>(
        m,
        "Gaussian",
        "Standard normal samples with the Box-Muller transform of mt19937 numbers.\n\n"
        "The optimizer builds its normal samples from a base sampler and a "
        "`GaussianTransformer` instead. Its `expected_length` is sqrt(d), not "
        "E||N(0, I)||.")
        .def(nb::init<size_t>(), nb::arg("d"), "Samples of dimension d.")
        .def("__call__", &Gaussian::operator(), "One sample.");

    nb::class_<FastGaussian, Sampler>(
        m,
        "FastGaussian",
        "Standard normal samples from xoshiro256++ and a ziggurat (Settings.fast_sampling).\n\n"
        "Faster than the default UNIFORM + GAUSSIAN sampler, but a different random stream.")
        .def(
            "__init__",
            // seeded from a fork of the global generator (set_seed)
            [](FastGaussian *self, const size_t d) { new (self) FastGaussian(d, *rng::fork()); },
            nb::arg("d"),
            "Samples of dimension d, seeded from the global generator (`set_seed`).")
        .def("__call__", &FastGaussian::operator(), "One sample.");

    nb::class_<Uniform, Sampler>(
        m, "Uniform", "Pseudo-random uniform samples in (0, 1)^d (mt19937); BaseSampler.UNIFORM.")
        .def(nb::init<size_t>(), nb::arg("d"), "Samples of dimension d.")
        .def("__call__", &Uniform::operator(), "One sample.");

    nb::class_<Sobol, Sampler>(
        m,
        "Sobol",
        "Points of the Sobol sequence in (0, 1)^d, in a random order; BaseSampler.SOBOL.\n\n"
        "The points are computed when the sampler is created: n of them, from index 2 of the "
        "sequence on (the first two, 0 and (1/2, ..., 1/2), are left out), with n = "
        "max(constants.cache_min_samples, the largest power of 2 <= "
        "constants.cache_max_doubles / d). They are drawn in a random order that repeats "
        "after all n; the order is the same on every platform for the same seed (#116).")
        .def(nb::init<size_t>(),
             nb::arg("d"),
             "Samples of dimension d, at most 1111 (the table of direction numbers): a larger "
             "d terminates the process.")
        .def("__call__", &Sobol::operator(), "One sample: the next point in the random order.");

    nb::class_<Halton, Sampler>(m,
                                "Halton",
                                "Points of the Halton sequence in (0, 1)^d; BaseSampler.HALTON.\n\n"
                                "Coordinate j uses the j-th prime as its base; the sequence "
                                "starts at index 6. The optimizer uses the scrambled sequence.")
        .def(nb::init<size_t, bool>(),
             nb::arg("d"),
             nb::arg("scrambled") = true,
             "Samples of dimension d; with scrambled, the digits are permuted with random "
             "permutations, one per digit position and base (drawn once, from the global "
             "generator; the same on every platform for the same seed, #116).")
        .def("__call__", &Halton::operator(), "One sample: the next point of the sequence.");

    nb::class_<Mirrored, Sampler>(
        m,
        "Mirrored",
        "Mirrored samples: a sample of the wrapped sampler, then its negation, alternately; "
        "`reset` and `begin_generation` start a new pair.\n\n"
        "A standalone sampler: the optimizer does not use it. With `Modules.mirrored` the "
        "optimizer samples the pairs of each generation itself (offspring 2i and 2i + 1 are z "
        "and -z, and a rejected pair is sampled again as a pair).")
        .def(nb::init<const std::shared_ptr<Sampler>>(),
             nb::arg("sampler"),
             "Mirror the samples of sampler.")
        .def("__call__", &Mirrored::operator(), "One sample: a new one, or the last negated.");

    nb::class_<CachedSampler, Sampler>(
        m,
        "CachedSampler",
        "A sampler that cycles through a fixed set of points.\n\n"
        "The optimizer wraps its sampler in one when `constants.cache_samples` is set (off by "
        "default), except with SOBOL, which keeps its points itself. A restart does not empty "
        "the cache.")
        .def(nb::init<const std::shared_ptr<Sampler>>(),
             nb::arg("sampler"),
             "Cache the first n_samples samples of sampler and then repeat them; n_samples = "
             "max(constants.cache_min_samples, the largest power of 2 <= "
             "constants.cache_max_doubles / d).")
        .def(
            "__init__",
            [](CachedSampler *self, const std::vector<std::vector<Float>> &cache, bool transform_ppf) {
                std::vector<Vector> points;
                points.reserve(cache.size());
                for (const auto &point : cache)
                    points.push_back(to_vector(point));
                new (self) CachedSampler(points, transform_ppf);
            },
            nb::arg("cache"),
            nb::arg("transform_ppf") = false,
            "Cycle through the given points (a non-empty list of equally long vectors).\n\n"
            "With transform_ppf, every coordinate is first mapped through the inverse normal "
            "CDF, turning uniform numbers in (0, 1) into standard normal ones.")
        .def("__call__", &CachedSampler::operator(), "One sample: the next point of the cycle.")
        .def_ro("index",
                &CachedSampler::index,
                "A counter of the samples: the next sample is point index mod n_samples.")
        .def_ro("n_samples", &CachedSampler::n_samples, "The number of points in the cycle.")
        .def_ro("cache",
                &CachedSampler::cache,
                "The points (with a wrapped sampler: filled as they are drawn), a list of numpy "
                "views of them: writing to one changes the sampler's point. The views stay valid "
                "while the sampler exists (the points are never reallocated; #176).");

    nb::class_<Orthogonal, Sampler>(
        m,
        "Orthogonal",
        "Orthogonalized samples (Modules.orthogonal; Wang, Emmerich and Baeck 2019).\n\n"
        "The n_samples samples of a generation are orthogonalized in blocks: the first "
        "min(n_samples, d), then blocks of at most d (more than d vectors cannot be "
        "orthogonal). A block is drawn from the wrapped sampler when it is needed, "
        "orthonormalized as by the Gram-Schmidt process (a Householder QR decomposition with "
        "the signs corrected so that diag(R) > 0: the first sample keeps its direction), and "
        "each vector gets the length of the sample it came from. `begin_generation` starts a "
        "new generation; without it, every n_samples consecutive samples are one. Samples "
        "that replace rejected ones count towards the generation. Wang et al. draw the "
        "samples beyond the first d at random; here they are orthogonalized in further "
        "blocks.")
        .def(nb::init<const std::shared_ptr<Sampler>, size_t>(),
             nb::arg("sampler"),
             nb::arg("n_samples"),
             "Orthogonalize the samples of sampler, n_samples per generation. The optimizer "
             "uses n_samples = lambda, or ceil(lambda / 2) with mirrored sampling.")
        .def("__call__", &Orthogonal::operator(), "One sample: the next of the block.");

    nb::class_<SampleTransformer, Sampler>(
        m,
        "SampleTransformer",
        "The base class of the transformers: map the uniform numbers in (0, 1) of a base "
        "sampler, coordinate by coordinate, to another distribution "
        "(options.SampleTranformerType).")
        .def("raw",
             &SampleTransformer::raw,
             "One sample of the wrapped sampler, without the transformation.");

    auto identity = nb::class_<IdentityTransformer, SampleTransformer>(
        m,
        "IdentityTransformer",
        "No transformation: the uniform numbers of the base sampler as they are "
        "(SampleTranformerType.NONE), the raw base sampler, for testing.\n\n"
        "Its expected length is E||u|| of u uniform in (0, 1)^d.");
    transformer_members<IdentityTransformer>(identity);
    auto gaussian = nb::class_<GaussianTransformer, SampleTransformer>(
        m,
        "GaussianTransformer",
        "Standard normal numbers: the inverse normal CDF of each uniform number, "
        "or the Box-Muller transform of pairs with `constants.use_box_muller` "
        "(SampleTranformerType.GAUSSIAN, the default).\n\n"
        "Its expected length is that of a d-dimensional standard normal vector, in Hansen's "
        "approximation sqrt(d) (1 - 1/(4 d) + 1/(21 d^2)) (Hansen 2016; `FastGaussian` uses "
        "the same).");
    transformer_members<GaussianTransformer>(gaussian);
    auto uniform = nb::class_<UniformScaler, SampleTransformer>(
        m,
        "UniformScaler",
        "Uniform numbers in (-sqrt(3), sqrt(3)), unit variance: -sqrt(3) + 2 sqrt(3) u "
        "(SampleTranformerType.SCALED_UNIFORM).\n\n"
        "Its expected length is E||z|| of this distribution in d dimensions, computed "
        "numerically (it was sqrt(d) before 2.0).");
    transformer_members<UniformScaler>(uniform);
    auto laplace = nb::class_<LaplaceTransformer, SampleTransformer>(
        m,
        "LaplaceTransformer",
        "Laplace numbers with scale b = 1/sqrt(2), unit variance: b ln(2u) for u < 0.5, "
        "-b ln(2 (1 - u)) otherwise (SampleTranformerType.LAPLACE).\n\n"
        "Its expected length is E||z|| of this distribution in d dimensions, computed "
        "numerically (it was sqrt(d) before 2.0).");
    transformer_members<LaplaceTransformer>(laplace);
    auto logistic = nb::class_<LogisticTransformer, SampleTransformer>(
        m,
        "LogisticTransformer",
        "Logistic numbers with scale s = sqrt(3)/pi, unit variance: s ln(u / (1 - u)) "
        "(SampleTranformerType.LOGISTIC).\n\n"
        "Its expected length is E||z|| of this distribution in d dimensions, computed "
        "numerically (it was sqrt(d) before 2.0).");
    transformer_members<LogisticTransformer>(logistic);
    auto cauchy = nb::class_<CauchyTransformer, SampleTransformer>(
        m,
        "CauchyTransformer",
        "Standard Cauchy numbers: tan(pi (u - 0.5)) (SampleTranformerType.CAUCHY).\n\n"
        "The length of a Cauchy vector has no expectation; its expected length is an "
        "empirical approximation of the median, d (1 + 0.18 tanh(d / 4.07)).");
    transformer_members<CauchyTransformer>(cauchy);
    auto weibull = nb::class_<DoubleWeibullTransformer, SampleTransformer>(
        m,
        "DoubleWeibullTransformer",
        "Double Weibull numbers with shape 2 and scale 1, unit variance: -sqrt(-ln(2u)) for "
        "u < 0.5, sqrt(-ln(2 (1 - u))) otherwise (SampleTranformerType.DOUBLE_WEIBULL).\n\n"
        "Its expected length is Gamma(d + 1/2) / Gamma(d), exactly: each z_i^2 is exponential "
        "with mean 1, so ||z||^2 is Gamma(d, 1)-distributed (it was sqrt(d) before 2.0).");
    transformer_members<DoubleWeibullTransformer>(weibull);

    // Pickling and copying (#91): every sampler of the library; not PySampler, which calls a
    // Python function
    def_pickle<Gaussian, Sampler>();
    def_pickle<FastGaussian, Sampler>();
    def_pickle<Uniform, Sampler>();
    def_pickle<Sobol, Sampler>();
    def_pickle<Halton, Sampler>();
    def_pickle<Mirrored, Sampler>();
    def_pickle<CachedSampler, Sampler>();
    def_pickle<Orthogonal, Sampler>();
    def_pickle<IdentityTransformer, Sampler>();
    def_pickle<GaussianTransformer, Sampler>();
    def_pickle<UniformScaler, Sampler>();
    def_pickle<LaplaceTransformer, Sampler>();
    def_pickle<LogisticTransformer, Sampler>();
    def_pickle<CauchyTransformer, Sampler>();
    def_pickle<DoubleWeibullTransformer, Sampler>();
}
} // namespace modcma_python
