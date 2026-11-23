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
    cls.def(nb::init<const std::shared_ptr<sampling::Sampler>>(), nb::arg("sampler"))
        .def("transform", &T::transform, nb::arg("x"))
        .def("__call__", &T::operator())
        .def("expected_length", &T::expected_length);
}
} // namespace

void define_samplers(nb::module_ &main)
{
    using namespace sampling;

    auto m = main.def_submodule("sampling");

    nb::class_<Sampler>(m, "Sampler")
        .def_ro("d", &Sampler::d)
        .def("reset", &Sampler::reset, nb::arg("modules"), nb::arg("lamb"))
        .def("expected_length", &Sampler::expected_length);

    nb::class_<PySampler, Sampler>(m, "PySampler")
        .def(nb::init<size_t, std::function<Float()>>(), nb::arg("d"), nb::arg("function"))
        .def("__call__", &Sampler::operator());

    nb::class_<Gaussian, Sampler>(m, "Gaussian")
        .def(nb::init<size_t>(), nb::arg("d"))
        .def("__call__", &Gaussian::operator());

    nb::class_<FastGaussian, Sampler>(
        m,
        "FastGaussian",
        "Standard normal samples from xoshiro256++ and a ziggurat (Settings.fast_sampling)")
        .def(
            "__init__",
            // seeded from a fork of the global generator (utils.set_seed)
            [](FastGaussian *self, const size_t d) { new (self) FastGaussian(d, *rng::fork()); },
            nb::arg("d"))
        .def("__call__", &FastGaussian::operator());

    nb::class_<Uniform, Sampler>(m, "Uniform")
        .def(nb::init<size_t>(), nb::arg("d"))
        .def("__call__", &Uniform::operator());

    nb::class_<Sobol, Sampler>(m, "Sobol")
        .def(nb::init<size_t>(), nb::arg("d"))
        .def_ro("cache", &Sobol::cache)
        .def("__call__", &Sobol::operator());

    nb::class_<Halton, Sampler>(m, "Halton")
        .def(nb::init<size_t, bool>(), nb::arg("d"), nb::arg("scrambled") = true)
        .def("__call__", &Halton::operator());

    nb::class_<Mirrored, Sampler>(m, "Mirrored")
        .def(nb::init<const std::shared_ptr<Sampler>>(), nb::arg("sampler"))
        .def("__call__", &Mirrored::operator());

    nb::class_<CachedSampler, Sampler>(m, "CachedSampler")
        .def(nb::init<const std::shared_ptr<Sampler>>(), nb::arg("sampler"))
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
            nb::arg("transform_ppf") = false)
        .def("__call__", &CachedSampler::operator())
        .def_ro("index", &CachedSampler::index)
        .def_ro("n_samples", &CachedSampler::n_samples)
        .def_ro("cache", &CachedSampler::cache);

    nb::class_<Orthogonal, Sampler>(m, "Orthogonal")
        .def(nb::init<const std::shared_ptr<Sampler>, size_t>(),
             nb::arg("sampler"),
             nb::arg("n_samples"))
        .def("__call__", &Orthogonal::operator());

    nb::class_<SampleTransformer, Sampler>(m, "SampleTransformer").def("raw", &SampleTransformer::raw);

    auto identity = nb::class_<IdentityTransformer, SampleTransformer>(m, "IdentityTransformer");
    transformer_members<IdentityTransformer>(identity);
    auto gaussian = nb::class_<GaussianTransformer, SampleTransformer>(m, "GaussianTransformer");
    transformer_members<GaussianTransformer>(gaussian);
    auto uniform = nb::class_<UniformScaler, SampleTransformer>(m, "UniformScaler");
    transformer_members<UniformScaler>(uniform);
    auto laplace = nb::class_<LaplaceTransformer, SampleTransformer>(m, "LaplaceTransformer");
    transformer_members<LaplaceTransformer>(laplace);
    auto logistic = nb::class_<LogisticTransformer, SampleTransformer>(m, "LogisticTransformer");
    transformer_members<LogisticTransformer>(logistic);
    auto cauchy = nb::class_<CauchyTransformer, SampleTransformer>(m, "CauchyTransformer");
    transformer_members<CauchyTransformer>(cauchy);
    auto weibull =
        nb::class_<DoubleWeibullTransformer, SampleTransformer>(m, "DoubleWeibullTransformer");
    transformer_members<DoubleWeibullTransformer>(weibull);
}
} // namespace modcma_python
