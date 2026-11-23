#include "common.hpp"

#include <nanobind/ndarray.h>

namespace modcma_python
{
namespace
{
template <typename RNG> Float random_double()
{
    static RNG gen;
    return gen(rng::GENERATOR);
}

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
} // namespace

void define_constants(nb::module_ &m)
{
    nb::class_<constants_w>(m, "constants")
        .def_prop_rw_static(
            "cache_max_doubles",
            [](nb::handle) { return constants::cache_max_doubles; },
            [](nb::handle, size_t a) { constants::cache_max_doubles = a; })
        .def_prop_rw_static(
            "cache_min_samples",
            [](nb::handle) { return constants::cache_min_samples; },
            [](nb::handle, size_t a) { constants::cache_min_samples = a; })
        .def_prop_rw_static(
            "cache_samples",
            [](nb::handle) { return constants::cache_samples; },
            [](nb::handle, nb::handle a) { constants::cache_samples = to_bool(a); })
        .def_prop_rw_static(
            "clip_sigma",
            [](nb::handle) { return constants::clip_sigma; },
            [](nb::handle, nb::handle a) { constants::clip_sigma = to_bool(a); })
        .def_prop_rw_static(
            "use_box_muller",
            [](nb::handle) { return constants::use_box_muller; },
            [](nb::handle, nb::handle a) { constants::use_box_muller = to_bool(a); });
}

void define_options(nb::module_ &main)
{
    auto m = main.def_submodule("options");
    using namespace parameters;
    nb::enum_<RecombinationWeights>(m, "RecombinationWeights")
        .value("DEFAULT", RecombinationWeights::DEFAULT)
        .value("EQUAL", RecombinationWeights::EQUAL)
        .value("EXPONENTIAL", RecombinationWeights::EXPONENTIAL);

    nb::enum_<BaseSampler>(m, "BaseSampler")
        .value("UNIFORM", BaseSampler::UNIFORM)
        .value("SOBOL", BaseSampler::SOBOL)
        .value("HALTON", BaseSampler::HALTON);

    nb::enum_<SampleTranformerType>(m, "SampleTranformerType")
        .value("NONE", SampleTranformerType::NONE)
        .value("GAUSSIAN", SampleTranformerType::GAUSSIAN)
        .value("SCALED_UNIFORM", SampleTranformerType::SCALED_UNIFORM)
        .value("LAPLACE", SampleTranformerType::LAPLACE)
        .value("LOGISTIC", SampleTranformerType::LOGISTIC)
        .value("CAUCHY", SampleTranformerType::CAUCHY)
        .value("DOUBLE_WEIBULL", SampleTranformerType::DOUBLE_WEIBULL);

    nb::enum_<Mirror>(m, "Mirror")
        .value("NONE", Mirror::NONE)
        .value("MIRRORED", Mirror::MIRRORED)
        .value("PAIRWISE", Mirror::PAIRWISE);

    nb::enum_<StepSizeAdaptation>(m, "StepSizeAdaptation")
        .value("CSA", StepSizeAdaptation::CSA)
        .value("TPA", StepSizeAdaptation::TPA)
        .value("MSR", StepSizeAdaptation::MSR)
        .value("XNES", StepSizeAdaptation::XNES)
        .value("MXNES", StepSizeAdaptation::MXNES)
        .value("LPXNES", StepSizeAdaptation::LPXNES)
        .value("PSR", StepSizeAdaptation::PSR)
        .value("SR", StepSizeAdaptation::SR)
        .value("SA", StepSizeAdaptation::SA);

    nb::enum_<CorrectionMethod>(m, "CorrectionMethod")
        .value("NONE", CorrectionMethod::NONE)
        .value("MIRROR", CorrectionMethod::MIRROR)
        .value("COTN", CorrectionMethod::COTN)
        .value("UNIFORM_RESAMPLE", CorrectionMethod::UNIFORM_RESAMPLE)
        .value("SATURATE", CorrectionMethod::SATURATE)
        .value("TOROIDAL", CorrectionMethod::TOROIDAL)
        .value("RESAMPLE", CorrectionMethod::RESAMPLE);

    nb::enum_<RestartStrategyType>(m, "RestartStrategy")
        .value("NONE", RestartStrategyType::NONE)
        .value("STOP", RestartStrategyType::STOP)
        .value("RESTART", RestartStrategyType::RESTART)
        .value("IPOP", RestartStrategyType::IPOP)
        .value("BIPOP", RestartStrategyType::BIPOP);

    nb::enum_<MatrixAdaptationType>(m, "MatrixAdaptationType")
        .value("COVARIANCE", MatrixAdaptationType::COVARIANCE)
        .value("NONE", MatrixAdaptationType::NONE)
        .value("MATRIX", MatrixAdaptationType::MATRIX)
        .value("SEPARABLE", MatrixAdaptationType::SEPARABLE)
        .value("CHOLESKY", MatrixAdaptationType::CHOLESKY)
        .value("CMSA", MatrixAdaptationType::CMSA)
        .value("COVARIANCE_NO_EIGV", MatrixAdaptationType::COVARIANCE_NO_EIGV)
        .value("NATURAL_GRADIENT", MatrixAdaptationType::NATURAL_GRADIENT);

    nb::enum_<CenterPlacement>(m, "CenterPlacement")
        .value("X0", CenterPlacement::X0)
        .value("ZERO", CenterPlacement::ZERO)
        .value("UNIFORM", CenterPlacement::UNIFORM)
        .value("CENTER", CenterPlacement::CENTER)
        .value("NOVELTY_WEIGHTED", CenterPlacement::NOVELTY_WEIGHTED)
        .value("MAXIMIN_TABOO", CenterPlacement::MAXIMIN_TABOO);

    nb::enum_<RepellingType>(m, "RepellingType")
        .value("NONE", RepellingType::NONE)
        .value("COVERAGE", RepellingType::COVERAGE)
        .value("ADAPTIVE", RepellingType::ADAPTIVE);

    nb::enum_<RepellingMetric>(m, "RepellingMetric")
        .value("EUCLIDEAN", RepellingMetric::EUCLIDEAN)
        .value("CURRENT", RepellingMetric::CURRENT)
        .value("ARCHIVED", RepellingMetric::ARCHIVED);
}

void define_utils(nb::module_ &main)
{
    auto m = main.def_submodule("utils");
    m.def(
        "build_info",
        []() {
            return std::map<std::string, std::string>{
                {"compiler", compiler_version()},
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
                {"fma", "yes"},
#else
                {"fma", "no"},
#endif
            };
        },
        "How the extension was built: compiler, and whether it may use fused multiply-add "
        "instructions (which change the last bits of results)");
    m.def("cdf", &cdf, nb::arg("x"));
    m.def("ppf", &ppf, nb::arg("x"));
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
        "Write the next point of the Sobol sequence into quasi and return the next seed");
    m.def("compute_ert", &utils::compute_ert, nb::arg("running_times"), nb::arg("budget"));
    m.def("set_seed", &rng::set_seed, nb::arg("seed"), "Set the random seed");
    m.def("random_uniform",
          &random_double<rng::uniform<Float>>,
          "Generate a uniform random number in (0, 1)");
    m.def("random_normal",
          &random_double<rng::normal<Float>>,
          "Generate a standard normal random number");

    nb::class_<rng::Shuffler>(m, "Shuffler")
        .def(nb::init<size_t, size_t>(), nb::arg("start"), nb::arg("stop"))
        .def(nb::init<size_t>(), nb::arg("stop"))
        .def("next", &rng::Shuffler::next)
        .def_rw("start", &rng::Shuffler::start)
        .def_rw("stop", &rng::Shuffler::stop)
        .def_rw("n", &rng::Shuffler::n)
        .def_rw("seed", &rng::Shuffler::seed)
        .def_rw("offset", &rng::Shuffler::offset)
        .def_rw("multiplier", &rng::Shuffler::multiplier)
        .def_rw("modulus", &rng::Shuffler::modulus)
        .def_rw("found", &rng::Shuffler::found);

    nb::class_<rng::CachedShuffleSequence>(m, "CachedShuffleSequence")
        .def(nb::init<size_t>(), nb::arg("dim"))
        .def("fill", &rng::CachedShuffleSequence::fill, nb::arg("c"))
        .def("get_index", &rng::CachedShuffleSequence::get_index, nb::arg("index"))
        .def("next", &rng::CachedShuffleSequence::next);
}
} // namespace modcma_python

NB_MODULE(cmaescpp, m)
{
    using namespace modcma_python;
    // The registries of Python-defined restart criteria and samplers keep Python objects alive
    // for the life of the process.
    nb::set_leak_warnings(false);

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
