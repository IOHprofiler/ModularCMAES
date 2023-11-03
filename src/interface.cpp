#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/c_maes.hpp"
#include "c_maes.hpp"
#include "to_string.hpp"

namespace py = pybind11;

template <typename RNG>
double random_double()
{
    static RNG gen;
    return gen(rng::GENERATOR);
}

void define_options(py::module &main)
{
    auto m = main.def_submodule("options");
    using namespace parameters;
    py::enum_<RecombinationWeights>(m, "RecombinationWeights")
        .value("DEFAULT", parameters::RecombinationWeights::DEFAULT)
        .value("EQUAL", parameters::RecombinationWeights::EQUAL)
        .value("HALF_POWER_LAMBDA", parameters::RecombinationWeights::HALF_POWER_LAMBDA)
        .export_values();

    py::enum_<BaseSampler>(m, "BaseSampler")
        .value("GAUSSIAN", BaseSampler::GAUSSIAN)
        .value("SOBOL", BaseSampler::SOBOL)
        .value("HALTON", BaseSampler::HALTON)
        .export_values();

    py::enum_<Mirror>(m, "Mirror")
        .value("NONE", Mirror::NONE)
        .value("MIRRORED", Mirror::MIRRORED)
        .value("PAIRWISE", Mirror::PAIRWISE)
        .export_values();

    py::enum_<StepSizeAdaptation>(m, "StepSizeAdaptation")
        .value("CSA", StepSizeAdaptation::CSA)
        .value("TPA", StepSizeAdaptation::TPA)
        .value("MSR", StepSizeAdaptation::MSR)
        .value("XNES", StepSizeAdaptation::XNES)
        .value("MXNES", StepSizeAdaptation::MXNES)
        .value("LPXNES", StepSizeAdaptation::LPXNES)
        .value("PSR", StepSizeAdaptation::PSR)
        .export_values();

    py::enum_<CorrectionMethod>(m, "CorrectionMethod")
        .value("NONE", CorrectionMethod::NONE)
        .value("COUNT", CorrectionMethod::COUNT)
        .value("MIRROR", CorrectionMethod::MIRROR)
        .value("COTN", CorrectionMethod::COTN)
        .value("UNIFORM_RESAMPLE", CorrectionMethod::UNIFORM_RESAMPLE)
        .value("SATURATE", CorrectionMethod::SATURATE)
        .value("TOROIDAL", CorrectionMethod::TOROIDAL)
        .export_values();

    py::enum_<RestartStrategyType>(m, "RestartStrategy")
        .value("NONE", RestartStrategyType::NONE)
        .value("STOP", RestartStrategyType::STOP)
        .value("RESTART", RestartStrategyType::RESTART)
        .value("IPOP", RestartStrategyType::IPOP)
        .value("BIPOP", RestartStrategyType::BIPOP)
        .export_values();

    py::enum_<MatrixAdaptationType>(m, "MatrixAdaptationType")
        .value("COVARIANCE", MatrixAdaptationType::COVARIANCE)
        .value("NONE", MatrixAdaptationType::NONE)
        .value("MATRIX", MatrixAdaptationType::MATRIX)
        .export_values();
}

struct PySampler : sampling::Sampler
{
    std::function<double()> func;

    PySampler(size_t d, std::function<double()> f) : Sampler::Sampler(d), func(f) {}

    Vector operator()() override
    {
        Vector res(d);
        for (size_t j = 0; j < d; ++j)
            res(j) = func();
        return res;
    };
};

void define_samplers(py::module &main)
{
    using namespace sampling;

    auto m = main.def_submodule("sampling");

    py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Sampler")
        .def_readonly("d", &Sampler::d);

    py::class_<PySampler, Sampler, std::shared_ptr<PySampler>>(m, "PySampler")
        .def(py::init<size_t, std::function<double()>>(), py::arg("d"), py::arg("function"))
        .def("__call__", &PySampler::operator());

    py::class_<Gaussian, Sampler, std::shared_ptr<Gaussian>>(m, "Gaussian")
        .def(py::init<size_t>(), py::arg("d"))
        .def("__call__", &Gaussian::operator());

    py::class_<Uniform, Sampler, std::shared_ptr<Uniform>>(m, "Uniform")
        .def(py::init<size_t>(), py::arg("d"))
        .def("__call__", &Uniform::operator());

    py::class_<Sobol, Sampler, std::shared_ptr<Sobol>>(m, "Sobol")
        .def(py::init<size_t>(), py::arg("d"))
        .def("__call__", &Sobol::operator());

    py::class_<Halton, Sampler, std::shared_ptr<Halton>>(m, "Halton")
        .def(py::init<size_t, size_t>(), py::arg("d"), py::arg("i") = 1)
        .def("__call__", &Halton::operator());

    py::class_<Mirrored, Sampler, std::shared_ptr<Mirrored>>(m, "Mirrored")
        .def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
        .def("__call__", &Mirrored::operator());

    py::class_<Orthogonal, Sampler, std::shared_ptr<Orthogonal>>(m, "Orthogonal")
        .def(py::init<const std::shared_ptr<Sampler>, size_t>(),
             py::arg("sampler"), py::arg("n_samples"))
        .def("__call__", &Orthogonal::operator());
}

void define_utils(py::module &main)
{
    auto m = main.def_submodule("utils");
    m.def("compute_ert", &utils::compute_ert, py::arg("running_times"), py::arg("budget"));
    m.def("set_seed", &rng::set_seed, py::arg("seed"), "Set the random seed");
    m.def("random_uniform", &random_double<rng::uniform<double>>, "Generate a uniform random number in [-1, 1]");
    m.def("random_normal", &random_double<rng::normal<double>>, "Generate a standard normal random number");
}

void define_selection(py::module &main)
{
    auto m = main.def_submodule("selection");
    using namespace selection;
    py::class_<Elitsm, std::shared_ptr<Elitsm>>(m, "Elitsm")
        .def(py::init<>())
        .def("__call__", &Elitsm::operator(), py::arg("parameters"));

    py::class_<NoElitsm, Elitsm, std::shared_ptr<NoElitsm>>(m, "NoElitsm")
        .def(py::init<>())
        .def("__call__", &NoElitsm::operator(), py::arg("parameters"));

    py::class_<Pairwise, std::shared_ptr<Pairwise>>(m, "Pairwise")
        .def(py::init<>())
        .def("__call__", &Pairwise::operator(), py::arg("parameters"));

    py::class_<NoPairwise, Pairwise, std::shared_ptr<NoPairwise>>(m, "NoPairwise")
        .def(py::init<>())
        .def("__call__", &NoPairwise::operator(), py::arg("parameters"));

    py::class_<Strategy, std::shared_ptr<Strategy>>(m, "Strategy")
        .def(py::init<parameters::Modules>(), py::arg("modules"))
        .def("select", &Strategy::select, py::arg("parameters"))
        .def_readwrite("pairwise", &Strategy::pairwise)
        .def_readwrite("elitsm", &Strategy::elitsm);
}

void define_matrix_adaptation(py::module &main)
{
    using namespace matrix_adaptation;
    auto m = main.def_submodule("matrix_adaptation");
    py::class_<Adaptation, std::shared_ptr<Adaptation>>(m, "Adaptation")
        .def_readwrite("m", &Adaptation::m)
        .def_readwrite("m_old", &Adaptation::m_old)
        .def_readwrite("dm", &Adaptation::dm)
        .def_readwrite("ps", &Adaptation::ps)
        .def_readwrite("dd", &Adaptation::dd)
        .def_readwrite("chiN", &Adaptation::chiN)
        .def("adapt_evolution_paths", &Adaptation::adapt_evolution_paths,
             py::arg("pop"),
             py::arg("weights"),
             py::arg("mutation"),
             py::arg("stats"),
             py::arg("mu"),
             py::arg("lamb"))
        .def("adapt_matrix", &Adaptation::adapt_matrix,
             py::arg("weights"),
             py::arg("modules"),
             py::arg("population"),
             py::arg("mu"),
             py::arg("settings"))
        .def("restart", &Adaptation::restart, py::arg("settings"))
        .def("scale_mutation_steps", &Adaptation::scale_mutation_steps, py::arg("pop"))
        .def("__repr__", [](Adaptation &dyn)
             {
            std::stringstream ss;
            ss << std::boolalpha;
            ss << "<Adaptation";
            ss << " m: " << dyn.m.transpose();
            ss << " m_old: " << dyn.m_old.transpose();
            ss << " dm: " << dyn.dm.transpose();
            ss << " ps: " << dyn.ps.transpose();
            ss << " dd: " << dyn.dd;
            ss << " chiN: " << dyn.chiN;
            ss << ">";
            return ss.str(); });

    py::class_<CovarianceAdaptation, Adaptation, std::shared_ptr<CovarianceAdaptation>>(m, "CovarianceAdaptation")
        .def(py::init<size_t, Vector>(), py::arg("dimension"), py::arg("x0"))
        .def_readwrite("pc", &CovarianceAdaptation::pc)
        .def_readwrite("d", &CovarianceAdaptation::d)
        .def_readwrite("B", &CovarianceAdaptation::B)
        .def_readwrite("C", &CovarianceAdaptation::C)
        .def_readwrite("inv_root_C", &CovarianceAdaptation::inv_root_C)
        .def_readwrite("hs", &CovarianceAdaptation::hs)
        .def("adapt_covariance_matrix", &CovarianceAdaptation::adapt_covariance_matrix,
             py::arg("weights"),
             py::arg("modules"),
             py::arg("population"),
             py::arg("mu"))
        .def("perform_eigendecomposition", &CovarianceAdaptation::perform_eigendecomposition, py::arg("stats"))
        .def("__repr__", [](CovarianceAdaptation &dyn)
             {
            std::stringstream ss;
            ss << std::boolalpha;
            ss << "<CovarianceAdaptation";
            ss << " m: " << dyn.m.transpose();
            ss << " m_old: " << dyn.m_old.transpose();
            ss << " dm: " << dyn.dm.transpose();
            ss << " pc: " << dyn.pc.transpose();
            ss << " ps: " << dyn.ps.transpose();
            ss << " d: " << dyn.d.transpose();
            ss << " B: " << dyn.B;
            ss << " C: " << dyn.C;
            ss << " inv_root_C: " << dyn.inv_root_C;
            ss << " dd: " << dyn.dd;
            ss << " chiN: " << dyn.chiN;
            ss << " hs: " << dyn.hs;
            ss << ">";
            return ss.str(); });

    py::class_<MatrixAdaptation, Adaptation, std::shared_ptr<MatrixAdaptation>>(m, "MatrixAdaptation")
        .def(py::init<size_t, Vector>(), py::arg("dimension"), py::arg("x0"))
        .def_readwrite("M", &MatrixAdaptation::M)
        .def("__repr__", [](MatrixAdaptation &dyn)
             {
            std::stringstream ss;
            ss << std::boolalpha;
            ss << "<MatrixAdaptation";
            ss << " m: " << dyn.m.transpose();
            ss << " m_old: " << dyn.m_old.transpose();
            ss << " dm: " << dyn.dm.transpose();
            ss << " ps: " << dyn.ps.transpose();
            ss << " M: " << dyn.M;
            ss << " dd: " << dyn.dd;
            ss << " chiN: " << dyn.chiN;
            ss << ">";
            return ss.str(); });

    py::class_<None, Adaptation, std::shared_ptr<None>>(m, "NoAdaptation")
        .def(py::init<size_t, Vector>(), py::arg("dimension"), py::arg("x0"))
        .def("__repr__", [](None &dyn)
             {
            std::stringstream ss;
            ss << std::boolalpha;
            ss << "<NoAdaptation";
            ss << " m: " << dyn.m.transpose();
            ss << " m_old: " << dyn.m_old.transpose();
            ss << " dm: " << dyn.dm.transpose();
            ss << " ps: " << dyn.ps.transpose();
            ss << " dd: " << dyn.dd;
            ss << " chiN: " << dyn.chiN;
            ss << ">";
            return ss.str(); });
}

void define_parameters(py::module &main)
{
    auto m = main.def_submodule("parameters");
    using namespace parameters;

    py::class_<Modules>(m, "Modules")
        .def(py::init<>())
        .def_readwrite("elitist", &Modules::elitist)
        .def_readwrite("active", &Modules::active)
        .def_readwrite("orthogonal", &Modules::orthogonal)
        .def_readwrite("sequential_selection", &Modules::sequential_selection)
        .def_readwrite("threshold_convergence", &Modules::threshold_convergence)
        .def_readwrite("sample_sigma", &Modules::sample_sigma)
        .def_readwrite("weights", &Modules::weights)
        .def_readwrite("sampler", &Modules::sampler)
        .def_readwrite("mirrored", &Modules::mirrored)
        .def_readwrite("ssa", &Modules::ssa)
        .def_readwrite("bound_correction", &Modules::bound_correction)
        .def_readwrite("restart_strategy", &Modules::restart_strategy)
        .def_readwrite("matrix_adaptation", &Modules::matrix_adaptation)
        .def("__repr__", [](Modules &mod)
             { return to_string(mod); });

    py::class_<Stats>(m, "Stats")
        .def(py::init<>())
        .def_readwrite("t", &Stats::t)
        .def_readwrite("evaluations", &Stats::evaluations)
        .def_readwrite("xopt", &Stats::xopt)
        .def_readwrite("fopt", &Stats::fopt)
        .def("__repr__", [](Stats &stats)
             {
            std::stringstream ss;
            ss << std::boolalpha;
            ss << "<Stats";
            ss << " t: " << stats.t;
            ss << " evaluations: " << stats.evaluations;
            ss << " xopt: " << stats.xopt.transpose();
            ss << " fopt: " << stats.fopt;
            ss << ">";
            return ss.str(); });

    py::class_<Weights>(m, "Weights")
        .def(
            py::init<size_t, size_t, size_t, Settings>(),
            py::arg("dimension"),
            py::arg("mu0"),
            py::arg("lambda0"),
            py::arg("modules"))
        .def_readwrite("mueff", &Weights::mueff)
        .def_readwrite("mueff_neg", &Weights::mueff_neg)
        .def_readwrite("c1", &Weights::c1)
        .def_readwrite("cmu", &Weights::cmu)
        .def_readwrite("cc", &Weights::cc)
        .def_readwrite("weights", &Weights::weights)
        .def_readwrite("positive", &Weights::positive)
        .def_readwrite("negative", &Weights::negative)
        .def("__repr__", [](Weights &weights)
             {
            std::stringstream ss;
            ss << std::boolalpha;
            ss << "<Weights";
            ss << " mueff: " << weights.mueff;
            ss << " mueff_neg: " << weights.mueff_neg;
            ss << " c1: " << weights.c1;
            ss << " cmu: " << weights.cmu;
            ss << " cc: " << weights.cc;
            ss << " weights: " << weights.weights.transpose();
            ss << " positive: " << weights.positive.transpose();
            ss << " negative: " << weights.negative.transpose();
            ss << ">";
            return ss.str(); });

    py::class_<Settings, std::shared_ptr<Settings>>(m, "Settings")
        .def(py::init<size_t, std::optional<Modules>, std::optional<double>, size_to, size_to, std::optional<double>,
                      std::optional<size_t>, std::optional<size_t>, std::optional<Vector>,
                      std::optional<Vector>, std::optional<Vector>,
                      std::optional<double>, std::optional<double>, std::optional<double>,
                      std::optional<double>, bool>(),
             py::arg("dim"),
             py::arg("modules") = std::nullopt,
             py::arg("target") = std::nullopt,
             py::arg("max_generations") = std::nullopt,
             py::arg("budget") = std::nullopt,
             py::arg("sigma0") = std::nullopt,
             py::arg("lambda0") = std::nullopt,
             py::arg("mu0") = std::nullopt,
             py::arg("x0") = std::nullopt,
             py::arg("lb") = std::nullopt,
             py::arg("ub") = std::nullopt,
             py::arg("cs") = std::nullopt,
             py::arg("cc") = std::nullopt,
             py::arg("cmu") = std::nullopt,
             py::arg("c1") = std::nullopt,
             py::arg("verbose") = false)
        .def_readwrite("dim", &Settings::dim)
        .def_readwrite("modules", &Settings::modules)
        .def_readwrite("target", &Settings::target)
        .def_readwrite("max_generations", &Settings::max_generations)
        .def_readwrite("budget", &Settings::budget)
        .def_readwrite("sigma0", &Settings::sigma0)
        .def_readwrite("lambda0", &Settings::lambda0)
        .def_readwrite("mu0", &Settings::mu0)
        .def_readwrite("x0", &Settings::x0)
        .def_readwrite("lb", &Settings::lb)
        .def_readwrite("ub", &Settings::ub)
        .def_readwrite("cs", &Settings::cs)
        .def_readwrite("cc", &Settings::cc)
        .def_readwrite("cmu", &Settings::cmu)
        .def_readwrite("c1", &Settings::c1)
        .def_readwrite("verbose", &Settings::verbose)
        .def("__repr__", [](Settings &settings)
             {
            std::stringstream ss;
            ss << std::boolalpha;
            ss << "<Settings";
            ss << " dim: " << settings.dim;
            ss << " modules: " << to_string(settings.modules);
            ss << " target: " << to_string(settings.target);
            ss << " max_generations: " << to_string(settings.max_generations);
            ss << " budget: " << settings.budget;
            ss << " sigma0: " << settings.sigma0;
            ss << " lambda0: " << settings.lambda0;
            ss << " mu0: " << settings.mu0;
            ss << " x0: " << to_string(settings.x0);
            ss << " lb: " << settings.lb.transpose();
            ss << " ub: " << settings.ub.transpose();
            ss << " cs: " << to_string(settings.cs);
            ss << " cc: " << to_string(settings.cc);
            ss << " cmu: " << to_string(settings.cmu);
            ss << " c1: " << to_string(settings.c1);
            ss << " verbose: " << settings.verbose;
            ss << ">";
            return ss.str(); });

    ;

    using AdaptationType = std::variant<
        std::shared_ptr<matrix_adaptation::MatrixAdaptation>, 
        std::shared_ptr<matrix_adaptation::CovarianceAdaptation>, 
        std::shared_ptr<matrix_adaptation::None>
    >;
    py::class_<Parameters, std::shared_ptr<Parameters>>(main, "Parameters")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def(py::init<Settings>(), py::arg("settings"))
        .def("adapt", &Parameters::adapt)
        .def("perform_restart", &Parameters::perform_restart, py::arg("sigma") = std::nullopt)
        .def_readwrite("settings", &Parameters::settings)
        .def_readwrite("mu", &Parameters::mu)
        .def_readwrite("lamb", &Parameters::lambda)
        .def_property(
            "adaptation",
            [](Parameters &self) -> AdaptationType
            {
                switch (self.settings.modules.matrix_adaptation)
                {
                case MatrixAdaptationType::MATRIX:
                    return std::dynamic_pointer_cast<matrix_adaptation::MatrixAdaptation>(self.adaptation);
                case MatrixAdaptationType::NONE:
                    return std::dynamic_pointer_cast<matrix_adaptation::None>(self.adaptation);
                default:
                case MatrixAdaptationType::COVARIANCE:
                    return std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(self.adaptation);
                }
            },
            [](Parameters &self, std::shared_ptr<matrix_adaptation::Adaptation> adaptation)
            {
                self.adaptation = adaptation;
            })
        .def_readwrite("stats", &Parameters::stats)
        .def_readwrite("weights", &Parameters::weights)
        .def_readwrite("pop", &Parameters::pop)
        .def_readwrite("old_pop", &Parameters::old_pop)
        .def_readwrite("sampler", &Parameters::sampler)
        .def_readwrite("mutation", &Parameters::mutation)
        .def_readwrite("selection", &Parameters::selection)
        .def_readwrite("restart", &Parameters::restart)
        .def_readwrite("bounds", &Parameters::bounds);
}

void define_bounds(py::module &main)
{
    auto m = main.def_submodule("bounds");
    using namespace bounds;

    py::class_<BoundCorrection, std::shared_ptr<BoundCorrection>>(m, "BoundCorrection")
        .def_readwrite("lb", &BoundCorrection::lb)
        .def_readwrite("ub", &BoundCorrection::ub)
        .def_readwrite("db", &BoundCorrection::db)
        .def_readwrite("diameter", &BoundCorrection::diameter)
        .def_readonly("n_out_of_bounds", &BoundCorrection::n_out_of_bounds)
        .def("correct", &BoundCorrection::correct,
             py::arg("population"), py::arg("m"));

    py::class_<NoCorrection, BoundCorrection, std::shared_ptr<NoCorrection>>(m, "NoCorrection")
        .def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"));

    py::class_<CountOutOfBounds, BoundCorrection, std::shared_ptr<CountOutOfBounds>>(m, "CountOutOfBounds")
        .def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"));

    py::class_<COTN, BoundCorrection, std::shared_ptr<COTN>>(m, "COTN")
        .def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"))
        .def_readonly("sampler", &COTN::sampler);

    py::class_<Mirror, BoundCorrection, std::shared_ptr<Mirror>>(m, "Mirror")
        .def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"));

    py::class_<UniformResample, BoundCorrection, std::shared_ptr<UniformResample>>(m, "UniformResample")
        .def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"));

    py::class_<Saturate, BoundCorrection, std::shared_ptr<Saturate>>(m, "Saturate")
        .def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"));

    py::class_<Toroidal, BoundCorrection, std::shared_ptr<Toroidal>>(m, "Toroidal")
        .def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"));
}

void define_mutation(py::module &main)
{
    auto m = main.def_submodule("mutation");
    using namespace mutation;

    py::class_<ThresholdConvergence, std::shared_ptr<ThresholdConvergence>>(m, "ThresholdConvergence")
        .def(py::init<>())
        .def_readwrite("init_threshold", &ThresholdConvergence::init_threshold)
        .def_readwrite("decay_factor", &ThresholdConvergence::decay_factor)
        .def("scale", &ThresholdConvergence::scale, py::arg("population"), py::arg("diameter"), py::arg("budget"), py::arg("evaluations"));

    py::class_<NoThresholdConvergence, ThresholdConvergence, std::shared_ptr<NoThresholdConvergence>>(m, "NoThresholdConvergence")
        .def(py::init<>());

    py::class_<SequentialSelection, std::shared_ptr<SequentialSelection>>(m, "SequentialSelection")
        .def(py::init<parameters::Mirror, size_t, double>(),
             py::arg("mirror"),
             py::arg("mu"),
             py::arg("seq_cuttoff_factor") = 1.0)
        .def("break_conditions", &SequentialSelection::break_conditions,
             py::arg("i"),
             py::arg("f"),
             py::arg("fopt"),
             py::arg("mirror"));

    py::class_<NoSequentialSelection, SequentialSelection, std::shared_ptr<NoSequentialSelection>>(m, "NoSequentialSelection")
        .def(py::init<parameters::Mirror, size_t, double>(),
             py::arg("mirror"),
             py::arg("mu"),
             py::arg("seq_cuttoff_factor") = 1.0);

    py::class_<SigmaSampler, std::shared_ptr<SigmaSampler>>(m, "SigmaSampler")
        .def(py::init<double>(), py::arg("dimension"))
        .def_readwrite("beta", &SigmaSampler::beta)
        .def("sample", &SigmaSampler::sample, py::arg("sigma"), py::arg("population"));

    py::class_<NoSigmaSampler, SigmaSampler, std::shared_ptr<NoSigmaSampler>>(m, "NoSigmaSampler")
        .def(py::init<double>(), py::arg("dimension"));

    py::class_<Strategy, std::shared_ptr<Strategy>>(m, "Strategy")
        .def("adapt", &Strategy::adapt, py::arg("weights"),
             py::arg("dynamic"),
             py::arg("population"),
             py::arg("old_population"),
             py::arg("stats"),
             py::arg("lamb"))
        .def_readwrite("threshold_convergence", &Strategy::tc)
        .def_readwrite("sequential_selection", &Strategy::sq)
        .def_readwrite("sigma_sampler", &Strategy::ss)
        .def_readwrite("cs", &Strategy::cs)
        .def_readwrite("sigma", &Strategy::sigma)
        .def_readwrite("s", &Strategy::s);

    py::class_<CSA, Strategy, std::shared_ptr<CSA>>(m, "CSA")
        .def(
            py::init<std::shared_ptr<ThresholdConvergence>, std::shared_ptr<SequentialSelection>, std::shared_ptr<SigmaSampler>, double, double, double>(),
            py::arg("threshold_convergence"),
            py::arg("sequential_selection"),
            py::arg("sigma_sampler"),
            py::arg("cs"),
            py::arg("damps"),
            py::arg("sigma0"))
        .def_readwrite("damps", &CSA::damps)
        .def(
            "mutate", &CSA::mutate,
            py::arg("objective"),
            py::arg("n_offspring"),
            py::arg("parameters"));

    py::class_<TPA, CSA, std::shared_ptr<TPA>>(m, "TPA")
        .def(py::init<std::shared_ptr<ThresholdConvergence>, std::shared_ptr<SequentialSelection>, std::shared_ptr<SigmaSampler>, double, double, double>(),
             py::arg("threshold_convergence"),
             py::arg("sequential_selection"),
             py::arg("sigma_sampler"),
             py::arg("cs"),
             py::arg("damps"),
             py::arg("sigma0"))
        .def_readwrite("a_tpa", &TPA::a_tpa)
        .def_readwrite("b_tpa", &TPA::b_tpa)
        .def_readwrite("rank_tpa", &TPA::rank_tpa);

    py::class_<MSR, CSA, std::shared_ptr<MSR>>(m, "MSR")
        .def(py::init<std::shared_ptr<ThresholdConvergence>, std::shared_ptr<SequentialSelection>, std::shared_ptr<SigmaSampler>, double, double, double>(),
             py::arg("threshold_convergence"),
             py::arg("sequential_selection"),
             py::arg("sigma_sampler"),
             py::arg("cs"),
             py::arg("damps"),
             py::arg("sigma0"));

    py::class_<PSR, CSA, std::shared_ptr<PSR>>(m, "PSR")
        .def(py::init<std::shared_ptr<ThresholdConvergence>, std::shared_ptr<SequentialSelection>, std::shared_ptr<SigmaSampler>, double, double, double>(),
             py::arg("threshold_convergence"),
             py::arg("sequential_selection"),
             py::arg("sigma_sampler"),
             py::arg("cs"),
             py::arg("damps"),
             py::arg("sigma0"))
        .def_readwrite("success_ratio", &PSR::succes_ratio);

    py::class_<XNES, CSA, std::shared_ptr<XNES>>(m, "XNES")
        .def(py::init<std::shared_ptr<ThresholdConvergence>, std::shared_ptr<SequentialSelection>, std::shared_ptr<SigmaSampler>, double, double, double>(),
             py::arg("threshold_convergence"),
             py::arg("sequential_selection"),
             py::arg("sigma_sampler"),
             py::arg("cs"),
             py::arg("damps"),
             py::arg("sigma0"));

    py::class_<MXNES, CSA, std::shared_ptr<MXNES>>(m, "MXNES")
        .def(py::init<std::shared_ptr<ThresholdConvergence>, std::shared_ptr<SequentialSelection>, std::shared_ptr<SigmaSampler>, double, double, double>(),
             py::arg("threshold_convergence"),
             py::arg("sequential_selection"),
             py::arg("sigma_sampler"),
             py::arg("cs"),
             py::arg("damps"),
             py::arg("sigma0"));

    py::class_<LPXNES, CSA, std::shared_ptr<LPXNES>>(m, "LPXNES")
        .def(py::init<std::shared_ptr<ThresholdConvergence>, std::shared_ptr<SequentialSelection>, std::shared_ptr<SigmaSampler>, double, double, double>(),
             py::arg("threshold_convergence"),
             py::arg("sequential_selection"),
             py::arg("sigma_sampler"),
             py::arg("cs"),
             py::arg("damps"),
             py::arg("sigma0"));
}

void define_population(py::module &main)
{
    py::class_<Population>(main, "Population")
        .def(py::init<size_t, size_t>(), py::arg("dimension"), py::arg("n"))
        .def(py::init<Matrix, Matrix, Matrix, Vector, Vector>(), py::arg("X"), py::arg("Z"), py::arg("Y"), py::arg("f"), py::arg("s"))
        .def("sort", &Population::sort)
        .def("resize_cols", &Population::resize_cols, py::arg("size"))
        .def("keep_only", &Population::keep_only, py::arg("idx"))
        .def_property_readonly("n_finite", &Population::n_finite)
        .def("__add__", &Population::operator+=, py::arg("other"))
        .def_readwrite("X", &Population::X)
        .def_readwrite("Z", &Population::Z)
        .def_readwrite("Y", &Population::Y)
        .def_readwrite("f", &Population::f)
        .def_readwrite("s", &Population::s)
        .def_readwrite("d", &Population::d)
        .def_readwrite("n", &Population::n);
}

void define_restart(py::module &main)
{
    auto m = main.def_submodule("restart");
    using namespace restart;

    py::class_<RestartCriteria>(m, "RestartCriteria")
        .def(py::init<double, double, size_t>(), py::arg("dimension"), py::arg("lamb"), py::arg("time"))
        .def("exceeded_max_iter", &RestartCriteria::exceeded_max_iter)
        .def("no_improvement", &RestartCriteria::no_improvement)
        .def("flat_fitness", &RestartCriteria::flat_fitness)
        .def("tolx", &RestartCriteria::tolx)
        .def("tolupsigma", &RestartCriteria::tolupsigma)
        .def("conditioncov", &RestartCriteria::conditioncov)
        .def("noeffectaxis", &RestartCriteria::noeffectaxis)
        .def("noeffectcoor", &RestartCriteria::noeffectcoor)
        .def("stagnation", &RestartCriteria::stagnation)
        .def_readonly("last_restart", &RestartCriteria::last_restart)
        .def_readonly("max_iter", &RestartCriteria::max_iter)
        .def_readonly("n_bin", &RestartCriteria::n_bin)
        .def_readonly("n_stagnation", &RestartCriteria::n_stagnation)
        .def_readonly("flat_fitness_index", &RestartCriteria::flat_fitness_index)
        .def_readonly("flat_fitnesses", &RestartCriteria::flat_fitnesses)
        .def_readonly("median_fitnesses", &RestartCriteria::median_fitnesses)
        .def_readonly("best_fitnesses", &RestartCriteria::best_fitnesses)
        .def_readonly("time_since_restart", &RestartCriteria::time_since_restart)
        .def_readonly("recent_improvement", &RestartCriteria::recent_improvement)
        .def_readonly("n_flat_fitness", &RestartCriteria::n_flat_fitness)
        .def_readonly("d_sigma", &RestartCriteria::d_sigma)
        .def_readonly("tolx_condition", &RestartCriteria::tolx_condition)
        .def_readonly("tolx_vector", &RestartCriteria::tolx_vector)
        .def_readonly("root_max_d", &RestartCriteria::root_max_d)
        .def_readonly("condition_c", &RestartCriteria::condition_c)
        .def_readonly("effect_coord", &RestartCriteria::effect_coord)
        .def_readonly("effect_axis", &RestartCriteria::effect_axis)
        .def_readonly("any", &RestartCriteria::any)
        .def("__call__", &RestartCriteria::operator(), py::arg("parameters"))
        .def("__repr__", [](const RestartCriteria &res)
             {
            std::stringstream ss;
            ss << std::boolalpha;
            ss <<  "<RestartCriteria";
            ss << " flat_fitness: " << res.flat_fitness();
            ss << " exeeded_max_iter: " << res.exceeded_max_iter();
            ss << " no_improvement: " << res.no_improvement();
            ss << " tolx: " << res.tolx();
            ss << " tolupsigma: " << res.tolupsigma();
            ss << " conditioncov: " << res.conditioncov();
            ss << " noeffectaxis: " << res.noeffectaxis();
            ss << " noeffectcoor: " << res.noeffectcoor();
            ss << " stagnation: " << res.stagnation() <<  ">";
            return ss.str(); });

    py::class_<Strategy, std::shared_ptr<Strategy>>(m, "Strategy")
        .def("evaluate", &Strategy::evaluate, py::arg("parameters"))
        .def_readwrite("criteria", &Strategy::criteria);

    py::class_<None, Strategy, std::shared_ptr<None>>(m, "NoRestart")
        .def(py::init<double, double>(), py::arg("dimension"), py::arg("lamb"))
        .def("restart", &None::restart, py::arg("parameters"));

    py::class_<Stop, Strategy, std::shared_ptr<Stop>>(m, "Stop")
        .def(py::init<double, double>(), py::arg("dimension"), py::arg("lamb"))
        .def("restart", &Stop::restart, py::arg("parameters"));

    py::class_<Restart, Strategy, std::shared_ptr<Restart>>(m, "Restart")
        .def(py::init<size_t, double>(), py::arg("dimension"), py::arg("lamb"))
        .def("restart", &Restart::restart, py::arg("parameters"));

    py::class_<IPOP, Strategy, std::shared_ptr<IPOP>>(m, "IPOP")
        .def(py::init<double, double>(), py::arg("dimension"), py::arg("lamb"))
        .def("restart", &IPOP::restart, py::arg("parameters"))
        .def_readwrite("ipop_factor", &IPOP::ipop_factor);

    py::class_<BIPOP, Strategy, std::shared_ptr<BIPOP>>(m, "BIPOP")
        .def(py::init<size_t, double, double, size_t>(), py::arg("dimension"), py::arg("lamb"), py::arg("mu"), py::arg("budget"))
        .def("restart", &BIPOP::restart, py::arg("parameters"))
        .def("large", &BIPOP::large)
        .def_readwrite("mu_factor", &BIPOP::mu_factor)
        .def_readwrite("lambda_init", &BIPOP::lambda_init)
        .def_readwrite("budget", &BIPOP::budget)
        .def_readwrite("lambda_large", &BIPOP::lambda_large)
        .def_readwrite("lambda_small", &BIPOP::lambda_small)
        .def_readwrite("budget_small", &BIPOP::budget_small)
        .def_readwrite("budget_large", &BIPOP::budget_large)
        .def_readonly("used_budget", &BIPOP::used_budget);
}

void define_cmaes(py::module &m)
{
    py::class_<ModularCMAES>(m, "ModularCMAES")
        .def(py::init<std::shared_ptr<parameters::Parameters>>(), py::arg("parameters"))
        .def("recombine", &ModularCMAES::recombine)
        .def("mutate", [](ModularCMAES &self, std::function<double(Vector)> objective)
             { self.p->mutation->mutate(objective, self.p->lambda, *self.p); })
        .def("select", [](ModularCMAES &self)
             { self.p->selection->select(*self.p); })
        .def("adapt", [](ModularCMAES &self)
             { self.p->adapt(); })
        .def("step", &ModularCMAES::step, py::arg("objective"))
        .def("__call__", &ModularCMAES::operator(), py::arg("objective"))
        .def("run", &ModularCMAES::operator(), py::arg("objective"))
        .def("break_conditions", &ModularCMAES::break_conditions)
        .def_readonly("p", &ModularCMAES::p);
}

PYBIND11_MODULE(cmaescpp, m)
{
    define_options(m);
    define_utils(m);
    define_population(m);
    define_samplers(m);
    define_mutation(m);
    define_restart(m);
    define_matrix_adaptation(m);
    define_parameters(m);
    define_bounds(m);
    define_selection(m);
    define_cmaes(m);
}