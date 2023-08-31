#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/c_maes.hpp"
#include "c_maes.hpp"

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

    py::enum_<parameters::RecombinationWeights>(m, "RecombinationWeights")
        .value("DEFAULT", parameters::RecombinationWeights::DEFAULT)
        .value("EQUAL", parameters::RecombinationWeights::EQUAL)
        .value("HALF_POWER_LAMBDA", parameters::RecombinationWeights::HALF_POWER_LAMBDA)
        .export_values();

    py::enum_<sampling::BaseSampler>(m, "BaseSampler")
        .value("GAUSSIAN", sampling::BaseSampler::GAUSSIAN)
        .value("SOBOL", sampling::BaseSampler::SOBOL)
        .value("HALTON", sampling::BaseSampler::HALTON)
        .export_values();

    py::enum_<sampling::Mirror>(m, "Mirror")
        .value("NONE", sampling::Mirror::NONE)
        .value("MIRRORED", sampling::Mirror::MIRRORED)
        .value("PAIRWISE", sampling::Mirror::PAIRWISE)
        .export_values();

    py::enum_<mutation::StepSizeAdaptation>(m, "StepSizeAdaptation")
        .value("CSA", mutation::StepSizeAdaptation::CSA)
        .value("TPA", mutation::StepSizeAdaptation::TPA)
        .value("MSR", mutation::StepSizeAdaptation::MSR)
        .value("XNES", mutation::StepSizeAdaptation::XNES)
        .value("MXNES", mutation::StepSizeAdaptation::MXNES)
        .value("LPXNES", mutation::StepSizeAdaptation::LPXNES)
        .value("PSR", mutation::StepSizeAdaptation::PSR)
        .export_values();

    py::enum_<bounds::CorrectionMethod>(m, "CorrectionMethod")
        .value("NONE", bounds::CorrectionMethod::NONE)
        .value("COUNT", bounds::CorrectionMethod::COUNT)
        .value("MIRROR", bounds::CorrectionMethod::MIRROR)
        .value("COTN", bounds::CorrectionMethod::COTN)
        .value("UNIFORM_RESAMPLE", bounds::CorrectionMethod::UNIFORM_RESAMPLE)
        .value("SATURATE", bounds::CorrectionMethod::SATURATE)
        .value("TOROIDAL", bounds::CorrectionMethod::TOROIDAL)
        .export_values();

    py::enum_<restart::StrategyType>(m, "RestartStrategy")
        .value("NONE", restart::StrategyType::NONE)
        .value("RESTART", restart::StrategyType::RESTART)
        .value("IPOP", restart::StrategyType::IPOP)
        .value("BIPOP", restart::StrategyType::BIPOP)
        .export_values();
}

class PySampler: public sampling::Sampler {
public:
    using Sampler::Sampler;

    Vector operator()() override {
        PYBIND11_OVERRIDE_PURE_NAME(Vector, sampling::Sampler, "__call__", operator());
    };
};

void define_samplers(py::module &main)
{
    using namespace sampling;

    auto m = main.def_submodule("sampling");

    py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Sampler")
        .def_readonly("d", &Sampler::d);

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
        .def(py::init<size_t, size_t>(), py::arg("d"), py::arg("i"))
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
        .def_readwrite("restart_strategy", &Modules::restart_strategy);

    py::class_<Stats>(m, "Stats")
        .def(py::init<>())
        .def_readwrite("t", &Stats::t)
        .def_readwrite("evaluations", &Stats::evaluations)
        .def_readwrite("target", &Stats::target)
        .def_readwrite("max_generations", &Stats::max_generations)
        .def_readwrite("budget", &Stats::budget)
        .def_readwrite("xopt", &Stats::xopt)
        .def_readwrite("fopt", &Stats::fopt);

    py::class_<Weights>(m, "Weights")
        .def(
            py::init<size_t, size_t, size_t, Modules>(),
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
        .def_readwrite("negative", &Weights::negative);

    py::class_<Dynamic>(m, "Dynamic")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def_readwrite("m", &Dynamic::m)
        .def_readwrite("m_old", &Dynamic::m_old)
        .def_readwrite("dm", &Dynamic::dm)
        .def_readwrite("pc", &Dynamic::pc)
        .def_readwrite("ps", &Dynamic::ps)
        .def_readwrite("d", &Dynamic::d)
        .def_readwrite("B", &Dynamic::B)
        .def_readwrite("C", &Dynamic::C)
        .def_readwrite("inv_root_C", &Dynamic::inv_root_C)
        .def_readwrite("dd", &Dynamic::dd)
        .def_readwrite("chiN", &Dynamic::chiN)
        .def_readwrite("hs", &Dynamic::hs)
        .def("adapt_evolution_paths", &Dynamic::adapt_evolution_paths,
             py::arg("weights"),
             py::arg("mutation"),
             py::arg("stats"),
             py::arg("lambda"))
        .def("adapt_covariance_matrix", &Dynamic::adapt_covariance_matrix,
             py::arg("weights"),
             py::arg("modules"),
             py::arg("population"),
             py::arg("mu"))
        .def("perform_eigendecomposition", &Dynamic::perform_eigendecomposition, py::arg("stats"));

    py::class_<Parameters, std::shared_ptr<Parameters>>(main, "Parameters")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def(py::init<size_t, Modules>(), py::arg("dimension"), py::arg("modules"))
        .def("adapt", &Parameters::adapt)
        .def("perform_restart", &Parameters::perform_restart, py::arg("sigma") = std::nullopt)
        .def_readonly("dim", &Parameters::dim)
        .def_readwrite("mu", &Parameters::mu)
        .def_readwrite("lamb", &Parameters::lambda)
        .def_readwrite("modules", &Parameters::modules)
        .def_readwrite("dynamic", &Parameters::dynamic)
        .def_readwrite("stats", &Parameters::stats)
        .def_readwrite("weights", &Parameters::weights)
        .def_readwrite("pop", &Parameters::pop)
        .def_readwrite("old_pop", &Parameters::old_pop)
        .def_readwrite("sampler", &Parameters::sampler)
        .def_readwrite("mutation", &Parameters::mutation)
        .def_readwrite("selection", &Parameters::selection)
        .def_readwrite("restart", &Parameters::restart)
        .def_readwrite("bounds", &Parameters::bounds)
        .def_readwrite("verbose", &Parameters::verbose);
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
        .def_readonly("n_out_of_bounds", &BoundCorrection::n_out_of_bounds);

    py::class_<NoCorrection, BoundCorrection, std::shared_ptr<NoCorrection>>(m, "NoCorrection")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("correct", &NoCorrection::correct,
             py::arg("X"), py::arg("Y"), py::arg("s"), py::arg("m"));

    py::class_<CountOutOfBounds, BoundCorrection, std::shared_ptr<CountOutOfBounds>>(m, "CountOutOfBounds")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("correct", &CountOutOfBounds::correct,
             py::arg("X"), py::arg("Y"), py::arg("s"), py::arg("m"));

    py::class_<COTN, BoundCorrection, std::shared_ptr<COTN>>(m, "COTN")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("correct", &COTN::correct,
             py::arg("X"), py::arg("Y"), py::arg("s"), py::arg("m"))
        .def_readonly("sampler", &COTN::sampler);

    py::class_<Mirror, BoundCorrection, std::shared_ptr<Mirror>>(m, "Mirror")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("correct", &Mirror::correct,
             py::arg("X"), py::arg("Y"), py::arg("s"), py::arg("m"));

    py::class_<UniformResample, BoundCorrection, std::shared_ptr<UniformResample>>(m, "UniformResample")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("correct", &UniformResample::correct,
             py::arg("X"), py::arg("Y"), py::arg("s"), py::arg("m"));

    py::class_<Saturate, BoundCorrection, std::shared_ptr<Saturate>>(m, "Saturate")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("correct", &Saturate::correct,
             py::arg("X"), py::arg("Y"), py::arg("s"), py::arg("m"));

    py::class_<Toroidal, BoundCorrection, std::shared_ptr<Toroidal>>(m, "Toroidal")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("correct", &Toroidal::correct,
             py::arg("X"), py::arg("Y"), py::arg("s"), py::arg("m"));
}

void define_mutation(py::module &main)
{
    auto m = main.def_submodule("mutation");
    using namespace mutation;

    py::class_<ThresholdConvergence, std::shared_ptr<ThresholdConvergence>>(m, "ThresholdConvergence")
        .def(py::init<>())
        .def("scale", &ThresholdConvergence::scale, py::arg("z"), py::arg("stats"), py::arg("bounds"));

    py::class_<NoThresholdConvergence, ThresholdConvergence, std::shared_ptr<NoThresholdConvergence>>(m, "NoThresholdConvergence")
        .def(py::init<>());

    py::class_<SequentialSelection, std::shared_ptr<SequentialSelection>>(m, "SequentialSelection")
        .def(py::init<sampling::Mirror, size_t, double>(),
             py::arg("mirror"),
             py::arg("mu"),
             py::arg("seq_cuttoff_factor") = 1.0)
        .def("break_conditions", &SequentialSelection::break_conditions,
             py::arg("i"),
             py::arg("f"),
             py::arg("fopt"),
             py::arg("mirror"));

    py::class_<NoSequentialSelection, SequentialSelection, std::shared_ptr<NoSequentialSelection>>(m, "NoSequentialSelection")
        .def(py::init<sampling::Mirror, size_t, double>(),
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
             py::arg("lambda"))
        .def("sample_sigma", &Strategy::sample_sigma, py::arg("population"))
        .def_readwrite("threshold_convergence", &Strategy::tc)
        .def_readwrite("sequential_selection", &Strategy::sq)
        .def_readwrite("sigma_sampler", &Strategy::ss)
        .def_readwrite("cs", &Strategy::cs)
        .def_readwrite("sigma0", &Strategy::sigma0)
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
            py::arg("parameters"))
        .def("adapt_sigma", &CSA::adapt_sigma,
             py::arg("weights"),
             py::arg("dynamic"),
             py::arg("population"),
             py::arg("old_pop"),
             py::arg("stats"),
             py::arg("lambda"));

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

    py::class_<Strategy, std::shared_ptr<Strategy>>(m, "Strategy");

    py::class_<None, Strategy, std::shared_ptr<None>>(m, "NoRestart")
        .def(py::init<>())
        .def("evaluate", &None::evaluate, py::arg("parameters"));

    py::class_<Restart, Strategy, std::shared_ptr<Restart>>(m, "Restart")
        .def(py::init<size_t, double>(), py::arg("dimension"), py::arg("lamb"))
        .def("evaluate", &Restart::evaluate, py::arg("parameters"))
        .def("setup", &Restart::setup, py::arg("dimension"), py::arg("lamb"), py::arg("t"))
        .def("termination_criteria", &Restart::termination_criteria, py::arg("parameters"))
        .def("restart", &Restart::restart, py::arg("parameters"))
        .def_readonly("last_restart", &Restart::last_restart)
        .def_readonly("max_iter", &Restart::max_iter)
        .def_readonly("n_bin", &Restart::n_bin)
        .def_readonly("n_stagnation", &Restart::n_stagnation)
        .def_readonly("flat_fitness_index", &Restart::flat_fitness_index)
        .def_readonly("flat_fitnesses", &Restart::flat_fitnesses)
        .def_readonly("median_fitnesses", &Restart::median_fitnesses)
        .def_readonly("best_fitnesses", &Restart::best_fitnesses);

    py::class_<IPOP, Restart, std::shared_ptr<IPOP>>(m, "IPOP")
        .def(py::init<size_t, double>(), py::arg("dimension"), py::arg("lamb"))
        .def_readwrite("ipop_factor", &IPOP::ipop_factor);

    py::class_<BIPOP, Restart, std::shared_ptr<BIPOP>>(m, "BIPOP")
        .def(py::init<size_t, double, double, size_t>(), py::arg("dimension"), py::arg("lamb"), py::arg("mu"), py::arg("budget"))
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
        .def("step", &ModularCMAES::step, py::arg("objective"))
        .def("__call__", &ModularCMAES::operator(), py::arg("objective"))
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
    define_parameters(m);
    define_bounds(m);
    define_selection(m);
    define_cmaes(m);
}