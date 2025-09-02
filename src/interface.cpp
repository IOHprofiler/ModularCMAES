#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "c_maes.hpp"
#include "to_string.hpp"
#include "es.hpp"

namespace py = pybind11;

template <typename RNG>
Float random_double()
{
	static RNG gen;
	return gen(rng::GENERATOR);
}

void define_options(py::module& main)
{
	auto m = main.def_submodule("options");
	using namespace parameters;
	py::enum_<RecombinationWeights>(m, "RecombinationWeights")
		.value("DEFAULT", parameters::RecombinationWeights::DEFAULT)
		.value("EQUAL", parameters::RecombinationWeights::EQUAL)
		.value("EXPONENTIAL", parameters::RecombinationWeights::EXPONENTIAL)
		.export_values();

	py::enum_<BaseSampler>(m, "BaseSampler")
		.value("UNIFORM", BaseSampler::UNIFORM)
		.value("SOBOL", BaseSampler::SOBOL)
		.value("HALTON", BaseSampler::HALTON)
		.export_values();

	py::enum_<SampleTranformerType>(m, "SampleTranformerType")
		.value("NONE", SampleTranformerType::NONE)
		.value("GAUSSIAN", SampleTranformerType::GAUSSIAN)
		.value("SCALED_UNIFORM", SampleTranformerType::SCALED_UNIFORM)
		.value("LAPLACE", SampleTranformerType::LAPLACE)
		.value("LOGISTIC", SampleTranformerType::LOGISTIC)
		.value("CAUCHY", SampleTranformerType::CAUCHY)
		.value("DOUBLE_WEIBULL", SampleTranformerType::DOUBLE_WEIBULL)
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
		.value("SR", StepSizeAdaptation::SR)
		.value("SA", StepSizeAdaptation::SA)
		.export_values();

	py::enum_<CorrectionMethod>(m, "CorrectionMethod")
		.value("NONE", CorrectionMethod::NONE)
		.value("MIRROR", CorrectionMethod::MIRROR)
		.value("COTN", CorrectionMethod::COTN)
		.value("UNIFORM_RESAMPLE", CorrectionMethod::UNIFORM_RESAMPLE)
		.value("SATURATE", CorrectionMethod::SATURATE)
		.value("TOROIDAL", CorrectionMethod::TOROIDAL)
		.value("RESAMPLE", CorrectionMethod::RESAMPLE)
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
		.value("SEPARABLE", MatrixAdaptationType::SEPARABLE)
		.value("CHOLESKY", MatrixAdaptationType::CHOLESKY)
		.value("CMSA", MatrixAdaptationType::CMSA)
		.value("COVARIANCE_NO_EIGV", MatrixAdaptationType::COVARIANCE_NO_EIGV)
		.value("NATURAL_GRADIENT", MatrixAdaptationType::NATURAL_GRADIENT)
		.export_values();

	py::enum_<CenterPlacement>(m, "CenterPlacement")
		.value("X0", CenterPlacement::X0)
		.value("ZERO", CenterPlacement::ZERO)
		.value("UNIFORM", CenterPlacement::UNIFORM)
		.export_values();
}

struct PySampler : sampling::Sampler
{
	std::function<Float()> func;

	PySampler(size_t d, std::function<Float()> f) : Sampler::Sampler(d), func(f) {}

	Vector operator()() override
	{
		Vector res(d);
		for (size_t j = 0; j < d; ++j)
			res(j) = func();
		return res;
	};
};

void define_samplers(py::module& main)
{
	using namespace sampling;

	auto m = main.def_submodule("sampling");

	py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Sampler")
		.def_readonly("d", &Sampler::d)
		.def("reset", &Sampler::reset)
		.def("expected_length", &Sampler::expected_length);

	py::class_<PySampler, Sampler, std::shared_ptr<PySampler>>(m, "PySampler")
		.def(py::init<size_t, std::function<Float()>>(), py::arg("d"), py::arg("function"))
		.def("__call__", &PySampler::operator());

	py::class_<Gaussian, Sampler, std::shared_ptr<Gaussian>>(m, "Gaussian")
		.def(py::init<size_t>(), py::arg("d"))
		.def("__call__", &Gaussian::operator());

	py::class_<Uniform, Sampler, std::shared_ptr<Uniform>>(m, "Uniform")
		.def(py::init<size_t>(), py::arg("d"))
		.def("__call__", &Uniform::operator());

	py::class_<Sobol, Sampler, std::shared_ptr<Sobol>>(m, "Sobol")
		.def(py::init<size_t>(), py::arg("d"))
		.def_readonly("cache", &Sobol::cache)
		.def("__call__", &Sobol::operator());

	py::class_<Halton, Sampler, std::shared_ptr<Halton>>(m, "Halton")
		.def(py::init<size_t, bool>(), py::arg("d"), py::arg("scrambled") = true)
		.def("__call__", &Halton::operator());

	py::class_<Mirrored, Sampler, std::shared_ptr<Mirrored>>(m, "Mirrored")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def("__call__", &Mirrored::operator());

	py::class_<CachedSampler, Sampler, std::shared_ptr<CachedSampler>>(m, "CachedSampler")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def(py::init<std::vector<Vector>, bool>(), py::arg("cache"), py::arg("transform_ppf") = false)
		.def("__call__", &CachedSampler::operator())
		.def_readonly("index", &CachedSampler::index)
		.def_readonly("n_samples", &CachedSampler::n_samples)
		.def_readonly("cache", &CachedSampler::cache);

	py::class_<Orthogonal, Sampler, std::shared_ptr<Orthogonal>>(m, "Orthogonal")
		.def(py::init<const std::shared_ptr<Sampler>, size_t>(),
			py::arg("sampler"), py::arg("n_samples"))
		.def("__call__", &Orthogonal::operator());

	py::class_<SampleTransformer, Sampler, std::shared_ptr<SampleTransformer>>(m, "SampleTransformer")
		.def("raw", &SampleTransformer::raw);

	py::class_<IdentityTransformer, SampleTransformer, std::shared_ptr<IdentityTransformer>>(m, "IdentityTransformer")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def("transform", &IdentityTransformer::transform)
		.def("__call__", &IdentityTransformer::operator())
		.def("expected_length", &IdentityTransformer::expected_length);

	py::class_<GaussianTransformer, SampleTransformer, std::shared_ptr<GaussianTransformer>>(m, "GaussianTransformer")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def("transform", &GaussianTransformer::transform)
		.def("__call__", &GaussianTransformer::operator())
		.def("expected_length", &GaussianTransformer::expected_length);

	py::class_<UniformScaler, SampleTransformer, std::shared_ptr<UniformScaler>>(m, "UniformScaler")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def("transform", &UniformScaler::transform)
		.def("__call__", &UniformScaler::operator())
		.def("expected_length", &UniformScaler::expected_length);

	py::class_<LaplaceTransformer, SampleTransformer, std::shared_ptr<LaplaceTransformer>>(m, "LaplaceTransformer")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def("transform", &LaplaceTransformer::transform)
		.def("__call__", &LaplaceTransformer::operator())
		.def("expected_length", &LaplaceTransformer::expected_length);

	py::class_<LogisticTransformer, SampleTransformer, std::shared_ptr<LogisticTransformer>>(m, "LogisticTransformer")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def("transform", &LogisticTransformer::transform)
		.def("__call__", &LogisticTransformer::operator())
		.def("expected_length", &LogisticTransformer::expected_length);

	py::class_<CauchyTransformer, SampleTransformer, std::shared_ptr<CauchyTransformer>>(m, "CauchyTransformer")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def("transform", &CauchyTransformer::transform)
		.def("__call__", &CauchyTransformer::operator())
		.def("expected_length", &CauchyTransformer::expected_length);

	py::class_<DoubleWeibullTransformer, SampleTransformer, std::shared_ptr<DoubleWeibullTransformer>>(m, "DoubleWeibullTransformer")
		.def(py::init<const std::shared_ptr<Sampler>>(), py::arg("sampler"))
		.def("transform", &DoubleWeibullTransformer::transform)
		.def("__call__", &DoubleWeibullTransformer::operator())
		.def("expected_length", &DoubleWeibullTransformer::expected_length);
}

void define_utils(py::module& main)
{
	auto m = main.def_submodule("utils");
	m.def("cdf", &cdf, py::arg("x"));
	m.def("ppf", &ppf, py::arg("x"));
	m.def("i8_sobol", &i8_sobol, py::arg("dim_num"), py::arg("seed"), py::arg("quasi"));
	m.def("compute_ert", &utils::compute_ert, py::arg("running_times"), py::arg("budget"));
	m.def("set_seed", &rng::set_seed, py::arg("seed"), "Set the random seed");
	m.def("random_uniform", &random_double<rng::uniform<Float>>, "Generate a uniform random number in [0, 1]");
	m.def("random_normal", &random_double<rng::normal<Float>>, "Generate a standard normal random number");

	py::class_<rng::Shuffler>(m, "Shuffler")
		.def(py::init<size_t, size_t>(), py::arg("start"), py::arg("stop"))
		.def(py::init<size_t>(), py::arg("stop"))
		.def("next", &rng::Shuffler::next)
		.def_readwrite("start", &rng::Shuffler::start)
		.def_readwrite("stop", &rng::Shuffler::stop)
		.def_readwrite("n", &rng::Shuffler::n)
		.def_readwrite("seed", &rng::Shuffler::seed)
		.def_readwrite("offset", &rng::Shuffler::offset)
		.def_readwrite("multiplier", &rng::Shuffler::multiplier)
		.def_readwrite("modulus", &rng::Shuffler::modulus)
		.def_readwrite("found", &rng::Shuffler::found);

	py::class_<rng::CachedShuffleSequence>(m, "CachedShuffleSequence")
		.def(py::init<size_t>(), py::arg("dim"))
		.def("fill", &rng::CachedShuffleSequence::fill)
		.def("get_index", &rng::CachedShuffleSequence::get_index, py::arg("index"))
		.def("next", &rng::CachedShuffleSequence::next);
}

void define_selection(py::module& main)
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

void define_center_placement(py::module& main)
{
	auto m = main.def_submodule("center");
	using namespace center;
	py::class_<Placement, std::shared_ptr<Placement>>(m, "Placement")
		.def("__call__", &Placement::operator(), py::arg("parameters"));

	py::class_<X0, Placement, std::shared_ptr<X0>>(m, "X0")
		.def(py::init<>());

	py::class_<Uniform, Placement, std::shared_ptr<Uniform>>(m, "Uniform")
		.def(py::init<>());

	py::class_<Zero, Placement, std::shared_ptr<Zero>>(m, "Zero")
		.def(py::init<>());
}

void define_repelling(py::module& main)
{
	using namespace repelling;
	auto m = main.def_submodule("repelling");

	py::class_<TabooPoint>(m, "TabooPoint")
		.def(py::init<Solution, Float>(), py::arg("solution"), py::arg("radius"))
		.def("rejects", &TabooPoint::rejects, py::arg("xi"), py::arg("p"), py::arg("attempts"))
		.def("shares_basin", &TabooPoint::shares_basin, py::arg("objective"), py::arg("xi"), py::arg("p"))
		.def("calculate_criticality", &TabooPoint::calculate_criticality, py::arg("p"))
		.def_readwrite("radius", &TabooPoint::radius)
		.def_readwrite("n_rep", &TabooPoint::n_rep)
		.def_readwrite("solution", &TabooPoint::solution)
		.def_readwrite("shrinkage", &TabooPoint::shrinkage)
		.def_readwrite("criticality", &TabooPoint::criticality)
		.def("__repr__", [] (TabooPoint& tb) {
		return "<TabooPoint nc:" + std::to_string(tb.n_rep) +
			"y: " + std::to_string(tb.solution.y) + ">";
			});

	py::class_<Repelling, std::shared_ptr<Repelling>>(m, "Repelling")
		.def(py::init<>())
		.def("is_rejected", &Repelling::is_rejected, py::arg("xi"), py::arg("p"))
		.def("update_archive", &Repelling::update_archive, py::arg("objective"), py::arg("p"))
		.def("prepare_sampling", &Repelling::prepare_sampling, py::arg("p"))
		.def_readwrite("archive", &Repelling::archive)
		.def_readwrite("coverage", &Repelling::coverage)
		.def_readwrite("attempts", &Repelling::attempts)
	;

	py::class_<NoRepelling, Repelling, std::shared_ptr<NoRepelling>>(m, "NoRepelling")
		.def(py::init<>());

	m.def("euclidian", &distance::euclidian, py::arg("u"), py::arg("v"));
	m.def("manhattan", &distance::manhattan, py::arg("u"), py::arg("v"));
	m.def("mahanolobis", &distance::mahanolobis, py::arg("u"), py::arg("v"), py::arg("C_inv"));
	m.def("hill_valley_test", &distance::hill_valley_test,
		py::arg("u"), py::arg("v"), py::arg("f"), py::arg("n_evals"));
}

void define_matrix_adaptation(py::module& main)
{
	using namespace matrix_adaptation;
	auto m = main.def_submodule("matrix_adaptation");
	py::class_<Adaptation, std::shared_ptr<Adaptation>>(m, "Adaptation")
		.def_readwrite("m", &Adaptation::m)
		.def_readwrite("m_old", &Adaptation::m_old)
		.def_readwrite("dm", &Adaptation::dm)
		.def_readwrite("ps", &Adaptation::ps)
		.def_readwrite("dz", &Adaptation::dz)
		.def_readwrite("dd", &Adaptation::dd)
		.def_readwrite("expected_length_z", &Adaptation::expected_length_z)
		.def("adapt_evolution_paths", &Adaptation::adapt_evolution_paths,
			py::arg("pop"),
			py::arg("weights"),
			py::arg("stats"),
			py::arg("settings"),
			py::arg("mu"),
			py::arg("lamb"))
		.def("adapt_evolution_paths_innner", &Adaptation::adapt_evolution_paths_inner,
			py::arg("pop"),
			py::arg("weights"),
			py::arg("stats"),
			py::arg("settings"),
			py::arg("mu"),
			py::arg("lamb"))
		.def("adapt_matrix", &Adaptation::adapt_matrix,
			py::arg("weights"),
			py::arg("modules"),
			py::arg("population"),
			py::arg("mu"),
			py::arg("settings"),
			py::arg("stats"))
		.def("restart", &Adaptation::restart, py::arg("settings"), py::arg("sigma"))
		.def("distance", &Adaptation::distance, py::arg("u"), py::arg("v"))
		.def("distance_from_center", &Adaptation::distance_from_center, py::arg("x"))
		.def("compute_y", &Adaptation::compute_y, py::arg("zi"))
		.def("invert_x", &Adaptation::invert_x, py::arg("xi"), py::arg("sigma"))
		.def("invert_y", &Adaptation::invert_y, py::arg("yi"))
		.def("__repr__", [] (Adaptation& dyn)
			{
				std::stringstream ss;
				ss << std::boolalpha;
				ss << "<Adaptation";
				ss << " m: " << dyn.m.transpose();
				ss << " m_old: " << dyn.m_old.transpose();
				ss << " dm: " << dyn.dm.transpose();
				ss << " ps: " << dyn.ps.transpose();
				ss << " dd: " << dyn.dd;
				ss << " expected_length_z: " << dyn.expected_length_z;
				ss << ">";
				return ss.str(); });


	py::class_<None, Adaptation, std::shared_ptr<None>>(m, "NoAdaptation")
		.def(py::init<size_t, Vector, Float>(), py::arg("dimension"), py::arg("x0"), py::arg("expected_length_z"))
		.def("__repr__", [] (None& dyn)
			{
				std::stringstream ss;
				ss << std::boolalpha;
				ss << "<NoAdaptation";
				ss << " m: " << dyn.m.transpose();
				ss << " m_old: " << dyn.m_old.transpose();
				ss << " dm: " << dyn.dm.transpose();
				ss << " ps: " << dyn.ps.transpose();
				ss << " dd: " << dyn.dd;
				ss << " expected_length_z: " << dyn.expected_length_z;
				ss << ">";
				return ss.str(); });


	py::class_<CovarianceAdaptation, Adaptation, std::shared_ptr<CovarianceAdaptation>>(m, "CovarianceAdaptation")
		.def(py::init<size_t, Vector, Float>(), py::arg("dimension"), py::arg("x0"), py::arg("expected_length_z"))
		.def_readwrite("pc", &CovarianceAdaptation::pc)
		.def_readwrite("d", &CovarianceAdaptation::d)
		.def_readwrite("B", &CovarianceAdaptation::B)
		.def_readwrite("C", &CovarianceAdaptation::C)
		.def_readwrite("A", &CovarianceAdaptation::A)
		.def_readwrite("inv_root_C", &CovarianceAdaptation::inv_root_C)
		.def_readwrite("hs", &CovarianceAdaptation::hs)
		.def("adapt_covariance_matrix", &CovarianceAdaptation::adapt_covariance_matrix,
			py::arg("weights"),
			py::arg("modules"),
			py::arg("population"),
			py::arg("mu"))
		.def("perform_eigendecomposition", &CovarianceAdaptation::perform_eigendecomposition, py::arg("stats"))
		.def("adapt_ps", &CovarianceAdaptation::adapt_ps, py::arg("weights"))
		.def("__repr__", [] (CovarianceAdaptation& dyn)
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
				ss << " expected_length_z: " << dyn.expected_length_z;
				ss << " hs: " << dyn.hs;
				ss << ">";
				return ss.str(); });

	py::class_<SeparableAdaptation, Adaptation, std::shared_ptr<SeparableAdaptation>>(m, "SeparableAdaptation")
		.def(py::init<size_t, Vector, Float>(), py::arg("dimension"), py::arg("x0"), py::arg("expected_length_z"))
		.def_readwrite("c", &SeparableAdaptation::c)
		.def_readwrite("pc", &SeparableAdaptation::pc)
		.def_readwrite("d", &SeparableAdaptation::d)
		.def("__repr__", [] (SeparableAdaptation& dyn)
			{
				std::stringstream ss;
				ss << std::boolalpha;
				ss << "<SeparableAdaptation";
				ss << " m: " << dyn.m.transpose();
				ss << " m_old: " << dyn.m_old.transpose();
				ss << " dm: " << dyn.dm.transpose();
				ss << " pc: " << dyn.pc.transpose();
				ss << " ps: " << dyn.ps.transpose();
				ss << " d: " << dyn.d.transpose();
				ss << " c: " << dyn.c.transpose();
				ss << " expected_length_z: " << dyn.expected_length_z;
				ss << " hs: " << dyn.hs;
				ss << ">";
				return ss.str(); });

	py::class_<MatrixAdaptation, Adaptation, std::shared_ptr<MatrixAdaptation>>(m, "MatrixAdaptation")
		.def(py::init<size_t, Vector, Float>(), py::arg("dimension"), py::arg("x0"), py::arg("expected_length_z"))
		.def_readwrite("M", &MatrixAdaptation::M)
		.def_readwrite("M_inv", &MatrixAdaptation::M_inv)
		.def("__repr__", [] (MatrixAdaptation& dyn)
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
				ss << " expected_length_z: " << dyn.expected_length_z;
				ss << ">";
				return ss.str(); });

	py::class_<CholeskyAdaptation, Adaptation, std::shared_ptr<CholeskyAdaptation>>(m, "CholeskyAdaptation")
		.def(py::init<size_t, Vector, Float>(), py::arg("dimension"), py::arg("x0"), py::arg("expected_length_z"))
		.def_readwrite("A", &CholeskyAdaptation::A)
		.def_readwrite("pc", &CholeskyAdaptation::pc);

	py::class_<SelfAdaptation, Adaptation, std::shared_ptr<SelfAdaptation>>(m, "SelfAdaptation")
		.def(py::init<size_t, Vector, Float>(), py::arg("dimension"), py::arg("x0"), py::arg("expected_length_z"))
		.def_readwrite("A", &SelfAdaptation::A)
		.def_readwrite("C", &SelfAdaptation::C);

	py::class_<CovarianceNoEigvAdaptation, CovarianceAdaptation, std::shared_ptr<CovarianceNoEigvAdaptation>>(m, "CovarianceNoEigvAdaptation")
		;

	py::class_<NaturalGradientAdaptation, Adaptation, std::shared_ptr<NaturalGradientAdaptation>>(m, "NaturalGradientAdaptation")
		.def(py::init<size_t, Vector, Float, Float>(), py::arg("dimension"), py::arg("x0"), py::arg("expected_length_z"), py::arg("sigma"))
		.def_readwrite("A", &NaturalGradientAdaptation::A)
		.def_readwrite("A_inv", &NaturalGradientAdaptation::A_inv)
		.def_readwrite("G", &NaturalGradientAdaptation::G)
		.def_readwrite("sigma_g", &NaturalGradientAdaptation::sigma_g)
		.def("compute_gradients", &NaturalGradientAdaptation::compute_gradients, py::arg("pop"),
			py::arg("weights"),
			py::arg("stats"),
			py::arg("settings"),
			py::arg("mu"),
			py::arg("lamb")
		)
		;
}

void define_parameters(py::module& main)
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
		.def_readwrite("repelling_restart", &Modules::repelling_restart)
		.def_readwrite("matrix_adaptation", &Modules::matrix_adaptation)
		.def_readwrite("center_placement", &Modules::center_placement)
		.def_readwrite("sample_transformation", &Modules::sample_transformation)
		.def("__repr__", [] (Modules& mod)
			{ return to_string(mod); });

	py::class_<Solution>(m, "Solution")
		.def(py::init<>())
		.def_readwrite("x", &Solution::x)
		.def_readwrite("y", &Solution::y)
		.def_readwrite("t", &Solution::t)
		.def_readwrite("e", &Solution::e)
		.def("__repr__", &Solution::repr);

	py::class_<Stats>(m, "Stats")
		.def(py::init<>())
		.def_readwrite("t", &Stats::t)
		.def_readwrite("evaluations", &Stats::evaluations)
		.def_readwrite("current_avg", &Stats::current_avg)
		.def_readwrite("solutions", &Stats::solutions)
		.def_readwrite("centers", &Stats::centers)
		.def_readwrite("current_best", &Stats::current_best)
		.def_readwrite("global_best", &Stats::global_best)
		.def_readwrite("has_improved", &Stats::has_improved)
		.def_readwrite("success_ratio", &Stats::success_ratio)
		.def_readwrite("last_update", &Stats::last_update)
		.def_readwrite("n_updates", &Stats::n_updates)
		.def("__repr__", [] (Stats& stats)
			{
				std::stringstream ss;
				ss << std::boolalpha;
				ss << "<Stats";
				ss << " t: " << stats.t;
				ss << " e: " << stats.evaluations;
				ss << " best: " << stats.global_best;
				ss << " improved: " << stats.has_improved;
				ss << ">";
				return ss.str(); });

	py::class_<Weights>(m, "Weights")
		.def(
			py::init<size_t, size_t, size_t, Settings, Float>(),
			py::arg("dimension"),
			py::arg("mu0"),
			py::arg("lambda0"),
			py::arg("modules"),
			py::arg("expected_length_z")
		)
		.def_readwrite("mueff", &Weights::mueff)
		.def_readwrite("mueff_neg", &Weights::mueff_neg)
		.def_readwrite("c1", &Weights::c1)
		.def_readwrite("cmu", &Weights::cmu)
		.def_readwrite("cc", &Weights::cc)
		.def_readwrite("cs", &Weights::cs)
		.def_readwrite("damps", &Weights::damps)
		.def_readwrite("sqrt_cc_mueff", &Weights::sqrt_cc_mueff)
		.def_readwrite("sqrt_cs_mueff", &Weights::sqrt_cs_mueff)
		.def_readwrite("lazy_update_interval", &Weights::lazy_update_interval)
		.def_readwrite("expected_length_z", &Weights::expected_length_z)
		.def_readwrite("expected_length_ps", &Weights::expected_length_ps)
		.def_readwrite("beta", &Weights::beta)
		.def_readwrite("weights", &Weights::weights)
		.def_readwrite("positive", &Weights::positive)
		.def_readwrite("negative", &Weights::negative)
		.def("__repr__", [] (Weights& weights)
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
		.def(py::init<size_t, std::optional<Modules>, std::optional<Float>, size_to, size_to, std::optional<Float>,
			std::optional<size_t>, std::optional<size_t>, std::optional<Vector>,
			std::optional<Vector>, std::optional<Vector>,
			std::optional<Float>, std::optional<Float>, std::optional<Float>,
			std::optional<Float>, std::optional<Float>, std::optional<Float>,
			bool, bool>(),
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
			py::arg("damps") = std::nullopt,
			py::arg("acov") = std::nullopt,
			py::arg("verbose") = false,
			py::arg("always_compute_eigv") = false

		)
		.def_readonly("dim", &Settings::dim)
		.def_readonly("modules", &Settings::modules)
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
		.def_readwrite("damps", &Settings::damps)
		.def_readwrite("acov", &Settings::acov)
		.def_readwrite("verbose", &Settings::verbose)
		.def_readonly("volume", &Settings::volume)
		.def_readonly("one_plus_one", &Settings::one_plus_one)
		.def("__repr__", [] (Settings& settings)
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
		std::shared_ptr<matrix_adaptation::None>,
		std::shared_ptr<matrix_adaptation::SeparableAdaptation>,
		std::shared_ptr<matrix_adaptation::CholeskyAdaptation>,
		std::shared_ptr<matrix_adaptation::SelfAdaptation>,
		std::shared_ptr<matrix_adaptation::CovarianceNoEigvAdaptation>,
		std::shared_ptr<matrix_adaptation::NaturalGradientAdaptation>,
		std::shared_ptr<matrix_adaptation::CovarianceAdaptation>
	>;

	py::class_<Parameters, std::shared_ptr<Parameters>>(main, "Parameters")
		.def(py::init<size_t>(), py::arg("dimension"))
		.def(py::init<Settings>(), py::arg("settings"))
		.def("adapt", &Parameters::adapt)
		.def("start", &Parameters::start, py::arg("objective"))
		.def("perform_restart", &Parameters::perform_restart, py::arg("objective"),
			py::arg("sigma") = std::nullopt)
		.def_readwrite("settings", &Parameters::settings)
		.def_readwrite("mu", &Parameters::mu)
		.def_readwrite("lamb", &Parameters::lambda)
		.def_property(
			"adaptation",
			[] (Parameters& self) -> AdaptationType
			{
				switch (self.settings.modules.matrix_adaptation)
				{
					case MatrixAdaptationType::MATRIX:
						return std::dynamic_pointer_cast<matrix_adaptation::MatrixAdaptation>(self.adaptation);
					case MatrixAdaptationType::NONE:
						return std::dynamic_pointer_cast<matrix_adaptation::None>(self.adaptation);
					case MatrixAdaptationType::SEPARABLE:
						return std::dynamic_pointer_cast<matrix_adaptation::SeparableAdaptation>(self.adaptation);
					case MatrixAdaptationType::CHOLESKY:
						return std::dynamic_pointer_cast<matrix_adaptation::CholeskyAdaptation>(self.adaptation);
					case MatrixAdaptationType::CMSA:
						return std::dynamic_pointer_cast<matrix_adaptation::SelfAdaptation>(self.adaptation);
					case MatrixAdaptationType::COVARIANCE_NO_EIGV:
						return std::dynamic_pointer_cast<matrix_adaptation::CovarianceNoEigvAdaptation>(self.adaptation);
					case MatrixAdaptationType::NATURAL_GRADIENT:
						return std::dynamic_pointer_cast<matrix_adaptation::NaturalGradientAdaptation>(self.adaptation);
					default:
					case MatrixAdaptationType::COVARIANCE:
						return std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(self.adaptation);
				}
			},
			[] (Parameters& self, std::shared_ptr<matrix_adaptation::Adaptation> adaptation)
			{
				self.adaptation = adaptation;
			})
		.def_readwrite("criteria", &Parameters::criteria)
		.def_readwrite("stats", &Parameters::stats)
		.def_readwrite("weights", &Parameters::weights)
		.def_readwrite("pop", &Parameters::pop)
		.def_readwrite("old_pop", &Parameters::old_pop)
		.def_readwrite("sampler", &Parameters::sampler)
		.def_readwrite("mutation", &Parameters::mutation)
		.def_readwrite("selection", &Parameters::selection)
		.def_readwrite("restart_strategy", &Parameters::restart_strategy)
		.def_readwrite("repelling", &Parameters::repelling)
		.def_readwrite("bounds", &Parameters::bounds)
		.def_readwrite("center_placement", &Parameters::center_placement);
}

void define_bounds(py::module& main)
{
	auto m = main.def_submodule("bounds");
	using namespace bounds;

	py::class_<BoundCorrection, std::shared_ptr<BoundCorrection>>(m, "BoundCorrection")
		.def_readwrite("lb", &BoundCorrection::lb)
		.def_readwrite("ub", &BoundCorrection::ub)
		.def_readwrite("db", &BoundCorrection::db)
		.def_readwrite("diameter", &BoundCorrection::diameter)
		.def_readwrite("has_bounds", &BoundCorrection::has_bounds)
		.def_readonly("n_out_of_bounds", &BoundCorrection::n_out_of_bounds)
		.def("correct", &BoundCorrection::correct,
			py::arg("population"), py::arg("m"))
		.def("delta_out_of_bounds", &BoundCorrection::delta_out_of_bounds, py::arg("xi"), py::arg("oob"))	
		.def("is_out_of_bounds", &BoundCorrection::is_out_of_bounds, py::arg("xi"))
		;

	py::class_<Resample, BoundCorrection, std::shared_ptr<Resample>>(m, "Resample")
		.def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"));

	py::class_<NoCorrection, BoundCorrection, std::shared_ptr<NoCorrection>>(m, "NoCorrection")
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

void define_mutation(py::module& main)
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
		.def(py::init<parameters::Mirror, size_t, Float>(),
			py::arg("mirror"),
			py::arg("mu"),
			py::arg("seq_cuttoff_factor") = 1.0)
		.def("break_conditions", &SequentialSelection::break_conditions,
			py::arg("i"),
			py::arg("f"),
			py::arg("fopt"),
			py::arg("mirror"));

	py::class_<NoSequentialSelection, SequentialSelection, std::shared_ptr<NoSequentialSelection>>(m, "NoSequentialSelection")
		.def(py::init<parameters::Mirror, size_t, Float>(),
			py::arg("mirror"),
			py::arg("mu"),
			py::arg("seq_cuttoff_factor") = 1.0);

	py::class_<SigmaSampler, std::shared_ptr<SigmaSampler>>(m, "SigmaSampler")
		.def(py::init<Float>(), py::arg("dimension"))
		.def("sample", &SigmaSampler::sample, py::arg("sigma"), py::arg("population"), py::arg("tau"));

	py::class_<NoSigmaSampler, SigmaSampler, std::shared_ptr<NoSigmaSampler>>(m, "NoSigmaSampler")
		.def(py::init<Float>(), py::arg("dimension"));

	py::class_<Strategy, std::shared_ptr<Strategy>>(m, "Strategy")
		.def(
			py::init<
			std::shared_ptr<ThresholdConvergence>,
			std::shared_ptr<SequentialSelection>,
			std::shared_ptr<SigmaSampler>,
			Float
			>(),
			py::arg("threshold_convergence"),
			py::arg("sequential_selection"),
			py::arg("sigma_sampler"),
			py::arg("sigma0"))
		.def("adapt", &Strategy::adapt, py::arg("weights"),
			py::arg("dynamic"),
			py::arg("population"),
			py::arg("old_population"),
			py::arg("stats"),
			py::arg("lamb"))
		.def(
			"mutate", &CSA::mutate, py::arg("objective"),
			py::arg("n_offspring"),
			py::arg("parameters"))
		.def_readwrite("threshold_convergence", &Strategy::tc)
		.def_readwrite("sequential_selection", &Strategy::sq)
		.def_readwrite("sigma_sampler", &Strategy::ss)
		.def_readwrite("sigma", &Strategy::sigma)
		.def_readwrite("s", &Strategy::s)
		;

	py::class_<CSA, Strategy, std::shared_ptr<CSA>>(m, "CSA");
	py::class_<TPA, Strategy, std::shared_ptr<TPA>>(m, "TPA")
		.def_readwrite("a_tpa", &TPA::a_tpa)
		.def_readwrite("b_tpa", &TPA::b_tpa)
		.def_readwrite("rank_tpa", &TPA::rank_tpa);

	py::class_<MSR, Strategy, std::shared_ptr<MSR>>(m, "MSR");
	py::class_<PSR, Strategy, std::shared_ptr<PSR>>(m, "PSR")
		.def_readwrite("success_ratio", &PSR::success_ratio);

	py::class_<XNES, Strategy, std::shared_ptr<XNES>>(m, "XNES");
	py::class_<MXNES, Strategy, std::shared_ptr<MXNES>>(m, "MXNES");
	py::class_<LPXNES, Strategy, std::shared_ptr<LPXNES>>(m, "LPXNES");
	py::class_<SR, Strategy, std::shared_ptr<SR>>(m, "SR");
	py::class_<SA, Strategy, std::shared_ptr<SA>>(m, "SA");


}

void define_population(py::module& main)
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
		.def_readwrite("n", &Population::n)
		.def_readwrite("t", &Population::t);
}

class constants_w
{};

void define_constants(py::module& m)
{
	py::class_<constants_w>(m, "constants")
		.def_property_static(
			"cache_max_doubles",
			[] (py::object)
			{ return constants::cache_max_doubles; },
			[] (py::object, size_t a)
			{ constants::cache_max_doubles = a; })
		.def_property_static(
			"cache_min_samples",
			[] (py::object)
			{ return constants::cache_min_samples; },
			[] (py::object, size_t a)
			{ constants::cache_min_samples = a; })
		.def_property_static(
			"cache_samples",
			[] (py::object)
			{ return constants::cache_samples; },
			[] (py::object, bool a)
			{ constants::cache_samples = a; })
		.def_property_static(
			"clip_sigma",
			[] (py::object)
			{ return constants::clip_sigma; },
			[] (py::object, bool a)
			{ constants::clip_sigma = a; })
		.def_property_static(
			"use_box_muller",
			[] (py::object)
			{ return constants::use_box_muller; },
			[] (py::object, bool a)
			{ constants::use_box_muller = a; })
		;
}

struct PyCriterion : restart::Criterion
{
	PyCriterion(const std::string& name) : restart::Criterion(name) {}

	void update(const parameters::Parameters& p) override
	{
		PYBIND11_OVERRIDE_PURE(void, restart::Criterion, update, p);
	}

	void on_reset(const parameters::Parameters& p) override
	{
		PYBIND11_OVERRIDE(void, restart::Criterion, on_reset, p);
	}
};

void define_restart_criteria(py::module& main)
{
	auto m = main.def_submodule("restart");
	using namespace restart;

	py::class_<Criterion, PyCriterion, std::shared_ptr<Criterion>>(m, "Criterion")
		.def(py::init<std::string>(), py::arg("name"))
		.def("on_reset", &Criterion::on_reset, py::arg("parameters"))
		.def("update", &Criterion::update, py::arg("parameters"))
		.def("reset", &Criterion::reset, py::arg("parameters"))
		.def_readwrite("met", &Criterion::met)
		.def_readwrite("name", &Criterion::name)
		.def_readwrite("last_restart", &Criterion::last_restart)
		.def("__repr__", [] (Criterion& self)
			{ return "<" + self.name + " met: " + std::to_string(self.met) + ">"; });
	;

	py::class_<ExceededMaxIter, Criterion, std::shared_ptr<ExceededMaxIter>>(m, "ExceededMaxIter")
		.def(py::init<>())
		.def_readwrite("max_iter", &ExceededMaxIter::max_iter);

	py::class_<NoImprovement, Criterion, std::shared_ptr<NoImprovement>>(m, "NoImprovement")
		.def(py::init<>())
		.def_readwrite("n_bin", &NoImprovement::n_bin)
		.def_readwrite("best_fitnesses", &NoImprovement::best_fitnesses);

	py::class_<MaxSigma, Criterion, std::shared_ptr<MaxSigma>>(m, "MaxSigma")
		.def(py::init<>())
		.def_readwrite_static("tolerance", &MaxSigma::tolerance);

	py::class_<MinSigma, Criterion, std::shared_ptr<MinSigma>>(m, "MinSigma")
		.def(py::init<>())
		.def_readwrite_static("tolerance", &MinSigma::tolerance);

	py::class_<UnableToAdapt, Criterion, std::shared_ptr<UnableToAdapt>>(m, "UnableToAdapt")
		.def(py::init<>());

	py::class_<FlatFitness, Criterion, std::shared_ptr<FlatFitness>>(m, "FlatFitness")
		.def(py::init<>())
		.def_readwrite("max_flat_fitness", &FlatFitness::max_flat_fitness)
		.def_readwrite("flat_fitness_index", &FlatFitness::flat_fitness_index)
		.def_readwrite("flat_fitnesses", &FlatFitness::flat_fitnesses);

	py::class_<TolX, Criterion, std::shared_ptr<TolX>>(m, "TolX")
		.def(py::init<>())
		.def_readwrite("tolx_vector", &TolX::tolx_vector)
		.def_readwrite_static("tolerance", &TolX::tolerance)
		;

	py::class_<MaxDSigma, Criterion, std::shared_ptr<MaxDSigma>>(m, "MaxDSigma")
		.def(py::init<>())
		.def_readwrite_static("tolerance", &MaxDSigma::tolerance);

	py::class_<MinDSigma, Criterion, std::shared_ptr<MinDSigma>>(m, "MinDSigma")
		.def(py::init<>())
		.def_readwrite_static("tolerance", &MinDSigma::tolerance);

	py::class_<ConditionC, Criterion, std::shared_ptr<ConditionC>>(m, "ConditionC")
		.def(py::init<>())
		.def_readwrite_static("tolerance", &ConditionC::tolerance);

	py::class_<NoEffectAxis, Criterion, std::shared_ptr<NoEffectAxis>>(m, "NoEffectAxis")
		.def(py::init<>())
		.def_readwrite_static("tolerance", &NoEffectAxis::tolerance)
		;

	py::class_<NoEffectCoord, Criterion, std::shared_ptr<NoEffectCoord>>(m, "NoEffectCoord")
		.def(py::init<>())
		.def_readwrite_static("tolerance", &NoEffectCoord::tolerance);

	py::class_<Stagnation, Criterion, std::shared_ptr<Stagnation>>(m, "Stagnation")
		.def(py::init<>())
		.def_readwrite("n_stagnation", &Stagnation::n_stagnation)
		.def_readwrite("median_fitnesses", &Stagnation::median_fitnesses)
		.def_readwrite("best_fitnesses", &Stagnation::best_fitnesses)
		.def_readwrite_static("tolerance", &Stagnation::tolerance);

	py::class_<Criteria>(m, "Criteria")
		.def_readwrite("items", &Criteria::items)
		.def("reset", &Criteria::reset, py::arg("parameters"))
		.def("update", &Criteria::update, py::arg("parameters"))
		.def("reason", &Criteria::reason)
		.def("any", &Criteria::any);

	py::class_<TooMuchRepelling, Criterion, std::shared_ptr<TooMuchRepelling>>(m, "TooMuchRepelling")
		.def(py::init<>())
		.def_readwrite_static("tolerance", &TooMuchRepelling::tolerance);

}

void define_restart_strategy(py::module& main)
{
	auto m = main.def_submodule("restart");
	using namespace restart;

	py::class_<Strategy, std::shared_ptr<Strategy>>(m, "Strategy")
		// .def("evaluate", &Strategy::evaluate, py::arg("objective"), py::arg("parameters"))
		// .def_readwrite("criteria", &Strategy::criteria)
		.def("update", &Strategy::update, py::arg("parameters"));
	;

	py::class_<IPOP, Strategy, std::shared_ptr<IPOP>>(m, "IPOP")
		// .def(py::init<Float, Float, Float>(), py::arg("sigma"), py::arg("dimension"), py::arg("lamb"))
		.def_readwrite("ipop_factor", &IPOP::ipop_factor);

	py::class_<BIPOP, Strategy, std::shared_ptr<BIPOP>>(m, "BIPOP")
		// .def(py::init<Float, Float, Float, Float, size_t>(), py::arg("sigma"), py::arg("dimension"), py::arg("lamb"), py::arg("mu"), py::arg("budget"))
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

void define_cmaes(py::module& m)
{
	py::class_<ModularCMAES>(m, "ModularCMAES")
		.def(py::init<std::shared_ptr<parameters::Parameters>>(), py::arg("parameters"))
		.def(py::init<size_t>(), py::arg("dimension"))
		.def(py::init<parameters::Settings>(), py::arg("settings"))
		.def("recombine", &ModularCMAES::recombine)
		.def("mutate", &ModularCMAES::mutate, py::arg("objective"))
		.def("select", &ModularCMAES::select)
		.def("adapt", &ModularCMAES::adapt)
		.def("step", &ModularCMAES::step, py::arg("objective"))
		.def("__call__", &ModularCMAES::operator(), py::arg("objective"))
		.def("run", &ModularCMAES::operator(), py::arg("objective"))
		.def("break_conditions", &ModularCMAES::break_conditions)
		.def_readonly("p", &ModularCMAES::p);
}

void define_es(py::module& main)
{
	auto m = main.def_submodule("es");
	parameters::Modules default_modules;
	using namespace es;
	py::class_<OnePlusOneES, std::shared_ptr<OnePlusOneES>>(m, "OnePlusOneES")
		.def(
			py::init<
			size_t,
			Vector,
			Float,
			Float,
			size_t,
			Float,
			parameters::Modules>(),
			py::arg("d"),
			py::arg("x0"),
			py::arg("f0"),
			py::arg("sigma0") = 1.0,
			py::arg("budget") = 10'000,
			py::arg("target") = 1e-8,
			py::arg("modules") = default_modules)
		.def("__call__", &OnePlusOneES::operator())
		.def("step", &OnePlusOneES::step)
		.def("sample", &OnePlusOneES::sample)
		.def_readwrite("d", &OnePlusOneES::d)
		.def_readwrite("sigma", &OnePlusOneES::sigma)
		.def_readwrite("decay", &OnePlusOneES::decay)
		.def_readwrite("x", &OnePlusOneES::x)
		.def_readwrite("f", &OnePlusOneES::f)
		.def_readwrite("t", &OnePlusOneES::t)
		.def_readwrite("budget", &OnePlusOneES::budget)
		.def_readwrite("target", &OnePlusOneES::target)
		.def_readwrite("sampler", &OnePlusOneES::sampler)
		.def_readwrite("rejection_sampling", &OnePlusOneES::rejection_sampling)
		.def_readwrite("corrector", &OnePlusOneES::corrector);

	py::class_<MuCommaLambdaES, std::shared_ptr<MuCommaLambdaES>>(m, "MuCommaLambdaES")
		.def(
			py::init<
			size_t,
			Vector,
			Float,
			size_t,
			Float,
			parameters::Modules>(),
			py::arg("d"),
			py::arg("x0"),
			py::arg("sigma0") = 1.0,
			py::arg("budget") = 10'000,
			py::arg("target") = 1e-8,
			py::arg("modules") = default_modules)
		.def("__call__", &MuCommaLambdaES::operator())
		.def("step", &MuCommaLambdaES::step)
		.def("sample", &MuCommaLambdaES::sample)
		.def_readwrite("d", &MuCommaLambdaES::d)
		.def_readwrite("lamb", &MuCommaLambdaES::lambda)
		.def_readwrite("mu", &MuCommaLambdaES::mu)

		.def_readwrite("sigma", &MuCommaLambdaES::sigma)
		.def_readwrite("m", &MuCommaLambdaES::m)

		.def_readwrite("X", &MuCommaLambdaES::X)
		.def_readwrite("S", &MuCommaLambdaES::S)
		.def_readwrite("f", &MuCommaLambdaES::f)

		.def_readwrite("tau", &MuCommaLambdaES::tau)
		.def_readwrite("tau_i", &MuCommaLambdaES::tau_i)
		.def_readwrite("mu_inv", &MuCommaLambdaES::mu_inv)

		.def_readwrite("f_min", &MuCommaLambdaES::f_min)
		.def_readwrite("x_min", &MuCommaLambdaES::x_min)
		.def_readwrite("t", &MuCommaLambdaES::t)
		.def_readwrite("e", &MuCommaLambdaES::e)
		.def_readwrite("budget", &MuCommaLambdaES::budget)
		.def_readwrite("target", &MuCommaLambdaES::target)
		.def_readwrite("sampler", &MuCommaLambdaES::sampler)
		.def_readwrite("sigma_sampler", &MuCommaLambdaES::sigma_sampler)
		.def_readwrite("rejection_sampling", &MuCommaLambdaES::rejection_sampling)
		.def_readwrite("corrector", &MuCommaLambdaES::corrector);
}

PYBIND11_MODULE(cmaescpp, m)
{
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
