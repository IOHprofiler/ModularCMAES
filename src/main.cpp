#include "c_maes.hpp"
#include "to_string.hpp"
#include <chrono>
#include <iomanip>

using namespace std::placeholders;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

static int dim = 5;
static bool rotated = true;
static functions::ObjectiveFunction fun_t = functions::ObjectiveFunction::RASTRIGIN;
static size_t budget = dim * 10000;


struct Ellipse
{
	size_t evals;
	Matrix R;
	FunctionType function;

	Ellipse(const int dim, const bool rotated, const functions::ObjectiveFunction ft) :
		evals(0),
		R{ rotated ? functions::random_rotation_matrix(dim, 1) : Matrix::Identity(dim, dim) },
		function(functions::get(ft))
	{
	}

	Float operator()(const Vector& x)
	{
		evals++;
		const auto x_shift = R * (x.array() - 1.).matrix();
		return function(x_shift);
	}
};


template <typename Callable>
void call(Callable& o)
{
	static_assert(std::is_invocable_r_v<Float, Callable, Vector>, "Incorrect objective function type");
	const Float result = o(Vector::Ones(10));
	std::cout << result;
}

struct Timer
{
	std::chrono::time_point<std::chrono::high_resolution_clock> t1;
	Timer() : t1(high_resolution_clock::now()) {}

	~Timer()
	{
		const auto t2 = high_resolution_clock::now();
		const auto ms_int = duration_cast<milliseconds>(t2 - t1);
		std::cout << "Time elapsed: " << std::defaultfloat << std::setprecision(5) << 
			static_cast<Float>(ms_int.count()) / 1000.0 << "s\n\n";
	}
};

struct Run {
	int budget_used;
	double fval;
	bool solved;
};


Run run_modcma(parameters::MatrixAdaptationType mat_t, parameters::StepSizeAdaptation ssa)
{
	//rng::set_seed(412);
	parameters::Modules m;
	m.matrix_adaptation = mat_t;
	m.ssa = ssa;
	//m.active = false;
	//m.sampler = parameters::BaseSampler::HALTON;
	//m.restart_strategy = parameters::RestartStrategyType::IPOP;
	//m.sample_transformation = parameters::SampleTranformerType::CAUCHY;
	//m.elitist = false;
	//m.sequential_selection = true;
	//m.threshold_convergence = true;
	//m.weights = parameters::RecombinationWeights::EQUAL;
	//m.repelling_restart = true;
	
	parameters::Settings settings(
		dim, 
		m, 
		-std::numeric_limits<double>::infinity(),
		std::nullopt, 
		budget, 
		2.0//,
		//500
		//1,
		//1
	);
	settings.verbose = true;
	auto p = std::make_shared<parameters::Parameters>(settings);
	auto cma = ModularCMAES(p);

	Timer t;
	FunctionType f = Ellipse(dim, rotated, fun_t);
	while (cma.step(f))
	{
		
	/*	std::cout << "evals: " << cma.p->stats.evaluations << "/" << budget << ": ";
		std::cout << "iters: " << cma.p->stats.t << ": ";
		std::cout << "sigma: " << cma.p->mutation->sigma << ": ";
		std::cout << "best_y: " << cma.p->stats.global_best.y;
		std::cout << "	n_resamples: " << cma.p->repelling->attempts;
	    std::cout << std::endl;*/

		if (cma.p->stats.global_best.y < 1e-9)
			break;
	}

	std::cout << "modcmaes: " << parameters::to_string(settings.modules.matrix_adaptation) << std::defaultfloat;
	std::cout << " - " << parameters::to_string(settings.modules.ssa);
	if (m.active)
		std::cout << " ACTIVE";
	
	if (m.elitist)
		std::cout << " ELITIST";

	std::cout << "\nfunction: " << functions::to_string(fun_t) << " " << dim << "D";
	if (rotated)
		std::cout << " (rotated)";
	const Float budget_used = static_cast<Float>(cma.p->stats.evaluations) / static_cast<Float>(budget) * 100;
	std::cout << "\nevals: " << cma.p->stats.evaluations << "/" << budget;
	std::cout << " ~ (" << std::defaultfloat << std::setprecision(3) << budget_used << "%)" << std::endl;
	std::cout << "iters: " << cma.p->stats.t << std::endl;
	std::cout << "updates: " << cma.p->stats.n_updates << "\n" << std::scientific << std::setprecision(3);
	std::cout << "sigma: " << cma.p->mutation->sigma << std::endl;
	std::cout << "best_y: " << cma.p->stats.global_best.y << std::endl;
	std::cout << "solved: " << std::boolalpha << (cma.p->stats.global_best.y < 1e-8) << std::endl;
	return { 
		(int)cma.p->stats.evaluations, 
		cma.p->stats.global_best.y, 
		cma.p->stats.global_best.y < 1e-8 
	};
}

void ert_exp(parameters::MatrixAdaptationType mat_t, parameters::StepSizeAdaptation ssa, int n_runs)
{
	double rt = 0;
	int n_succ = 0;
	for (int i = 0; i < n_runs; i++) {
		auto run_dat = run_modcma(mat_t, ssa);
		rt += run_dat.budget_used;
		n_succ += run_dat.solved;
	}
	std::cout << "ERT:";
	if (n_succ == 0) 
	{
		std::cout << "inf";
	}
	else {
		std::cout << std::defaultfloat << rt / n_succ;
		std::cout << " SR: " << n_succ << "/" << n_runs;
	}
	std::cout << std::endl;
	

}

int main()
{
	auto ssa = parameters::StepSizeAdaptation::CSA;
	constants::use_box_muller = true;
	run_modcma(parameters::MatrixAdaptationType::COVARIANCE, ssa);
	//run_modcma(parameters::MatrixAdaptationType::SEPARABLE, ssa);/*
	//run_modcma(parameters::MatrixAdaptationType::MATRIX, ssa);
	//run_modcma(parameters::MatrixAdaptationType::CHOLESKY, ssa);
	//run_modcma(parameters::MatrixAdaptationType::COVARIANCE, ssa);
	//run_modcma(parameters::MatrixAdaptationType::COVARIANCE_NO_EIGV, ssa);
	//run_modcma(parameters::MatrixAdaptationType::NATURAL_GRADIENT, ssa);*/
}