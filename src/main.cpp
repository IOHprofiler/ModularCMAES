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
static bool rotated = false;
static functions::ObjectiveFunction fun_t = functions::ObjectiveFunction::SPHERE;
static size_t budget = dim * 100000;


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


void run_modcma(parameters::MatrixAdaptationType mat_t, parameters::StepSizeAdaptation ssa)
{
	rng::set_seed(412);
	parameters::Modules m;
	m.matrix_adaptation = mat_t;
	m.ssa = ssa;
	m.active = false;
	m.sampler = parameters::BaseSampler::HALTON;
	m.restart_strategy = parameters::RestartStrategyType::RESTART;
	m.sample_transformation = parameters::SampleTranformerType::CAUCHY;
	m.elitist = false;
	m.sequential_selection = true;
	m.threshold_convergence = true;
	m.weights = parameters::RecombinationWeights::EQUAL;
	m.repelling_restart = true;
	
	parameters::Settings settings(
		dim, 
		m, 
		-std::numeric_limits<double>::infinity(),
		std::nullopt, 
		budget, 
		2.0,
		27,
		17
	);
	auto p = std::make_shared<parameters::Parameters>(settings);
	auto cma = ModularCMAES(p);

	Timer t;
	FunctionType f = Ellipse(dim, rotated, fun_t);
	while (cma.step(f))
	{
		
		std::cout << "evals: " << cma.p->stats.evaluations << "/" << budget << ": ";
		std::cout << "iters: " << cma.p->stats.t << ": ";
		std::cout << "sigma: " << cma.p->mutation->sigma << ": ";
		std::cout << "best_y: " << cma.p->stats.global_best.y;
		std::cout << "	n_resamples: " << cma.p->repelling->attempts;
			
		std::cout << std::endl;

		if (cma.p->stats.global_best.y < 1e-9)
			break;
	}

	std::cout << "modcmaes: " << parameters::to_string(mat_t) << std::defaultfloat;
	std::cout << " - " << parameters::to_string(ssa);
	if (m.active)
		std::cout << " ACTIVE";
	
	if (m.elitist)
		std::cout << " ELITIST";

	std::cout << "\nfunction: " << functions::to_string(fun_t) << " " << dim << "D";
	if (rotated)
		std::cout << " (rotated)";
	std::cout << "\nevals: " << cma.p->stats.evaluations << "/" << budget << std::endl;
	std::cout << "iters: " << cma.p->stats.t << std::endl;
	std::cout << "updates: " << cma.p->stats.n_updates << "\n" << std::scientific << std::setprecision(3);
	std::cout << "sigma: " << cma.p->mutation->sigma << std::endl;
	std::cout << "best_y: " << cma.p->stats.global_best.y << std::endl;
	std::cout << "solved: " << std::boolalpha << (cma.p->stats.global_best.y < 1e-8) << std::endl;
}

int main()
{
	auto ssa = parameters::StepSizeAdaptation::MSR;
	
	//run_modcma(parameters::MatrixAdaptationType::NONE, ft, ssa);
	//run_modcma(parameters::MatrixAdaptationType::SEPERABLE, ft, ssa);
	//run_modcma(parameters::MatrixAdaptationType::MATRIX, ft, ssa);



	//run_modcma(parameters::MatrixAdaptationType::CHOLESKY, ft, ssa);
	//run_modcma(parameters::MatrixAdaptationType::COVARIANCE, ft, ssa);
	//run_modcma(parameters::MatrixAdaptationType::COVARIANCE_NO_EIGV, ft, ssa);
	run_modcma(parameters::MatrixAdaptationType::NATURAL_GRADIENT, ssa);
}