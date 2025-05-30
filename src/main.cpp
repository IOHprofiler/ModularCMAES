#include "c_maes.hpp"
#include "to_string.hpp"
#include <chrono>
#include <iomanip>

using namespace std::placeholders;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


static int dim = 50;
static bool rotated = true;
static size_t budget = dim * 4000;

struct Ellipse
{
	size_t evals = 0;
	Matrix R;

	Ellipse(const int dim, const bool rotated = false) :
		R{ rotated ? functions::random_rotation_matrix(dim, 1) : Matrix::Identity(dim, dim) }
	{
	}

	Float operator()(const Vector& x)
	{
		evals++;
		const auto x_shift = R * (x.array() - 1.).matrix();
		return functions::rosenbrock(x_shift);
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
		std::cout << "Time elapsed: " << static_cast<Float>(ms_int.count()) / 1000.0 << "s\n\n";
	}
};


void run_modcma(parameters::MatrixAdaptationType mat_t)
{
	rng::set_seed(42);
	parameters::Modules m;
	m.matrix_adaptation = mat_t;
	m.elitist = true;
	parameters::Settings settings(
		dim, m, -std::numeric_limits<double>::infinity(),
		std::nullopt, budget, 2.0
	);
	auto p = std::make_shared<parameters::Parameters>(settings);
	auto cma = ModularCMAES(p);

	Timer t;
	FunctionType f = Ellipse(dim, rotated);
	while (cma.step(f))
	{
		//std::cout << cma.p->stats << std::endl;
		//std::cout << cma.p->mutation->sigma << std::endl;
		//auto sr = std::dynamic_pointer_cast<mutation::SR>(cma.p->mutation);
		//std::cout << "p_succ: " << sr->success_ratio << ", " << sr->max_success_ratio << std::endl;

		//if (cma.p->stats.current_best.y < 1e-8)
		//	break;

		// No rotation
		// e:    Stats t=549 e=5490
		// no-e: Stats t=594 e=5940
		// Rotation
		// e: Stats t = 559 e = 5590
		// no-e: Stats t=549 e=5490

		// Rosen
		// no rotation
		// e: Stats t = 617 e = 6170
		// noe: Stats t=625 e=6250 
		// rotation: 
		// e: Stats t=618 e=6180 
		// no-e Stats t=568 e=5680 
		// 
	}

	std::cout << "modcmaes: " << parameters::to_string(mat_t) << "\n" << std::defaultfloat;
	std::cout << "evals: " << cma.p->stats.evaluations << std::endl;
	std::cout << "iters: " << cma.p->stats.t << std::endl;
	std::cout << "updates: " << cma.p->stats.n_updates << std::endl;
	std::cout << "best_y: " << std::scientific << std::setprecision(3) << cma.p->stats.global_best.y << std::endl;
}

int main()
{
	run_modcma(parameters::MatrixAdaptationType::CHOLESKY);
	//run_modcma(parameters::MatrixAdaptationType::MATRIX);
	run_modcma(parameters::MatrixAdaptationType::COVARIANCE);
}