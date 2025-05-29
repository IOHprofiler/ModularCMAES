#include "c_maes.hpp"
#include <chrono>

using namespace std::placeholders;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;



struct Function
{
	size_t evals = 0;

	Float operator()(const Vector& x)
	{
		evals++;
		const auto x_shift = (x.array() - 1.).matrix();
		return functions::ellipse(x_shift);
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
		std::cout << "Time elapsed: " << static_cast<Float>(ms_int.count()) / 1000.0 << "s\n";
	}
};


// int main()
// {
// 	rng::set_seed(42);
// 	const size_t dim = 100;
// 	const size_t budget = dim * 1000;

// 	parameters::Modules m;
// 	//m.matrix_adaptation = parameters::MatrixAdaptationType::MATRIX;
// 	m.sample_transformation = parameters::SampleTranformerType::SCALED_UNIFORM;
// 	m.bound_correction = parameters::CorrectionMethod::NONE;

// 	parameters::Settings settings(dim, m, -std::numeric_limits<double>::infinity(), 
// 		std::nullopt, budget, 2.0);
// 	auto p = std::make_shared<parameters::Parameters>(settings);

// 	auto cma = ModularCMAES(p);

// 	Timer t;
// 	FunctionType f = Function();
// 	while (cma.step(f))
// 	{
// 		//std::cout << cma.p->stats << std::endl;
// 		//std::cout << cma.p->mutation->sigma << std::endl;
// 		//auto sr = std::dynamic_pointer_cast<mutation::SR>(cma.p->mutation);
// 		//std::cout << "p_succ: " << sr->success_ratio << ", " << sr->max_success_ratio << std::endl;
// 	}
// 	std::cout << cma.p->stats.evaluations << std::endl;
// 	std::cout << cma.p->stats.t << std::endl;
// 	std::cout << cma.p->stats.n_updates << std::endl;
// 	std::cout << cma.p->stats << std::endl;
// }
