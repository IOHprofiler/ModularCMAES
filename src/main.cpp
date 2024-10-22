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

	double operator()(const Vector& x)
	{
		evals++;
		const auto x_shift = (x.array() - 1.).matrix();
		return functions::rastrigin(x_shift);
	}
};


template <typename Callable>
void call(Callable& o)
{
	static_assert(std::is_invocable_r_v<double, Callable, Vector>, "Incorrect objective function type");
	const double result = o(Vector::Ones(10));
	std::cout << result;
}

struct Timer
{
	std::chrono::time_point<std::chrono::steady_clock> t1;
	Timer() : t1(high_resolution_clock::now()) {}

	~Timer()
	{
		const auto t2 = high_resolution_clock::now();
		const auto ms_int = duration_cast<milliseconds>(t2 - t1);
		std::cout << "Time elapsed: " << static_cast<double>(ms_int.count()) / 1000.0 << "s\n";
	}
};


int main()
{
	rng::set_seed(43);
	const auto dim = 2;
	parameters::Settings settings(dim);
	//settings.target = 1e-8;
	settings.modules.sampler = parameters::BaseSampler::GAUSSIAN;
	settings.modules.mirrored = parameters::Mirror::NONE;
	settings.modules.orthogonal = true;
	settings.modules.restart_strategy = parameters::RestartStrategyType::RESTART;
	settings.modules.repelling_restart = true;
	settings.verbose = true;

	const ModularCMAES cma (settings);
	Function f;
	cma(static_cast<FunctionType>(f));


	for (const auto& p: cma.p->repelling->archive)
		std::cout << p.solution << std::endl;



}
