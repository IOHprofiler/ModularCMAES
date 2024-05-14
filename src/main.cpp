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
	rng::set_seed(42);
	const size_t dim = 2;

	constants::shuffle_cache_max_doubles = 0;
	constants::shuffle_cache_min_samples = 6;

	parameters::Settings settings(dim);
	settings.modules.sampler = parameters::BaseSampler::SOBOL;
	parameters::Parameters p(settings);

	const auto sampler = std::dynamic_pointer_cast<sampling::Sobol>(p.sampler);
	std::cout << sampler->cache.n_samples << std::endl;

	for(size_t j =0; j < 3; j++)
	{
		for (size_t i = 0; i < constants::shuffle_cache_min_samples; i++)
			std::cout << sampler->operator()().transpose() << std::endl;
		std::cout << std::endl;
	}
}
