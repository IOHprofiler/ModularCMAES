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
		return functions::ellipse(x_shift);
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
	parameters::Modules m;
	parameters::Settings settings(dim, m, 1e-8, 1000, 1000, 2.0, 1);
	auto p = std::make_shared<parameters::Parameters> (settings);

	auto cma = ModularCMAES(p);

	FunctionType f = Function();
		
	while(cma.step(f))
	{
		std::cout << cma.p->stats << std::endl;
		std::cout << cma.p->mutation->sigma << std::endl;
		//auto sr = std::dynamic_pointer_cast<mutation::SR>(cma.p->mutation);
		//std::cout << "p_succ: " << sr->success_ratio << ", " << sr->max_success_ratio << std::endl;
	}


}
