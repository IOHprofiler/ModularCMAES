#include "c_maes.hpp"
#include <chrono>


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

int main()
{
	using namespace std::placeholders;
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	rng::set_seed(10);

	Solution s1({}, 10.0);
	Solution s2({}, 1.0);

	assert(s1 > s2);


	constexpr size_t dim = 2;
	parameters::Settings s(dim);
	s.budget = 10'00 * dim;
	s.modules.elitist = true;
	s.modules.matrix_adaptation = parameters::MatrixAdaptationType::COVARIANCE;
	s.modules.restart_strategy = parameters::RestartStrategyType::RESTART;
	s.modules.repelling_restart = false;
	s.modules.threshold_convergence = false;
	s.modules.bound_correction = parameters::CorrectionMethod::NONE;

	auto p = std::make_shared<parameters::Parameters>(s);
	Function f;

	ModularCMAES cma(p);

	FunctionType func = std::bind(&Function::operator(), &f, _1);
	auto t1 = high_resolution_clock::now();
	cma(func);
	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;
	std::cout << ms_int.count() << "ms\n";
	std::cout << "completed\n";
	std::cout << p->stats.evaluations << ", " << f.evals << '\n';
	std::cout << "global best: " << p->stats.global_best << '\n';

	for (auto& x : p->stats.solutions)
	{
		std::cout << x << '\n';
	}
}
