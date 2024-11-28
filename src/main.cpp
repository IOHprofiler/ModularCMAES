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


parameters::Stats do_single_run(const int seed, const int dim, const bool verbose = false)
{
	rng::set_seed(seed);
	parameters::Settings settings(dim);
	
	settings.target = 1e-8;
	settings.modules.sampler = parameters::BaseSampler::GAUSSIAN;
	settings.modules.mirrored = parameters::Mirror::NONE;
	settings.modules.orthogonal = true;
	settings.modules.restart_strategy = parameters::RestartStrategyType::RESTART;
	settings.modules.matrix_adaptation = parameters::MatrixAdaptationType::SEPERABLE;
	settings.modules.repelling_restart = true;
	settings.verbose = false;

	const ModularCMAES cma (settings);
	Function f;
	cma(static_cast<FunctionType>(f));

	if (verbose) 
	{
		std::cout << cma.p->stats << std::endl;
		std::cout << "size of archive: " << cma.p->repelling->archive.size() << std::endl;
 		for (const auto& p: cma.p->repelling->archive)
			std::cout << p.solution << std::endl;
	}
	return cma.p->stats;
}


int main()
{
	constants::do_hill_valley = true;
	constants::repelling_coverage = 200.0;

	const int dim = 20;
	const double tgt = 1e-8;
	const int n_runs = 1;
	const bool verbose = false;
	int n_suc = 0;
	int n_evals = 0;

	for (size_t i = 1; i < n_runs + 1; i++) {
		auto stats = do_single_run(43 + i * 10, dim, verbose);
		//std::cout << stats << std::endl;
		const bool success = stats.global_best.y < tgt;
		n_suc += success;
		n_evals += stats.evaluations;
	}

	std::cout << "ERT: ";
	if (n_suc == 0)
		std::cout << "inf\n";
	else
		std::cout << n_evals / n_suc << "\n";

}		
