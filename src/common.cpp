#include "common.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& x)
{
	for (auto& xi : x)
		os << xi << ' ';
	return os;
}


namespace constants
{
	double tolup_sigma = std::pow(10., 20.);
	double tol_condition_cov = pow(10., 14.);
	double tol_min_sigma = 1e-8;
	double stagnation_quantile = 0.3;
	double sigma_threshold = 1e-4;
	size_t shuffle_cache_max_doubles = 2'000'000;
	size_t shuffle_cache_min_samples = 128;
}

namespace utils
{
	std::vector<size_t> sort_indexes(const Vector& v)
	{
		std::vector<size_t> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);

		std::stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2)
			{
				return v[i1] < v[i2];
			});

		return idx;
	}

	std::vector<size_t> sort_indexes(const std::vector<size_t>& v)
	{
		std::vector<size_t> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);

		std::stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2)
			{
				return v[i1] < v[i2];
			});

		return idx;
	}

	void hstack(Matrix& X, const Matrix& Y)
	{
		X.conservativeResize(Eigen::NoChange, X.cols() + Y.cols());
		X.rightCols(Y.cols()) = Y;
	}

	void vstack(Matrix& X, const Matrix& Y)
	{
		X.conservativeResize(X.rows() + Y.rows(), Eigen::NoChange);
		X.bottomRows(Y.rows()) = Y;
	}

	void concat(Vector& x, const Vector& y)
	{
		x.conservativeResize(x.rows() + y.rows(), Eigen::NoChange);
		x.bottomRows(y.rows()) = y;
	}

	std::pair<double, size_t> compute_ert(const std::vector<size_t>& running_times, const size_t budget)
	{
		size_t successful_runs = 0, total_rt = 0;

		for (const auto& rt : running_times)
		{
			if (rt < budget)
				successful_runs++;
			total_rt += rt;
		}
		return { static_cast<double>(total_rt) / successful_runs, successful_runs };
	}
}

namespace rng
{
	int SEED = std::random_device()();
	std::mt19937 GENERATOR(SEED);

	void set_seed(const int seed)
	{
		SEED = seed;
		GENERATOR.seed(seed);
		srand(seed);
	}

	int random_integer(const int l, const int h)
	{
		std::uniform_int_distribution<> distrib(l, std::max(l, h - 1));
		return distrib(GENERATOR);
	}

	void Shuffler::advance()
	{
		do {
			seed = (seed * multiplier + offset) % modulus;
		} while (seed > n);
	}

	size_t Shuffler::next()
	{
		if (found > 0)
			advance();
		found++;
		return start + seed;
	}


	CachedShuffleSequence::CachedShuffleSequence(const size_t d) :
		dim(d),
		n_samples(std::max(constants::shuffle_cache_min_samples, utils::nearest_power_of_2(constants::shuffle_cache_max_doubles / d))),
		cache(n_samples * d, 0.0),
		shuffler(n_samples)
	{
	}

	void CachedShuffleSequence::fill(const std::vector<double>& c)
	{
		std::copy(c.begin(), c.end(), cache.begin());
	}


	Vector CachedShuffleSequence::get_index(const size_t idx)
	{
		return Eigen::Map<Vector>(cache.data() + (idx * dim), static_cast<Eigen::Index>(dim));
	}

	Vector CachedShuffleSequence::next()
	{
		return get_index(shuffler.next());
	}




}

namespace functions
{
	double sphere(const Vector& x)
	{
		double res = 0;
		for (auto& xi : x)
			res += xi * xi;
		return res;
	}

	double rastrigin(const Vector& x)
	{
		constexpr double a = 10.;
		constexpr double pi2 = 2. * M_PI;
		double res = 0;
		for (auto& xi : x)
			res += xi * xi - a * std::cos(pi2 * xi);
		return a * static_cast<double>(x.size()) + res;
	}
}
