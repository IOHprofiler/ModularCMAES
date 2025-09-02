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
	size_t cache_max_doubles = 2'000'000;
	size_t cache_min_samples = 128;
	bool cache_samples = false;
	bool clip_sigma = false;
	bool use_box_muller = false;
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

	std::pair<Float, size_t> compute_ert(const std::vector<size_t>& running_times, const size_t budget)
	{
		size_t successful_runs = 0, total_rt = 0;

		for (const auto& rt : running_times)
		{
			if (rt < budget)
				successful_runs++;
			total_rt += rt;
		}
		return { static_cast<Float>(total_rt) / successful_runs, successful_runs };
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
		} while (seed >= n);
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
		n_samples(std::max(constants::cache_min_samples, utils::nearest_power_of_2(constants::cache_max_doubles / d))),
		cache(n_samples* d, 0.0),
		shuffler(n_samples)
	{
	}

	void CachedShuffleSequence::fill(const std::vector<Float>& c)
	{
		std::copy(c.begin(), c.end(), cache.begin());
	}

	void CachedShuffleSequence::transform(const std::function<Float(Float)>& f)
	{
		for (Float& i : cache) i = f(i);
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
	Float sphere(const Vector& x)
	{
		Float res = 0;
		for (auto& xi : x)
			res += xi * xi;
		return res;
	}

	Float rastrigin(const Vector& x)
	{
		constexpr Float a = 10.;
		constexpr Float pi2 = 2. * M_PI;
		Float res = 0;
		for (auto& xi : x)
			res += xi * xi - a * std::cos(pi2 * xi);
		return a * static_cast<Float>(x.size()) + res;
	}

	Float ellipse(const Vector& x)
	{
		Float res = 0;
		for (auto i = 0; i < x.size(); ++i)
			res += pow(1.0e6, static_cast<Float>(i) / (static_cast<Float>(x.size()) - 1)) * x(i) * x(i);
		return res;
	}

	Float rosenbrock(const Vector& x) {
		Float sum = 0.0;
		for (auto i = 0; i < x.size() - 1; ++i) {
			Float xi = x[i];
			Float xi1 = x[i + 1];
			Float term1 = 100.0 * std::pow(xi1 - xi * xi, 2);
			Float term2 = std::pow(1.0 - xi, 2);
			sum += term1 + term2;
		}
		return sum;
	}

	Matrix random_rotation_matrix(int n, int seed) {
		std::mt19937 gen(seed);
		std::normal_distribution<> d(0, 1);

		Matrix A(n, n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				A(i, j) = d(gen);

		Eigen::HouseholderQR<Matrix> qr(A);
		Matrix Q = qr.householderQ();

		if (Q.determinant() < 0) {
			Q.col(0) *= -1;
		}

		return Q;
	}



}
