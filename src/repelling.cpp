#include "repelling.hpp"
#include "parameters.hpp"


namespace repelling
{
	namespace distance
	{
		double euclidian(const Vector& u, const Vector& v)
		{
			return (u - v).norm();
		}

		double mahanolobis(const Vector& u, const Vector& v, const Matrix& C_inv)
		{
			return 0.0;
		}
	}

	bool TabooPoint::rejects(const Vector& xi, const parameters::Parameters& p, const int attempts) const
	{
		const double rejection_radius = std::pow(shrinkage, attempts) * delta;
		double delta_xi;

		if (
			p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::NONE ||
			p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::MATRIX)
		{
			constexpr double stretch = 1.0;
			delta_xi = distance::euclidian(xi, solution.x) / (p.mutation->sigma * stretch);
		}
		else
		{
			using namespace matrix_adaptation;
			const std::shared_ptr<CovarianceAdaptation> dynamic = std::dynamic_pointer_cast<CovarianceAdaptation>(
				p.adaptation);
			delta_xi = distance::mahanolobis(xi, solution.x, dynamic->inv_C) / p.mutation->sigma;
		}

		if (delta_xi < rejection_radius)
			return true;

		return false;
	}

	void Repelling::update_archive(parameters::Parameters& p)
	{
		constexpr double delta0 = 1.0;
		archive.emplace_back(p.stats.current_best, delta0);
	}

	bool Repelling::is_rejected(const Vector& xi, parameters::Parameters& p)
	{
		if (!archive.empty())
			for (const auto& point : archive)
				if (point.rejects(xi, p, attempts))
				{
					attempts++;
					return true;
				}

		return false;
	}
}
