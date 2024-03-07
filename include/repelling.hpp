#pragma once

#include "common.hpp"
#include <modules.hpp>

namespace parameters
{
	struct Parameters;
}

namespace repelling
{
	namespace distance
	{
		double euclidian(const Vector& u, const Vector& v);
		double mahanolobis(const Vector& u, const Vector& v);
	}

	struct TabooPoint
	{
		Solution solution;
		double delta;
		double shrinkage;

		TabooPoint(const Solution& s, const double delta) : solution(s),
			delta(delta), shrinkage(std::pow(0.99, 1. / static_cast<double>(s.x.size()))) {}

		/**
		 * \brief Rejection rule for a taboo point for a given xi
		 * \param xi the sampled solution
		 * \param p parameters
		 * \param attempts determines the amount of shrinkage applied; delta = pow(shrinkage, attempts) * delta
		 * \return 
		 */
		bool rejects(const Vector& xi, const parameters::Parameters& p, const int attempts) const;
	};

	struct Repelling
	{
		std::vector<TabooPoint> archive;
		int attempts = 0;

		virtual ~Repelling() = default;

		/**
		 * \brief Application of the rejection rule for a sampled solution
		 * xi to all the points in the current archive
		 * \param xi 
		 * \param p
		 * \return 
		 */
		virtual bool is_rejected(const Vector& xi, parameters::Parameters& p);

		/**
		 * \brief Update the archive of points
		 * \param p 
		 */
		virtual void update_archive(parameters::Parameters& p);
	};

	struct NoRepelling final : Repelling
	{
		bool is_rejected(const Vector& xi, parameters::Parameters& p)  override
		{
			return false;
		}

		void update_archive(parameters::Parameters& p) override
		{
		}
	};


	inline std::shared_ptr<Repelling> get(const parameters::Modules& m)
	{
		if (m.repelling_restart)
			return std::make_shared<Repelling>();
		return std::make_shared<NoRepelling>();
	}
}
