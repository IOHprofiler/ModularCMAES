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
		double manhattan(const Vector &u, const Vector &v);
		double euclidian(const Vector &u, const Vector &v);
		double mahanolobis(const Vector &u, const Vector &v, const Matrix &C_inv);

		bool hill_valley_test(
			const Solution &u,
			const Solution &v,
			FunctionType &f,
			const size_t n_evals);

		bool hill_valley_test_p(
			const Solution &u,
			const Solution &v,
			FunctionType &f,
			const size_t n_evals,
			parameters::Parameters &p);
	}

	struct TabooPoint
	{
		Solution solution;
		double radius;
		double shrinkage;
		int n_rep;
		double criticality;
		Matrix C;
		Matrix C_inv;

		TabooPoint(
			const Solution &s,
			const double radius,
			const Matrix &C, const Matrix &C_inv) : solution(s),
													radius(radius),
													shrinkage(std::pow(0.99, 1. / static_cast<double>(s.x.size()))),
													n_rep(1),
													criticality(0.0),
													C(C),
													C_inv(C_inv) {}

		/**
		 * \brief Rejection rule for a taboo point for a given xi
		 * \param xi the sampled solution
		 * \param p parameters
		 * \param attempts determines the amount of shrinkage applied; radius = pow(shrinkage, attempts) * radius
		 * \return
		 */
		bool rejects(const Vector &xi, const parameters::Parameters &p, const int attempts) const;

		bool shares_basin(FunctionType &objective, const Solution &sol, parameters::Parameters &p) const;

		void calculate_criticality(const parameters::Parameters &p);
	};

	struct Repelling
	{
		std::vector<TabooPoint> archive;
		int attempts = 0;
		double coverage = 20.0;
		Matrix C;
		Matrix C_inv;

		virtual ~Repelling() = default;

		/**
		 * \brief Application of the rejection rule for a sampled solution
		 * xi to all the points in the current archive
		 * \param xi
		 * \param p
		 * \return
		 */
		virtual bool is_rejected(const Vector &xi, parameters::Parameters &p);

		/**
		 * \brief Update the archive of points
		 * \param p
		 */
		virtual void update_archive(FunctionType &objective, parameters::Parameters &p);

		/**
		 * \brief Hook before sampling starts
		 */
		virtual void prepare_sampling(const parameters::Parameters &p);
	};

	struct NoRepelling final : Repelling
	{

		bool is_rejected(const Vector &xi, parameters::Parameters &p) override
		{
			return false;
		}

		void update_archive(FunctionType &objective, parameters::Parameters &p) override
		{
		}

		void prepare_sampling(const parameters::Parameters &p) override
		{
		}
	};

	inline std::shared_ptr<Repelling> get(const parameters::Modules &m)
	{
		if (m.repelling_restart)
			return std::make_shared<Repelling>();
		return std::make_shared<NoRepelling>();
	}
}
