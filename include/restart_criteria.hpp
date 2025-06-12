#pragma once

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
	struct Parameters;
}

namespace restart
{
	struct Criterion
	{
		bool met;
		std::string name;
		size_t last_restart;

		Criterion(const std::string& name) : met(false), name(name), last_restart(0) {}

		virtual ~Criterion() = default;

		void reset(const parameters::Parameters& p);

		virtual void update(const parameters::Parameters& p) = 0;

		virtual void on_reset(const parameters::Parameters& p) {};
	};

	using vCriteria = std::vector<std::shared_ptr<restart::Criterion>>;

	struct Criteria
	{
		Criteria(const vCriteria& c) : items(c) {}

		void update(const parameters::Parameters& p)
		{
			for (auto& c : items)
			{
				c->update(p);
			}
		}

		void reset(const parameters::Parameters& p)
		{
			for (auto& c : items)
				c->reset(p);
		}

		bool any() const
		{
			for (const auto& c : items)
				if (c->met)
					return true;
			return false;
		}

		std::string reason() const
		{
			std::string res;
			for (const auto& c : items)
				if (c->met)
					res += c->name + "; ";
			return res;
		}

		vCriteria items;

		static Criteria get(const parameters::Modules modules);
	};

	struct ExceededMaxIter : Criterion
	{
		size_t max_iter;
		ExceededMaxIter() : Criterion("ExceededMaxIter") {}
		void update(const parameters::Parameters& p) override;
		void on_reset(const parameters::Parameters& p) override;
	};

	struct NoImprovement : Criterion
	{
		size_t n_bin;
		std::vector<Float> best_fitnesses;
		NoImprovement() : Criterion("NoImprovement") {}
		void update(const parameters::Parameters& p) override;
		void on_reset(const parameters::Parameters& p) override;
	};

	struct MaxSigma : Criterion
	{
		static inline Float tolerance = 1e4;
		MaxSigma() : Criterion("MaxSigma") {}
		void update(const parameters::Parameters& p) override;
	};

	struct MinSigma : Criterion
	{
		static inline Float tolerance = 1e-20;
		MinSigma() : Criterion("MinSigma") {}
		void update(const parameters::Parameters& p) override;
	};

	struct UnableToAdapt : Criterion
	{
		UnableToAdapt() : Criterion("UnableToAdapt") {}
		void update(const parameters::Parameters& p) override;
	};

	struct FlatFitness : Criterion
	{
		size_t max_flat_fitness;
		size_t flat_fitness_index;
		Eigen::Array<int, Eigen::Dynamic, 1> flat_fitnesses;

		FlatFitness() : Criterion("FlatFitness") {}
		void update(const parameters::Parameters& p) override;
		void on_reset(const parameters::Parameters& p) override;
	};

	struct TolX : Criterion
	{
		static inline Float tolerance = 1e-11;
		Vector tolx_vector;
		TolX() : Criterion("TolX") {}
		void update(const parameters::Parameters& p) override;
		void on_reset(const parameters::Parameters& p) override;
	};

	struct MaxDSigma : Criterion
	{
		static inline Float tolerance = std::pow(10., 20.);
		MaxDSigma() : Criterion("MaxDSigma") {}
		void update(const parameters::Parameters& p) override;
	};

	struct MinDSigma : Criterion
	{
		static inline Float tolerance = std::pow(10., -20.);
		MinDSigma() : Criterion("MinDSigma") {}
		void update(const parameters::Parameters& p) override;
	};

	struct ConditionC : Criterion
	{
		static inline Float tolerance = std::pow(10., 14.);
		ConditionC() : Criterion("ConditionC") {}
		void update(const parameters::Parameters& p) override;
	};

	struct NoEffectAxis : Criterion
	{
		static inline Float tolerance = 10.0 * std::numeric_limits<Float>::epsilon();
		NoEffectAxis() : Criterion("NoEffectAxis") {}
		void update(const parameters::Parameters& p) override;
	};

	struct NoEffectCoord : Criterion
	{
		static inline Float tolerance = 10.0 * std::numeric_limits<Float>::epsilon();
		NoEffectCoord() : Criterion("NoEffectCoord") {}
		void update(const parameters::Parameters& p) override;
	};

	struct Stagnation : Criterion
	{
		static inline Float tolerance = 0.3;

		size_t n_stagnation;
		std::vector<Float> median_fitnesses;
		std::vector<Float> best_fitnesses;
		Stagnation() : Criterion("Stagnation") {}
		void update(const parameters::Parameters& p) override;
		void on_reset(const parameters::Parameters& p) override;
	};

	struct TooMuchRepelling : Criterion
	{
		static inline Float tolerance = 50;
		TooMuchRepelling() : Criterion("TooMuchRepelling") {}
		void update(const parameters::Parameters& p) override;
	};

}