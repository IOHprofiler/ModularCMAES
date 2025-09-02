#pragma once

#include "common.hpp"
#include "population.hpp"
#include "sampling.hpp"
#include "modules.hpp"

namespace parameters
{
	struct Stats;
	struct Parameters;
	struct Weights;
	struct Strategy;
	struct Modules;
}
namespace matrix_adaptation
{
	struct Adaptation;
}
namespace bounds
{
	struct BoundCorrection;
}

namespace mutation
{

	struct ThresholdConvergence
	{
		Float init_threshold = 0.1;
		Float decay_factor = 0.995;
		virtual Vector scale(const Vector& zi, const Float diameter, const size_t budget, const size_t evaluations);
	};

	struct NoThresholdConvergence : ThresholdConvergence
	{
		Vector scale(const Vector& zi, const Float diameter, const size_t budget, const size_t evaluations) override
		{
			return zi;
		}
	};

	class SequentialSelection
	{
		Float seq_cutoff_factor;
		size_t seq_cutoff;

	public:
		SequentialSelection(const parameters::Mirror& m, const size_t mu, const Float seq_cutoff_factor = 1.0) : seq_cutoff_factor(m == parameters::Mirror::PAIRWISE ? std::max(Float{ 2. }, seq_cutoff_factor) : seq_cutoff_factor),
			seq_cutoff(static_cast<size_t>(mu* seq_cutoff_factor))
		{}
		virtual bool break_conditions(const size_t i, const Float f, Float fopt, const parameters::Mirror& m);
	};

	struct NoSequentialSelection : SequentialSelection
	{
		using SequentialSelection::SequentialSelection;

		bool break_conditions(const size_t i, const Float f, Float fopt, const parameters::Mirror& m) override { return false; }
	};

	struct SigmaSampler
	{
		sampling::GaussianTransformer sampler;

		SigmaSampler(const Float d) : sampler{ std::make_shared<sampling::Uniform>(1) }
		{}

		virtual void sample(const Float sigma, Population& pop, const Float tau)
		{
			sampler.sampler->d = pop.s.rows();
			pop.s.noalias() = (sigma * (tau * sampler().array()).exp()).matrix().eval();
		}
	};

	struct NoSigmaSampler : SigmaSampler
	{
		using SigmaSampler::SigmaSampler;

		void sample(const Float sigma, Population& pop, const Float tau) override
		{
			pop.s.setConstant(sigma);
		}
	};

	struct Strategy
	{
		std::shared_ptr<ThresholdConvergence> tc;
		std::shared_ptr<SequentialSelection> sq;
		std::shared_ptr<SigmaSampler> ss;
		Float sigma;
		Float s = 0;

		Strategy(
			const std::shared_ptr<ThresholdConvergence>& threshold_covergence,
			const std::shared_ptr<SequentialSelection>& sequential_selection,
			const std::shared_ptr<SigmaSampler>& sigma_sampler,
			const Float sigma0) : tc(threshold_covergence), sq(sequential_selection), ss(sigma_sampler), sigma(sigma0)
		{}

		virtual void mutate(FunctionType& objective, const size_t n_offspring, parameters::Parameters& p);

		virtual void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) {};
	};

	struct CSA : Strategy
	{
		using Strategy::Strategy;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
	};

	struct TPA : Strategy
	{
		using Strategy::Strategy;

		Float a_tpa = 0.5;
		Float b_tpa = 0.0;
		Float rank_tpa = 0.0;

		void mutate(FunctionType& objective, const size_t n_offspring, parameters::Parameters& p) override;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
	};

	struct MSR : Strategy
	{
		using Strategy::Strategy;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
	};

	struct PSR : Strategy
	{
		Float success_ratio = .25;

		Vector combined;

		using Strategy::Strategy;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
	};

	struct XNES : Strategy
	{
		using Strategy::Strategy;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
	};

	struct MXNES : Strategy
	{
		using Strategy::Strategy;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
	};

	struct LPXNES : Strategy
	{
		using Strategy::Strategy;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
	};


	struct SR : Strategy
	{
		constexpr static Float tgt_success_ratio = 2.0 / 11.0;

		using Strategy::Strategy;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;
	};


	struct SA : Strategy
	{
		using Strategy::Strategy;

		void adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation, Population& pop,
			const Population& old_pop, const parameters::Stats& stats, const size_t lambda) override;

		void mutate(FunctionType& objective, const size_t n_offspring, parameters::Parameters& p) override;

	private:
		Float mean_sigma;
	};



	std::shared_ptr<Strategy> get(const parameters::Modules& m, const size_t mu, const Float d, const Float sigma);

}