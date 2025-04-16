#include "mutation.hpp"
#include "bounds.hpp"
#include "matrix_adaptation.hpp"
#include "parameters.hpp"

namespace mutation
{
	Vector ThresholdConvergence::scale(const Vector &zi, const Float diameter, const size_t budget,
									   const size_t evaluations)
	{
		const Float t = init_threshold * diameter * pow(static_cast<Float>(budget - evaluations) / static_cast<Float>(budget), decay_factor);

		if (const auto norm = zi.norm(); norm < t)
			return zi.array() * ((t + (t - norm)) / norm);
		return zi;
	}

	bool SequentialSelection::break_conditions(const size_t i, const Float f, Float fopt, const parameters::Mirror &m)
	{
		return (f < fopt) and (i >= seq_cutoff) and (m != parameters::Mirror::PAIRWISE or i % 2 == 0);
	}

	void CSA::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					Population &pop,
					const Population &old_pop, const parameters::Stats &stats, const size_t lambda)

	{
		sigma *= std::exp((cs / damps) * ((adaptation->ps.norm() / expected_length_z) - 1));
	}

	void CSA::mutate(FunctionType &objective, const size_t n_offspring, parameters::Parameters &p)
	{
		ss->sample(sigma, p.pop);
		p.bounds->n_out_of_bounds = 0;
		p.repelling->prepare_sampling(p);
		for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_offspring); ++i)
		{
			size_t n_rej = 0;
			do
			{
				p.pop.Z.col(i) = p.mutation->tc->scale((*p.sampler)(), p.bounds->diameter, p.settings.budget, p.stats.evaluations);
				p.pop.Y.col(i) = p.adaptation->compute_y(p.pop.Z.col(i));
				p.pop.X.col(i) = p.pop.Y.col(i) * p.pop.s(i) + p.adaptation->m;
				p.bounds->correct(i, p);
			} while (
				(p.settings.modules.bound_correction == parameters::CorrectionMethod::RESAMPLE && n_rej++ < 5*p.settings.dim && p.bounds->is_out_of_bounds(p.pop.X.col(i)).any()) || p.repelling->is_rejected(p.pop.X.col(i), p));
			
			p.pop.f(i) = objective(p.pop.X.col(i));
			p.stats.evaluations++;
			if (sq->break_conditions(i, p.pop.f(i), p.stats.global_best.y, p.settings.modules.mirrored))
				break;
		}
	}

	void TPA::mutate(FunctionType &objective, const size_t n_offspring_, parameters::Parameters &p)
	{
		CSA::mutate(objective, n_offspring_, p);

		const auto f_pos = objective(p.adaptation->m + (p.mutation->sigma * p.adaptation->dm));
		const auto f_neg = objective(p.adaptation->m + (p.mutation->sigma * -p.adaptation->dm));
		p.stats.evaluations += 2;
		this->rank_tpa = f_neg < f_pos ? -a_tpa : a_tpa + b_tpa;
	}

	void TPA::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					Population &pop,
					const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		s = ((1.0 - cs) * s) + (cs * this->rank_tpa);
		sigma *= std::exp(s);
	}

	//! Assumes the vector to be arready sorted
	Float median(const Vector &x)
	{
		if (x.size() % 2 == 0)
			return (x(x.size() / 2) + x(x.size() / 2 - 1)) / 2.0;
		return x(x.size() / 2);
	}

	void MSR::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					Population &pop,
					const Population &old_pop, const parameters::Stats &stats, const size_t lamb)
	{
		const auto n = std::min(pop.n_finite(), old_pop.n_finite());
		if (n != 0)
		{
			const Float lambda = static_cast<Float>(lamb);
			const Float k = (pop.f.array() < median(old_pop.f)).cast<Float>().sum();
			const auto z = (2.0 / lambda) * (k - ((lambda + 1.0) / 2.0));
			s = ((1.0 - cs) * s) + (cs * z);
			sigma *= std::exp(s / (2.0 - (2.0 / adaptation->dd)));
		}
	}

	//! Returns the indices of the elements of query in database
	Vector searchsorted(const Vector &query, const Vector &database)
	{
		Vector res(query.size());
		auto i = 0;

		for (const auto &xi : query)
		{
			auto it = std::find(std::begin(database), std::end(database), xi);
			res(i++) = static_cast<Float>(std::distance(std::begin(database), it));
		}
		return res;
	}

	void PSR::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					Population &pop,
					const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		const auto n = std::min(pop.n_finite(), old_pop.n_finite());
		if (n != 0)
		{
			combined.conservativeResize(n + n);
			combined.head(n) = pop.f.head(n);
			combined.tail(n) = old_pop.f.head(n);
			const auto idx = utils::sort_indexes(combined);
			const auto oidx = utils::sort_indexes(idx);

			Float delta_r = 0.0;
			for (size_t i = 0; i < n; i++)
			{
				Float r = oidx[i];
				Float r_old = oidx[n + i];
				delta_r += (r_old - r);
			}

			const auto z = delta_r / std::pow(n, 2) - success_ratio;
			s = (1.0 - cs) * s + (cs * z);
			sigma *= std::exp(s / (2.0 - (2.0 / adaptation->dd)));
		}
	}

	void XNES::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					 Population &pop,
					 const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		// const Float z = ((std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(adaptation)->inv_root_C *  .Y).colwise().norm().array().pow(2.) - adaptation->dd).matrix() * w.clipped();
		const Float z = ((pop.Z).colwise().norm().array().pow(2.) - adaptation->dd).matrix() * w.clipped();
		sigma *= std::exp((cs / std::sqrt(adaptation->dd)) * z);
	}

	void MXNES::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					  Population &pop,
					  const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		const auto n = std::min(pop.n_finite(), old_pop.n_finite());
		if (n != 0)
		{
			// const auto z = (w.mueff * std::pow((dynamic.inv_root_C * dynamic.dm).norm(), 2)) - dynamic.dd;
			const auto mu = pop.n - lambda;
			const auto dz = (pop.Z.leftCols(mu).array().rowwise() * w.positive.array().transpose()).rowwise().sum().matrix();
			const auto z = (w.mueff * std::pow(dz.norm(), 2)) - adaptation->dd;
			sigma *= std::exp((cs / adaptation->dd) * z);
		}
	}

	void LPXNES::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					   Population &pop,
					   const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		const auto z = std::exp(cs * pop.s.array().log().matrix().dot(w.clipped()));
		sigma = std::pow(sigma, 1.0 - cs) * z;
	}

	void SR::adapt(const parameters::Weights& w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
		Population& pop,
		const Population& old_pop, const parameters::Stats& stats, const size_t lambda)
	{
		sigma *= std::exp((1 / damps) * ((stats.success_ratio - tgt_success_ratio) / (1.0 - tgt_success_ratio)));
	}


	std::shared_ptr<Strategy> get(const parameters::Modules &m, const size_t mu, const Float mueff,
								  const Float d, const Float sigma, const std::optional<Float> cs0,
								  const Float expected_z)
	{
		using namespace parameters;

		auto tc = m.threshold_convergence
					  ? std::make_shared<ThresholdConvergence>()
					  : std::make_shared<NoThresholdConvergence>();

		auto sq = m.sequential_selection
					  ? std::make_shared<SequentialSelection>(m.mirrored, mu)
					  : std::make_shared<NoSequentialSelection>(m.mirrored, mu);

		auto ss = (m.sample_sigma or m.ssa == StepSizeAdaptation::LPXNES)
					  ? std::make_shared<SigmaSampler>(d)
					  : std::make_shared<NoSigmaSampler>(d);

		Float cs = cs0.value_or(0.3);
		Float damps = 0.0;

		switch (m.ssa)
		{
		case StepSizeAdaptation::TPA:
			return std::make_shared<TPA>(tc, sq, ss, cs, damps, sigma, expected_z);
		case StepSizeAdaptation::MSR:
			return std::make_shared<MSR>(tc, sq, ss, cs, damps, sigma, expected_z);
		case StepSizeAdaptation::XNES:
			cs = cs0.value_or(mueff / (2.0 * std::log(std::max(Float{2.}, d)) * sqrt(d)));
			return std::make_shared<XNES>(tc, sq, ss, cs, damps, sigma, expected_z);
		case StepSizeAdaptation::MXNES:
			cs = cs0.value_or(1.);
			return std::make_shared<MXNES>(tc, sq, ss, cs, damps, sigma, expected_z);
		case StepSizeAdaptation::LPXNES:
			cs = cs0.value_or(9.0 * mueff / (10.0 * sqrt(d)));
			return std::make_shared<LPXNES>(tc, sq, ss, cs, damps, sigma, expected_z);
		case StepSizeAdaptation::PSR:
			cs = cs0.value_or(.9);
			return std::make_shared<PSR>(tc, sq, ss, cs, damps, sigma, expected_z);
		case StepSizeAdaptation::SR:
			cs = cs0.value_or(1.0 / 12.0);
			damps = 1.0 + (d / 2.0);
			return std::make_shared<SR>(tc, sq, ss, cs, damps, sigma, expected_z);
		default:
		case StepSizeAdaptation::CSA:
			cs = cs0.value_or((mueff + 2.0) / (d + mueff + 5.0));
			damps = 1.0 + (2.0 * std::max(0.0, sqrt((mueff - 1.0) / (d + 1)) - 1) + cs);
			return std::make_shared<CSA>(tc, sq, ss, cs, damps, sigma, expected_z);
		}
	}
}
