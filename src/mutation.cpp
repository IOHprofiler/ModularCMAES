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

	void SigmaSampler::apply_integer_bounds(parameters::Parameters &p)
	{
		const Array &Cdiag = p.adaptation->coordinate_wise_variances;
		for (const auto &iidx : p.settings.integer_variables)
		{
			const Float Cii = std::max(Cdiag[iidx], Float(1e-16));
			const Float lb_sigma = p.weights.int_lb_sigma / std::sqrt(Cii);

			for (size_t i = 0; i < p.pop.n; i++)
				p.pop.S(iidx, i) = std::max(p.pop.S(iidx, i), lb_sigma);
		}
	}

	void SigmaSampler::sample(const Float sigma, parameters::Parameters &p)
	{
		for (size_t i = 0; i < p.pop.n; i++)
		{
			const auto z = sampler();
			p.pop.S.col(i) = (sigma * (p.weights.beta * z.array()).exp()).matrix();
		}
		apply_integer_bounds(p);
	}

	void NoSigmaSampler::sample(const Float sigma, parameters::Parameters &p)
	{
		p.pop.S.setConstant(sigma);
		apply_integer_bounds(p);
	}

	void Strategy::mutate(FunctionType &objective, const size_t n_offspring, parameters::Parameters &p)
	{
		ss->sample(sigma, p);
		p.bounds->n_out_of_bounds = 0;
		p.repelling->prepare_sampling(p);

		for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_offspring); ++i)
		{
			size_t n_rej = 0;
			do
			{
				p.pop.t(i) = p.stats.t;
				const auto &zi = (*p.sampler)();
				const auto &zi_scaled = p.mutation->tc->scale(
					zi, p.settings.diameter, p.settings.budget, p.stats.evaluations);
				p.pop.Z.col(i).noalias() = zi_scaled;
				p.pop.Y.col(i).noalias() = p.adaptation->compute_y(p.pop.Z.col(i));
				p.pop.X.col(i).array() = p.pop.Y.col(i).array() * p.pop.S.col(i).array() + p.adaptation->m.array();
				p.bounds->correct(i, p);

			} while (
				(p.settings.modules.bound_correction == parameters::CorrectionMethod::RESAMPLE &&
				 n_rej++ < 5 * p.settings.dim && p.bounds->is_out_of_bounds(p.pop.X.col(i), p.settings).any()) ||
				p.repelling->is_rejected(p.pop.X.col(i), p));

			p.pop.X_transformed.col(i) = p.coordinate_mapping->transform(p.pop.X.col(i));

			p.pop.f(i) = objective(p.pop.X_transformed.col(i));
			p.stats.evaluations++;
			if (sq->break_conditions(i, p.pop.f(i), p.stats.global_best.y, p.settings.modules.mirrored))
			{
				// TODO: We should renormalize the weights
				break;
			}
		}
	}

	void CSA::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					Population &pop,
					const Population &old_pop, const parameters::Stats &stats, const size_t lambda)

	{
		Float l = (w.cs / w.damps) * ((adaptation->ps.norm() / w.expected_length_z) - 1);
		// Clamping as seen in pycma
		l = std::min(Float{1.0}, std::max(l, Float{-1.0}));
		sigma *= std::exp(l);
	}

	void TPA::mutate(FunctionType &objective, const size_t n_offspring_, parameters::Parameters &p)
	{
		Strategy::mutate(objective, n_offspring_, p);

		const auto x_pos = p.coordinate_mapping->transform(p.adaptation->m + (p.mutation->sigma * p.adaptation->dm));
		const auto x_neg = p.coordinate_mapping->transform(p.adaptation->m + (p.mutation->sigma * -p.adaptation->dm));
		const auto f_pos = objective(x_pos);
		const auto f_neg = objective(x_neg);
		p.stats.update_best(x_pos, f_pos);
		p.stats.update_best(x_neg, f_neg);
		p.stats.evaluations += 2;
		this->rank_tpa = f_neg < f_pos ? -a_tpa : a_tpa + b_tpa;
	}

	void TPA::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					Population &pop,
					const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		s = ((1.0 - w.cs) * s) + (w.cs * this->rank_tpa);
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
			s = ((1.0 - w.cs) * s) + (w.cs * z);
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

			if (idx.size() != n)
			{
				idx.resize(n);
				oidx.resize(n);
				std::iota(idx.begin(), idx.end(), 0);
				std::iota(oidx.begin(), oidx.end(), 0);
			}

			utils::sort_index_inplace(combined, idx);
			utils::sort_index_inplace(idx, oidx);

			Float delta_r = 0.0;
			for (size_t i = 0; i < n; i++)
			{
				Float r = oidx[i];
				Float r_old = oidx[n + i];
				delta_r += (r_old - r);
			}

			const auto z = delta_r / std::pow(n, 2) - success_ratio;
			s = (1.0 - w.cs) * s + (w.cs * z);
			sigma *= std::exp(s / (2.0 - (2.0 / adaptation->dd)));
		}
	}

	void XNES::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					 Population &pop,
					 const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		if (const auto dynamic = std::dynamic_pointer_cast<matrix_adaptation::NaturalGradientAdaptation>(adaptation))
		{
			sigma *= std::exp(w.cs / 2.0 * dynamic->sigma_g);
			return;
		}

		const Float z = ((pop.Z).colwise().squaredNorm().array() - adaptation->dd).matrix() * w.clipped();
		sigma *= std::exp((w.cs / std::sqrt(adaptation->dd)) * z);
	}

	void MXNES::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					  Population &pop,
					  const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		const Float delta = (w.mueff * adaptation->dz.squaredNorm() - adaptation->dd);

		sigma *= std::exp((w.cs / adaptation->dd) * delta);
	}

	void LPXNES::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					   Population &pop,
					   const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		const auto logS = (pop.S.array() / sigma).log();
		const Vector per_sample = logS.colwise().mean().transpose();
		const Float rel_log = per_sample.dot(w.clipped());
		sigma *= std::exp(w.cs * rel_log);
	}

	void SR::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
				   Population &pop,
				   const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		sigma *= std::exp((1 / w.damps) * ((stats.success_ratio - tgt_success_ratio) / (1.0 - tgt_success_ratio)));
	}

	void SA::mutate(FunctionType &objective, const size_t n_offspring, parameters::Parameters &p)
	{
		Strategy::mutate(objective, n_offspring, p);
		mean_sigma = std::exp(p.pop.S.array().log().mean());
	}

	void SA::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
				   Population &pop,
				   const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		const auto &sigma_l = pop.S.topRows(w.positive.rows());
		const auto &log_sigma_dim = sigma_l.array().log().rowwise().mean();
		const Float log_sigma = (w.positive.array() * log_sigma_dim).sum();
		sigma = std::exp(log_sigma) / mean_sigma;
	}

	std::shared_ptr<Strategy> get(const parameters::Modules &m, const size_t mu, const Float d, const Float sigma)
	{
		using namespace parameters;

		auto tc = m.threshold_convergence
					  ? std::make_shared<ThresholdConvergence>()
					  : std::make_shared<NoThresholdConvergence>();

		auto sq = m.sequential_selection
					  ? std::make_shared<SequentialSelection>(m.mirrored, mu)
					  : std::make_shared<NoSequentialSelection>(m.mirrored, mu);

		auto ss = (m.sample_sigma or m.ssa == StepSizeAdaptation::LPXNES or m.ssa == StepSizeAdaptation::SA)
					  ? std::make_shared<SigmaSampler>(d)
					  : std::make_shared<NoSigmaSampler>(d);

		switch (m.ssa)
		{
		case StepSizeAdaptation::TPA:
			return std::make_shared<TPA>(tc, sq, ss, sigma);
		case StepSizeAdaptation::MSR:
			return std::make_shared<MSR>(tc, sq, ss, sigma);
		case StepSizeAdaptation::XNES:
			return std::make_shared<XNES>(tc, sq, ss, sigma);
		case StepSizeAdaptation::MXNES:
			return std::make_shared<MXNES>(tc, sq, ss, sigma);
		case StepSizeAdaptation::LPXNES:
			return std::make_shared<LPXNES>(tc, sq, ss, sigma);
		case StepSizeAdaptation::PSR:
			return std::make_shared<PSR>(tc, sq, ss, sigma);
		case StepSizeAdaptation::SR:
			return std::make_shared<SR>(tc, sq, ss, sigma);
		case StepSizeAdaptation::SA:
			return std::make_shared<SA>(tc, sq, ss, sigma);
		default:
		case StepSizeAdaptation::CSA:
			return std::make_shared<CSA>(tc, sq, ss, sigma);
		}
	}
}
