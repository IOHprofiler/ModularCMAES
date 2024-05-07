#include "mutation.hpp"
#include "bounds.hpp"
#include "matrix_adaptation.hpp"
#include "parameters.hpp"

namespace mutation
{
	Vector ThresholdConvergence::scale(const Vector &zi, const double diameter, const size_t budget,
									   const size_t evaluations)
	{
		const double t = init_threshold * diameter * pow(
			static_cast<double>(budget - evaluations) / static_cast<double>(budget), decay_factor);

		if (const auto norm = zi.norm(); norm < t)
			return zi.array() * ((t + (t - norm)) / norm);
		return zi;
	}

	bool SequentialSelection::break_conditions(const size_t i, const double f, double fopt, const parameters::Mirror &m)
	{
		return (f < fopt) and (i >= seq_cutoff) and (m != parameters::Mirror::PAIRWISE or i % 2 == 0);
	}

	void CSA::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					Population &pop,
					const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		sigma *= std::exp((cs / damps) * ((adaptation->ps.norm() / adaptation->chiN) - 1));
	}

	void CSA::mutate(FunctionType &objective, const size_t n_offspring, parameters::Parameters &p)
	{
		ss->sample(sigma, p.pop);
		p.bounds->n_out_of_bounds = 0;
		p.repelling->prepare_sampling(p);

		for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_offspring); ++i)
		{
			do
			{
				p.pop.Z.col(i) = p.mutation->tc->scale((*p.sampler)(), p.bounds->diameter, p.settings.budget, p.stats.evaluations);

				p.pop.Y.col(i) = p.adaptation->compute_y(p.pop.Z.col(i));

				p.pop.X.col(i) = p.pop.Y.col(i) * p.pop.s(i) + p.adaptation->m;

				p.bounds->correct(i, p);
			} while (p.repelling->is_rejected(p.pop.X.col(i), p));

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
	double median(const Vector &x)
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
			const double lambda = static_cast<double>(lamb);
			const double k = (pop.f.array() < median(old_pop.f)).cast<double>().sum();
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
			res(i++) = static_cast<double>(std::distance(std::begin(database), it));
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

			double delta_r = 0.0;
			for (size_t i = 0; i < n; i++)
			{
				double r = oidx[i];
				double r_old = oidx[n + i];
				delta_r += (r_old - r);
			}

			const auto z = delta_r / std::pow(n, 2) - succes_ratio;
			s = (1.0 - cs) * s + (cs * z);
			sigma *= std::exp(s / (2.0 - (2.0 / adaptation->dd)));
		}
	}

	void XNES::adapt(const parameters::Weights &w, std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
					 Population &pop,
					 const Population &old_pop, const parameters::Stats &stats, const size_t lambda)
	{
		// const double z = ((std::dynamic_pointer_cast<matrix_adaptation::CovarianceAdaptation>(adaptation)->inv_root_C *  .Y).colwise().norm().array().pow(2.) - adaptation->dd).matrix() * w.clipped();
		const double z = ((pop.Z).colwise().norm().array().pow(2.) - adaptation->dd).matrix() * w.clipped();
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

	std::shared_ptr<Strategy> get(const parameters::Modules &m, const size_t mu, const double mueff,
								  const double d, const double sigma, const std::optional<double> cs0)
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

		double cs = cs0.value_or(0.3);
		double damps = 0.0;

		switch (m.ssa)
		{
		case StepSizeAdaptation::TPA:
			return std::make_shared<TPA>(tc, sq, ss, cs, damps, sigma);
		case StepSizeAdaptation::MSR:
			return std::make_shared<MSR>(tc, sq, ss, cs, damps, sigma);
		case StepSizeAdaptation::XNES:
			cs = cs0.value_or(mueff / (2.0 * std::log(std::max(2., d)) * sqrt(d)));
			return std::make_shared<XNES>(tc, sq, ss, cs, damps, sigma);
		case StepSizeAdaptation::MXNES:
			cs = cs0.value_or(1.);
			return std::make_shared<MXNES>(tc, sq, ss, cs, damps, sigma);
		case StepSizeAdaptation::LPXNES:
			cs = cs0.value_or(9.0 * mueff / (10.0 * sqrt(d)));
			return std::make_shared<LPXNES>(tc, sq, ss, cs, damps, sigma);
		case StepSizeAdaptation::PSR:
			cs = cs0.value_or(.9);
			return std::make_shared<PSR>(tc, sq, ss, cs, 0., sigma);
		default:
		case StepSizeAdaptation::CSA:
			cs = cs0.value_or((mueff + 2.0) / (d + mueff + 5.0));
			damps = 1.0 + (2.0 * std::max(0.0, sqrt((mueff - 1.0) / (d + 1)) - 1) + cs);
			return std::make_shared<CSA>(tc, sq, ss, cs, damps, sigma);
		}
	}
}
