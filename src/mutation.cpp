#include "mutation.hpp"

#include "bounds.hpp"
#include "matrix_adaptation.hpp"
#include "parameters.hpp"

namespace mutation
{
Float ThresholdConvergence::threshold(const Float expected_length,
                                      const size_t budget,
                                      const size_t evaluations) const
{
    // evaluations can exceed the budget in ask-tell: the remaining share is then 0 (size_t
    // would wrap around)
    const Float remaining =
        evaluations < budget
            ? static_cast<Float>(budget - evaluations) / static_cast<Float>(budget)
            : Float(0.0);
    return init_threshold * expected_length * std::pow(remaining, decay_factor);
}

void ThresholdConvergence::apply(Eigen::Ref<Vector> z, Eigen::Ref<Vector> y, const Float t) const
{
    // A length of z: relative to the search distribution (sigma and C), so it does not depend on
    // the box. Piad-Morffis et al. (2015) and the framework paper put the threshold on x - m, as a
    // fraction of the box diameter; with those defaults a run cannot converge before the budget
    // is nearly used up (BBOB: 8 of 350 runs reached 1e-8, 255 with the threshold on z; #78)
    const Float length = z.norm();
    if (length > 0 && length < t)
    {
        const Float factor = (t + (t - length)) / length;
        z *= factor;
        y *= factor;
    }
}

bool SequentialSelection::break_conditions(const size_t i,
                                           const Float f,
                                           Float fopt,
                                           const parameters::Mirror &m,
                                           const size_t mu)
{
    const size_t seq_cutoff = static_cast<size_t>(mu * seq_cutoff_factor);
    // With pairwise mirroring only stop after a complete pair (i odd: individuals 0..i are pairs)
    return (f < fopt) and (i >= seq_cutoff) and (m != parameters::Mirror::PAIRWISE or i % 2 == 1);
}

size_t SequentialSelection::first_stop(const size_t i,
                                       const parameters::Mirror &m,
                                       const size_t mu) const
{
    const size_t seq_cutoff = static_cast<size_t>(mu * seq_cutoff_factor);
    const size_t first = std::max(i, seq_cutoff);
    return (m == parameters::Mirror::PAIRWISE && first % 2 == 0) ? first + 1 : first;
}

void SigmaSampler::apply_integer_bounds(parameters::Parameters &p)
{
    apply_integer_bounds(p, p.pop.S, 0, static_cast<Eigen::Index>(p.pop.n));
}

void SigmaSampler::apply_integer_bounds(const parameters::Parameters &p,
                                        Eigen::Ref<Matrix> S,
                                        const Eigen::Index begin,
                                        const Eigen::Index n)
{
    // .array() is a view; binding the Vector to const Array & made a temporary copy
    const auto Cdiag = p.adaptation->coordinate_wise_variances.array();
    for (const auto &iidx : p.settings.integer_variables)
    {
        const Float Cii = std::max(Cdiag[iidx], Float(1e-16));
        const Float lb_sigma = p.weights.int_lb_sigma / std::sqrt(Cii);

        for (Eigen::Index i = begin; i < begin + n; i++)
            S(iidx, i) = std::max(S(iidx, i), lb_sigma);
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

void Strategy::begin_population(const size_t n_offspring, parameters::Parameters &p)
{
    p.pop.reserve(n_offspring);
    p.pop.n = n_offspring;
    p.pop.n_ranked = std::numeric_limits<size_t>::max();
    // A new generation: a new block of orthogonal samples, no pair sampled ahead (#75, #76)
    p.sampler->begin_generation();
    partner_sampled = false;
    ss->sample(sigma, p);
    generation_threshold =
        tc->threshold(p.sampler->expected_length(), p.settings.budget, p.stats.evaluations);
    p.bounds->n_out_of_bounds = 0;
    p.repelling->prepare_sampling(p);
}

Eigen::Index Strategy::sample_batch_from(const Eigen::Index i, parameters::Parameters &p)
{
    const auto n = static_cast<Eigen::Index>(p.pop.n);
    const auto stop = first_stop(static_cast<size_t>(i), p);
    const Eigen::Index last =
        stop >= static_cast<size_t>(n) ? n - 1 : static_cast<Eigen::Index>(stop);
    const Eigen::Index k = last - i + 1;
    sample_batch(i, k, p);
    return k;
}

void Strategy::tell_batch(const Eigen::Index i,
                          Eigen::Ref<const Vector> f,
                          Eigen::Ref<const Matrix> X,
                          const bool given,
                          parameters::Parameters &p)
{
    for (Eigen::Index j = 0; j < f.size(); ++j)
    {
        // tell(f, X): a point that differs from the asked one replaces it
        if (given && X.col(j) != p.pop.X_transformed.col(i + j))
            p.inject(i + j, X.col(j));
        if (evaluated(i + j, f(j), p))
        {
            end_population(p);
            return;
        }
    }
}

void Strategy::sample_batch(const Eigen::Index begin,
                            const Eigen::Index k,
                            parameters::Parameters &p)
{
    // Mirrored sampling (Brockhoff et al. 2010): individuals 2i and 2i + 1 of a generation are a
    // pair, z and -z; an odd last individual is unpaired. A pair is sampled, and rejected (bound
    // correction RESAMPLE, repelling), as one (#75, F10). The partner of the last individual of
    // a batch (sequential selection) is sampled with it, and queued with the next batch.
    const bool mirrored = p.settings.modules.mirrored != parameters::Mirror::NONE;
    const auto n = static_cast<Eigen::Index>(p.pop.n);
    const Eigen::Index first = partner_sampled ? begin + 1 : begin;
    Eigen::Index end = begin + k;
    partner_sampled = mirrored && end % 2 == 1 && end < n;
    if (partner_sampled)
        ++end;
    if (first >= end)
        return;

    // Every step writes into the columns of the population: no allocation per sample
    for (Eigen::Index i = first; i < end; ++i)
    {
        p.pop.t(i) = p.stats.t;
        if (mirrored && i % 2 == 1)
            p.pop.Z.col(i) = -p.pop.Z.col(i - 1);
        else
            p.sampler->sample(p.pop.Z.col(i));
    }
    p.adaptation->compute_y_batch(p.pop.Z.middleCols(first, end - first),
                                  p.pop.Y.middleCols(first, end - first));

    for (Eigen::Index i = first; i < end;)
    {
        const Eigen::Index size = mirrored && i + 1 < n ? 2 : 1;
        for (Eigen::Index j = i; j < i + size; ++j)
        {
            tc->apply(p.pop.Z.col(j), p.pop.Y.col(j), generation_threshold);
            finish_sample(j, p);
        }
        size_t n_rejections = 0;
        while (rejected(i, size, n_rejections, p))
        {
            p.sampler->sample(p.pop.Z.col(i));
            if (size == 2)
                p.pop.Z.col(i + 1) = -p.pop.Z.col(i);
            for (Eigen::Index j = i; j < i + size; ++j)
            {
                p.adaptation->compute_y_into(p.pop.Z.col(j), p.pop.Y.col(j));
                tc->apply(p.pop.Z.col(j), p.pop.Y.col(j), generation_threshold);
                finish_sample(j, p);
            }
        }
        for (Eigen::Index j = i; j < i + size; ++j)
            p.coordinate_mapping->transform_into(
                p.pop.X.col(j), p.pop.X_transformed.col(j), p.settings.lb, p.settings.ub);
        i += size;
    }
}

void Strategy::finish_sample(const Eigen::Index i, parameters::Parameters &p)
{
    p.pop.X.col(i).array() =
        p.pop.Y.col(i).array() * p.pop.S.col(i).array() + p.adaptation->m.array();
    p.bounds->correct(i, p);
}

bool Strategy::rejected(const Eigen::Index i,
                        const Eigen::Index size,
                        size_t &n_rejections,
                        parameters::Parameters &p)
{
    const auto out_of_bounds = [&p, i, size]() {
        for (Eigen::Index j = i; j < i + size; ++j)
            if (bounds::any_out_of_bounds(p.pop.X.col(j), p.settings.lb, p.settings.ub))
                return true;
        return false;
    };
    if (p.settings.modules.bound_correction == parameters::CorrectionMethod::RESAMPLE &&
        n_rejections++ < 5 * p.settings.dim && out_of_bounds())
        return true;
    for (Eigen::Index j = i; j < i + size; ++j)
        if (p.repelling->is_rejected(p.pop.X.col(j), p))
            return true;
    return false;
}

size_t Strategy::first_stop(const size_t i, const parameters::Parameters &p) const
{
    return sq->first_stop(i, p.settings.modules.mirrored, p.mu);
}

bool Strategy::evaluated(const Eigen::Index i, const Float f, parameters::Parameters &p)
{
    p.pop.f(i) = f;
    // Sequential selection stops at the first point better than the best of the previous
    // generation (old_pop is sorted); in the first generation of a run there is none (-inf)
    const Float reference = p.old_pop.n_finite() != 0 ? p.old_pop.f(0)
                                                        : -std::numeric_limits<Float>::infinity();
    if (sq->break_conditions(i, p.pop.f(i), reference, p.settings.modules.mirrored, p.mu))
    {
        // Sequential selection: the population is individuals 0..i. The columns after i keep
        // their (stale) contents but are no longer part of it; the weights of the updates are
        // renormalised for n < lambda (Weights::for_population).
        p.pop.n = static_cast<size_t>(i) + 1;
        return true;
    }
    return static_cast<size_t>(i) + 1 >= p.pop.n;
}

void Strategy::end_population(parameters::Parameters &p)
{
    p.repelling->adapt_radii_after_sampling(p);
    on_population_complete(p);
}

void CSA::adapt(const parameters::Weights &w,
                std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                Population &pop,
                const Population &old_pop,
                const parameters::Stats &stats)

{
    Float l = (w.cs / w.damps) * ((adaptation->ps.norm() / w.expected_length_z) - 1);
    // Clamping as seen in pycma
    l = std::min(Float{1.0}, std::max(l, Float{-1.0}));
    sigma *= std::exp(l);
}

void TPA::extra_points(Eigen::Ref<Matrix> out, const parameters::Parameters &p) const
{
    const Vector &lb = p.settings.lb, &ub = p.settings.ub;
    out.col(0) = p.coordinate_mapping->transform(
        p.adaptation->m + (p.mutation->sigma * p.adaptation->dm), lb, ub);
    out.col(1) = p.coordinate_mapping->transform(
        p.adaptation->m + (p.mutation->sigma * -p.adaptation->dm), lb, ub);
}

void TPA::tell_extra_points(Eigen::Ref<const Vector> f,
                            Eigen::Ref<const Matrix> X,
                            parameters::Parameters &p)
{
    const Float f_pos = f(0), f_neg = f(1);
    p.stats.update_best(X.col(0), f_pos);
    p.stats.update_best(X.col(1), f_neg);
    // A tie (as after a restart, when dm = 0 and both points are the mean) is no signal: 0, as the
    // sign of the rank difference in pycma's TPA; it counted as a success (sigma grew by 16 %)
    this->rank_tpa = f_neg < f_pos ? -a_tpa : (f_pos < f_neg ? a_tpa + b_tpa : Float(0.0));
}

void TPA::adapt(const parameters::Weights &w,
                std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                Population &pop,
                const Population &old_pop,
                const parameters::Stats &stats)
{
    s = ((1.0 - w.cs) * s) + (w.cs * this->rank_tpa);
    sigma *= std::exp(s);
}

void MSR::adapt(const parameters::Weights &w,
                std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                Population &pop,
                const Population &old_pop,
                const parameters::Stats &stats)
{
    const auto n = std::min(pop.n_finite(), old_pop.n_finite());
    if (n != 0)
    {
        // Ait ElHara, Auger & Hansen (2013): K, the number of offspring better than the j-th best
        // of the previous generation, with j = 0.3 lambda. As pycma
        // (CMAAdaptSigmaMedianImprovement), j is a fractional 0-based index into the sorted values
        // of the previous generation, and K is interpolated between its two neighbours.
        // The individuals of this generation: fewer than lambda after a sequential selection stop.
        // With pairwise selection the ranked ones, the winners of the mirrored pairs: Auger et al.
        // (2011, Alg. 1) adapt sigma from the ranked offspring only (#75, F9)
        const size_t n_cur = pop.ranked(), n_old = old_pop.ranked();
        const Float lambda = static_cast<Float>(n_cur);
        const auto f = pop.f.head(static_cast<Eigen::Index>(n_cur)).array();
        const Float j = quantile * static_cast<Float>(n_old);
        const auto j0 = std::min<Eigen::Index>(static_cast<Eigen::Index>(j),
                                               static_cast<Eigen::Index>(n_old) - 1);
        const auto j1 = std::min<Eigen::Index>(static_cast<Eigen::Index>(std::ceil(j)),
                                               static_cast<Eigen::Index>(n_old) - 1);
        const Float frac = j - std::floor(j);
        const Float k = (1.0 - frac) * (f < old_pop.f(j0)).cast<Float>().sum() +
                        frac * (f <= old_pop.f(j1)).cast<Float>().sum();
        // z >= 0 iff the median offspring is better than the j-th best of the previous generation.
        // Centered at lambda / 2 as pycma; Loshchilov (2014) restates the paper with (lambda + 1) / 2,
        // which on BBOB (d = 20) solved Rosenbrock and the bent cigar in 15 of 25 runs, pycma's in
        // 22 and 25 (#83)
        const auto z = (2.0 / lambda) * (k - (lambda / 2.0));
        s = ((1.0 - w.cs) * s) + (w.cs * z);
        sigma *= std::exp(s / w.damps);
    }
}

void PSR::adapt(const parameters::Weights &w,
                std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                Population &pop,
                const Population &old_pop,
                const parameters::Stats &stats)
{
    // With pairwise selection only the ranked individuals, the winners of the mirrored pairs, as
    // MSR (#75, F9)
    const auto n = std::min({pop.n_finite(), old_pop.n_finite(), pop.ranked(), old_pop.ranked()});
    if (n != 0)
    {
        combined.conservativeResize(n + n);
        combined.head(n) = pop.f.head(n);
        combined.tail(n) = old_pop.f.head(n);

        if (idx.size() != 2 * n)
        {
            idx.resize(2 * n);
            oidx.resize(2 * n);
            std::iota(idx.begin(), idx.end(), 0);
        }
        utils::sort_index_inplace(combined, idx);
        for (size_t rank = 0; rank < 2 * n; ++rank)
            oidx[idx[rank]] = rank;

        Float delta_r = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            Float r = oidx[i];
            Float r_old = oidx[n + i];
            delta_r += (r_old - r);
        }

        const auto z = delta_r / std::pow(n, 2) - success_ratio;
        s = (1.0 - w.cs) * s + (w.cs * z);
        sigma *= std::exp(s / w.damps);
    }
}

void XNES::adapt(const parameters::Weights &w,
                 std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                 Population &pop,
                 const Population &old_pop,
                 const parameters::Stats &stats)
{
    if (const auto dynamic =
            std::dynamic_pointer_cast<matrix_adaptation::NaturalGradientAdaptation>(adaptation))
    {
        sigma *= std::exp(w.cs / 2.0 * dynamic->sigma_g);
        return;
    }

    const Float z =
        ((pop.Z.leftCols(pop.n)).colwise().squaredNorm().array() - adaptation->dd).matrix() *
        w.clipped().head(pop.n);
    sigma *= std::exp((w.cs / std::sqrt(adaptation->dd)) * z);
}

void MXNES::adapt(const parameters::Weights &w,
                  std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                  Population &pop,
                  const Population &old_pop,
                  const parameters::Stats &stats)
{
    const Float delta = (w.mueff_mean * adaptation->dz.squaredNorm() - adaptation->dd);

    sigma *= std::exp((w.cs / adaptation->dd) * delta);
}

void LPXNES::adapt(const parameters::Weights &w,
                   std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                   Population &pop,
                   const Population &old_pop,
                   const parameters::Stats &stats)
{
    const auto logS = (pop.S.leftCols(pop.n).array() / sigma).log();
    const Vector per_sample = logS.colwise().mean().transpose();
    const Float rel_log = per_sample.dot(w.clipped().head(pop.n));
    sigma *= std::exp(w.cs * rel_log);
}

void SR::adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats)
{
    sigma *= std::exp((1 / w.damps) *
                      ((stats.success_ratio - tgt_success_ratio) / (1.0 - tgt_success_ratio)));
}

void SA::on_population_complete(parameters::Parameters &p)
{
    mean_sigma = std::exp(p.pop.S.leftCols(p.pop.n).array().log().mean());
}

void SA::adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats)
{
    const Eigen::Index mu = w.positive.size();

    if (mu <= 0 || mu > static_cast<Eigen::Index>(pop.n))
    {
        throw std::invalid_argument("SA::adapt: mu must be in [1, pop.n]");
    }

    if (pop.S.rows() == 0)
    {
        throw std::invalid_argument("SA::adapt: population sigma matrix has no dimensions");
    }

    const auto selected_sigma = pop.S.leftCols(mu);

    // One geometric-mean log-sigma per selected offspring.
    const Vector log_sigma_per_offspring =
        selected_sigma.array().log().colwise().mean().transpose();

    const Float weighted_log_sigma = w.positive.dot(log_sigma_per_offspring);

    const Float selected_mean_sigma = std::exp(weighted_log_sigma);

    if (!std::isfinite(selected_mean_sigma) || !std::isfinite(mean_sigma) || mean_sigma <= 0.0)
    {
        throw std::runtime_error("SA::adapt produced an invalid sigma update");
    }

    // Preserve the original intent of a normalized relative update.
    sigma *= selected_mean_sigma / mean_sigma;
}

std::shared_ptr<Strategy> get(const parameters::Modules &m,
                              const Float d,
                              const Float sigma,
                              const std::shared_ptr<rng::Generator> &generator)
{
    using namespace parameters;

    auto tc = m.threshold_convergence ? std::make_shared<ThresholdConvergence>()
                                      : std::make_shared<NoThresholdConvergence>();

    auto sq = m.sequential_selection ? std::make_shared<SequentialSelection>(m.mirrored)
                                     : std::make_shared<NoSequentialSelection>(m.mirrored);

    auto ss =
        (m.sample_sigma or m.ssa == StepSizeAdaptation::LPXNES or m.ssa == StepSizeAdaptation::SA)
            ? std::make_shared<SigmaSampler>(d, generator)
            : std::make_shared<NoSigmaSampler>(d, generator);

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
} // namespace mutation
