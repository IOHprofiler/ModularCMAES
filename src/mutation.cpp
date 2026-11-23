#include "mutation.hpp"

#include "bounds.hpp"
#include "matrix_adaptation.hpp"
#include "parameters.hpp"

namespace mutation
{
void ThresholdConvergence::scale_inplace(Eigen::Ref<Vector> zi,
                                         const Float diameter,
                                         const size_t budget,
                                         const size_t evaluations)
{
    const Float t =
        init_threshold * diameter *
        pow(static_cast<Float>(budget - evaluations) / static_cast<Float>(budget), decay_factor);

    if (const auto norm = zi.norm(); norm < t)
        zi.array() *= ((t + (t - norm)) / norm);
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
    // .array() is a view; binding the Vector to const Array & made a temporary copy
    const auto Cdiag = p.adaptation->coordinate_wise_variances.array();
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
    begin_population(n_offspring, p);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_offspring); ++i)
    {
        sample(i, p.stats.evaluations, p);
        if (evaluated(i, objective(p.pop.X_transformed.col(i)), p))
            break;
    }
    end_population(p);

    if (const auto k = static_cast<Eigen::Index>(n_extra_points()); k != 0)
    {
        Matrix X(p.settings.dim, k);
        extra_points(X, p);
        Vector f(k);
        for (Eigen::Index j = 0; j < k; ++j)
            f(j) = objective(X.col(j));
        tell_extra_points(f, X, p);
    }
}

void Strategy::begin_population(const size_t n_offspring, parameters::Parameters &p)
{
    p.pop.reserve(n_offspring);
    p.pop.n = n_offspring;
    ss->sample(sigma, p);
    p.bounds->n_out_of_bounds = 0;
    p.repelling->prepare_sampling(p);
}

void Strategy::sample(const Eigen::Index i, const size_t evaluations, parameters::Parameters &p)
{
    size_t n_rej = 0;
    do
    {
        // Every step writes into the columns of the population: no allocation per sample
        p.pop.t(i) = p.stats.t;
        p.sampler->sample(p.pop.Z.col(i));
        p.mutation->tc->scale_inplace(
            p.pop.Z.col(i), p.settings.diameter, p.settings.budget, evaluations);
        p.adaptation->compute_y_into(p.pop.Z.col(i), p.pop.Y.col(i));
        p.pop.X.col(i).array() =
            p.pop.Y.col(i).array() * p.pop.S.col(i).array() + p.adaptation->m.array();
        p.bounds->correct(i, p);

    } while ((p.settings.modules.bound_correction == parameters::CorrectionMethod::RESAMPLE &&
              n_rej++ < 5 * p.settings.dim &&
              bounds::any_out_of_bounds(p.pop.X.col(i), p.settings.lb, p.settings.ub)) ||
             p.repelling->is_rejected(p.pop.X.col(i), p));

    p.coordinate_mapping->transform_into(p.pop.X.col(i), p.pop.X_transformed.col(i));
}

size_t Strategy::first_stop(const size_t i, const parameters::Parameters &p) const
{
    return sq->first_stop(i, p.settings.modules.mirrored, p.mu);
}

bool Strategy::evaluated(const Eigen::Index i, const Float f, parameters::Parameters &p)
{
    p.pop.f(i) = f;
    p.stats.evaluations++;
    if (sq->break_conditions(
            i, p.pop.f(i), p.stats.global_best.y, p.settings.modules.mirrored, p.mu))
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
                const parameters::Stats &stats,
                const size_t lambda)

{
    Float l = (w.cs / w.damps) * ((adaptation->ps.norm() / w.expected_length_z) - 1);
    // Clamping as seen in pycma
    l = std::min(Float{1.0}, std::max(l, Float{-1.0}));
    sigma *= std::exp(l);
}

void TPA::extra_points(Eigen::Ref<Matrix> out, const parameters::Parameters &p) const
{
    out.col(0) =
        p.coordinate_mapping->transform(p.adaptation->m + (p.mutation->sigma * p.adaptation->dm));
    out.col(1) =
        p.coordinate_mapping->transform(p.adaptation->m + (p.mutation->sigma * -p.adaptation->dm));
}

void TPA::tell_extra_points(Eigen::Ref<const Vector> f,
                            Eigen::Ref<const Matrix> X,
                            parameters::Parameters &p)
{
    const Float f_pos = f(0), f_neg = f(1);
    p.stats.update_best(X.col(0), f_pos);
    p.stats.update_best(X.col(1), f_neg);
    p.stats.evaluations += 2;
    this->rank_tpa = f_neg < f_pos ? -a_tpa : a_tpa + b_tpa;
}

void TPA::adapt(const parameters::Weights &w,
                std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                Population &pop,
                const Population &old_pop,
                const parameters::Stats &stats,
                const size_t lambda)
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

void MSR::adapt(const parameters::Weights &w,
                std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                Population &pop,
                const Population &old_pop,
                const parameters::Stats &stats,
                const size_t lamb)
{
    const auto n = std::min(pop.n_finite(), old_pop.n_finite());
    if (n != 0)
    {
        // The individuals of this generation (fewer than lambda after a sequential selection stop)
        const Float lambda = static_cast<Float>(pop.n);
        const Float k = (pop.f.head(pop.n).array() < median(old_pop.f.head(old_pop.n)))
                            .cast<Float>()
                            .sum();
        const auto z = (2.0 / lambda) * (k - ((lambda + 1.0) / 2.0));
        s = ((1.0 - w.cs) * s) + (w.cs * z);
        sigma *= std::exp(s / (2.0 - (2.0 / adaptation->dd)));
    }
}

void PSR::adapt(const parameters::Weights &w,
                std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                Population &pop,
                const Population &old_pop,
                const parameters::Stats &stats,
                const size_t lambda)
{
    const auto n = std::min(pop.n_finite(), old_pop.n_finite());
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
        sigma *= std::exp(s / (2.0 - (2.0 / adaptation->dd)));
    }
}

void XNES::adapt(const parameters::Weights &w,
                 std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                 Population &pop,
                 const Population &old_pop,
                 const parameters::Stats &stats,
                 const size_t lambda)
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
                  const parameters::Stats &stats,
                  const size_t lambda)
{
    const Float delta = (w.mueff * adaptation->dz.squaredNorm() - adaptation->dd);

    sigma *= std::exp((w.cs / adaptation->dd) * delta);
}

void LPXNES::adapt(const parameters::Weights &w,
                   std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                   Population &pop,
                   const Population &old_pop,
                   const parameters::Stats &stats,
                   const size_t lambda)
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
               const parameters::Stats &stats,
               const size_t lambda)
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
               const parameters::Stats &stats,
               const size_t lambda)
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

std::shared_ptr<Strategy> get(const parameters::Modules &m, const Float d, const Float sigma)
{
    using namespace parameters;

    auto tc = m.threshold_convergence ? std::make_shared<ThresholdConvergence>()
                                      : std::make_shared<NoThresholdConvergence>();

    auto sq = m.sequential_selection ? std::make_shared<SequentialSelection>(m.mirrored)
                                     : std::make_shared<NoSequentialSelection>(m.mirrored);

    auto ss =
        (m.sample_sigma or m.ssa == StepSizeAdaptation::LPXNES or m.ssa == StepSizeAdaptation::SA)
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
} // namespace mutation
