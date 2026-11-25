#pragma once

#include "common.hpp"
#include "modules.hpp"
#include "population.hpp"
#include "sampling.hpp"

namespace parameters
{
struct Stats;
struct Parameters;
struct Weights;
struct Modules;
} // namespace parameters
namespace matrix_adaptation
{
struct Adaptation;
}

namespace mutation
{

struct ThresholdConvergence
{
    //! The threshold at the start of a run as a multiple of E||z|| (default 1)
    Float init_threshold = 1.0;
    Float decay_factor = 0.995;
    virtual ~ThresholdConvergence() = default;

    /**
     * The threshold after `evaluations` of `budget`, a length of z: init_threshold *
     * expected_length * ((budget - evaluations) / budget)^decay_factor, with expected_length =
     * E||z||; 0 once the budget is used up. The optimizer computes it once per generation, from the
     * evaluations at its start (T^(g) of the framework paper; Piad-Morffis et al. 2015).
     */
    [[nodiscard]] Float threshold(const Float expected_length,
                                  const size_t budget,
                                  const size_t evaluations) const;

    /**
     * Threshold convergence for one sample: when ||z|| < t, z and y = A z are scaled so that
     * ||z|| = 2 t - ||z|| (reflected across the threshold radius).
     */
    virtual void apply(Eigen::Ref<Vector> z, Eigen::Ref<Vector> y, const Float t) const;
};

struct NoThresholdConvergence : ThresholdConvergence
{
    void apply(Eigen::Ref<Vector>, Eigen::Ref<Vector>, const Float) const override {}
};

class SequentialSelection
{
    Float seq_cutoff_factor;

  public:
    SequentialSelection(const parameters::Mirror &m, const Float seq_cutoff_factor = 1.0) :
        seq_cutoff_factor(m == parameters::Mirror::PAIRWISE ? std::max(Float{2.}, seq_cutoff_factor)
                                                            : seq_cutoff_factor)
    {
    }
    virtual bool break_conditions(const size_t i,
                                  const Float f,
                                  Float fopt,
                                  const parameters::Mirror &m,
                                  const size_t mu);

    //! The first individual from i on after which break_conditions can be true (for any f)
    [[nodiscard]] virtual size_t
    first_stop(const size_t i, const parameters::Mirror &m, const size_t mu) const;
};

struct NoSequentialSelection : SequentialSelection
{
    using SequentialSelection::SequentialSelection;

    bool break_conditions(const size_t i,
                          const Float f,
                          Float fopt,
                          const parameters::Mirror &m,
                          const size_t mu) override
    {
        return false;
    }

    size_t first_stop(const size_t, const parameters::Mirror &, const size_t) const override
    {
        return std::numeric_limits<size_t>::max();
    }
};

struct SigmaSampler
{
    sampling::GaussianTransformer sampler;

    SigmaSampler(const Float d,
                 const std::shared_ptr<rng::Generator> &generator = rng::global()) :
        sampler{std::make_shared<sampling::Uniform>(static_cast<size_t>(d), generator)}
    {
    }

    void apply_integer_bounds(parameters::Parameters &p);

    //! The lower bound on the step size of the integer variables, for the columns [begin,
    //! begin + n) of S (Marty et al. 2024); also used for the elitist parents
    static void apply_integer_bounds(const parameters::Parameters &p,
                                     Eigen::Ref<Matrix> S,
                                     Eigen::Index begin,
                                     Eigen::Index n);

    virtual void sample(const Float sigma, parameters::Parameters &p);
};

struct NoSigmaSampler : SigmaSampler
{
    using SigmaSampler::SigmaSampler;

    void sample(const Float sigma, parameters::Parameters &p) override;
};

struct Strategy
{
    std::shared_ptr<ThresholdConvergence> tc;
    std::shared_ptr<SequentialSelection> sq;
    std::shared_ptr<SigmaSampler> ss;
    Float sigma;
    Float s = 0;
    //! The threshold of threshold convergence in the current generation (begin_population)
    Float generation_threshold = 0;

    Strategy(const std::shared_ptr<ThresholdConvergence> &threshold_covergence,
             const std::shared_ptr<SequentialSelection> &sequential_selection,
             const std::shared_ptr<SigmaSampler> &sigma_sampler,
             const Float sigma0) :
        tc(threshold_covergence),
        sq(sequential_selection),
        ss(sigma_sampler),
        sigma(sigma0)
    {
    }

    virtual ~Strategy() = default;

    /*
     * The population, in batches (ModularCMAES::advance queues them and keeps track of the next
     * individual). begin_population starts it; sample_batch_from samples the individuals from i up
     * to the first after which sequential selection can stop; tell_batch passes their values to
     * evaluated(), and ends the population when it is complete (pop.n is then the number of
     * evaluated individuals). A sample does not depend on the objective values of the ones before
     * it, so a batch is sampled before it is evaluated.
     */

    //! Start a population of n_offspring (pop.n): sigmas, bound and repelling state
    void begin_population(const size_t n_offspring, parameters::Parameters &p);

    //! Sample the batch of individuals i..first_stop(i); returns its size
    Eigen::Index sample_batch_from(const Eigen::Index i, parameters::Parameters &p);

    /**
     * The values f of the batch that starts at individual i, evaluated at the points X (given: by
     * tell(f, X), where a point that differs from the asked one replaces it); ends the population
     * when it is complete
     */
    void tell_batch(const Eigen::Index i,
                    Eigen::Ref<const Vector> f,
                    Eigen::Ref<const Matrix> X,
                    const bool given,
                    parameters::Parameters &p);

    /**
     * Sample individuals begin..begin+k-1. Their y are one matrix product, Y = A Z; an individual
     * that bound correction (RESAMPLE) or repelling rejects is sampled again on its own.
     */
    void sample_batch(const Eigen::Index begin, const Eigen::Index k, parameters::Parameters &p);

    //! The last individual of a batch sampled from i on: sequential selection can stop the
    //! population after it, but not before it
    [[nodiscard]] size_t first_stop(const size_t i, const parameters::Parameters &p) const;

    //! The objective value f of individual i; returns whether the population is complete (all
    //! individuals evaluated, or sequential selection stopped it: pop.n is then i + 1)
    bool evaluated(const Eigen::Index i, const Float f, parameters::Parameters &p);

    //! After the last individual
    void end_population(parameters::Parameters &p);

    //! The number of points the step-size adaptation evaluates with every population, fixed per
    //! configuration (the budget reserves them: Parameters::reserved_points)
    [[nodiscard]] virtual size_t n_extra_points() const { return 0; }

    //! Those points, as columns of out (in objective space)
    virtual void extra_points(Eigen::Ref<Matrix> out, const parameters::Parameters &p) const {}

    //! Their objective values f, at the points X
    virtual void tell_extra_points(Eigen::Ref<const Vector> f,
                                   Eigen::Ref<const Matrix> X,
                                   parameters::Parameters &p)
    {
    }

    virtual void adapt(const parameters::Weights &w,
                       std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
                       Population &pop,
                       const Population &old_pop,
                       const parameters::Stats &stats,
                       const size_t lambda) {};

  protected:
    //! Called by end_population
    virtual void on_population_complete(parameters::Parameters &p) {}

  private:
    //! x of individual i from its y and sigma, with bound correction
    void finish_sample(const Eigen::Index i, parameters::Parameters &p);

    //! Whether individual i must be sampled again (attempt n_rejections so far)
    bool rejected(const Eigen::Index i, size_t &n_rejections, parameters::Parameters &p);
};

struct CSA : Strategy
{
    using Strategy::Strategy;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;
};

struct TPA : Strategy
{
    using Strategy::Strategy;

    Float a_tpa = 0.5;
    Float b_tpa = 0.0;
    Float rank_tpa = 0.0;

    //! The two points m +- sigma * dm
    size_t n_extra_points() const override { return 2; }
    void extra_points(Eigen::Ref<Matrix> out, const parameters::Parameters &p) const override;
    void tell_extra_points(Eigen::Ref<const Vector> f,
                           Eigen::Ref<const Matrix> X,
                           parameters::Parameters &p) override;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;
};

struct MSR : Strategy
{
    //! The offspring are compared with the value at quantile * lambda (0-based) of the sorted
    //! values of the previous generation (Ait ElHara et al. 2013, pycma: 0.3)
    Float quantile = 0.3;

    using Strategy::Strategy;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;
};

struct PSR : Strategy
{
    Float success_ratio = .25;

    Vector combined;

    std::vector<size_t> idx;
    std::vector<size_t> oidx;

    using Strategy::Strategy;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;
};

struct XNES : Strategy
{
    using Strategy::Strategy;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;
};

struct MXNES : Strategy
{
    using Strategy::Strategy;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;
};

struct LPXNES : Strategy
{
    using Strategy::Strategy;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;
};

struct SR : Strategy
{
    constexpr static Float tgt_success_ratio = 2.0 / 11.0;

    using Strategy::Strategy;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;
};

struct SA : Strategy
{
    using Strategy::Strategy;

    void adapt(const parameters::Weights &w,
               std::shared_ptr<matrix_adaptation::Adaptation> adaptation,
               Population &pop,
               const Population &old_pop,
               const parameters::Stats &stats,
               const size_t lambda) override;

  protected:
    void on_population_complete(parameters::Parameters &p) override;

  private:
    Float mean_sigma;
};

//! The step-size adaptation of the modules; the sigma sampler draws from generator
std::shared_ptr<Strategy> get(const parameters::Modules &m,
                              const Float d,
                              const Float sigma,
                              const std::shared_ptr<rng::Generator> &generator = rng::global());

} // namespace mutation