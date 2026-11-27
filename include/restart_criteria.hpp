#pragma once

#include <cmath>
#include <set>

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
    //! Whether reset() ran (the built-in criteria size their buffers in on_reset)
    bool was_reset = false;

    Criterion(const std::string &name) :
        met(false),
        name(name),
        last_restart(0)
    {
    }

    virtual ~Criterion() = default;

    void reset(const parameters::Parameters &p);

    virtual void update(const parameters::Parameters &p) = 0;

    virtual void on_reset(const parameters::Parameters &p) {};

  protected:
    //! For a built-in criterion that sizes its buffers in on_reset: reset it when it was added
    //! after the last reset (from Python, #119), before its first update
    void reset_if_new(const parameters::Parameters &p)
    {
        if (!was_reset)
            reset(p);
    }
};

using vCriteria = std::vector<std::shared_ptr<restart::Criterion>>;

struct Criteria
{
    Criteria(const vCriteria &c) :
        items(c)
    {
    }

    void update(const parameters::Parameters &p)
    {
        for (auto &c : items)
        {
            c->update(p);
        }
    }

    void reset(const parameters::Parameters &p)
    {
        for (auto &c : items)
            c->reset(p);
    }

    bool any() const
    {
        for (const auto &c : items)
            if (c->met)
                return true;
        return false;
    }

    std::string reason() const
    {
        std::string res;
        for (const auto &c : items)
            if (c->met)
                res += c->name + "; ";
        return res;
    }

    vCriteria items;

    static Criteria get(const parameters::Modules modules, const int lambda);
};

struct ExceededMaxIter : Criterion
{
    size_t max_iter;
    ExceededMaxIter() :
        Criterion("ExceededMaxIter")
    {
    }
    void update(const parameters::Parameters &p) override;
    void on_reset(const parameters::Parameters &p) override;
};

struct NoImprovement : Criterion
{
    size_t n_bin;
    std::vector<Float> best_fitnesses;
    static inline Float tolerance = 1e-20;
    NoImprovement() :
        Criterion("NoImprovement")
    {
    }
    void update(const parameters::Parameters &p) override;
    void on_reset(const parameters::Parameters &p) override;
};

/**
 * TolXUp (Hansen 2016, App. B.3): met when sigma max(D) has grown by more than tolerance since the
 * start of the run, with D the axis lengths of the distribution (the square roots of the
 * eigenvalues of C). A far too small sigma0, or divergence. pycma's tolfacupx is the same test per
 * coordinate (sigma sqrt(C_ii) > 1e3 sigma0).
 */
struct MaxSigma : Criterion
{
    static inline Float tolerance = 1e4;
    //! sigma max(D) at the start of the run
    Float start = 1.0;
    MaxSigma() :
        Criterion("MaxSigma")
    {
    }
    void update(const parameters::Parameters &p) override;
    void on_reset(const parameters::Parameters &p) override;
};

struct MinSigma : Criterion
{
    static inline Float tolerance = 1e-20;
    MinSigma() :
        Criterion("MinSigma")
    {
    }
    void update(const parameters::Parameters &p) override;
};

struct UnableToAdapt : Criterion
{
    UnableToAdapt() :
        Criterion("UnableToAdapt")
    {
    }
    void update(const parameters::Parameters &p) override;
};

struct FlatFitness : Criterion
{
    size_t max_flat_fitness;
    size_t flat_fitness_index;
    Eigen::Array<int, Eigen::Dynamic, 1> flat_fitnesses;

    FlatFitness() :
        Criterion("FlatFitness")
    {
    }
    void update(const parameters::Parameters &p) override;
    void on_reset(const parameters::Parameters &p) override;
};

struct TolX : Criterion
{
    static inline Float tolerance = 1e-11;
    Vector tolx_vector;
    //! The step size at the start of the current run (sigma0, or the sigma of the restart)
    Float sigma_start = 1.0;
    TolX() :
        Criterion("TolX")
    {
    }
    void update(const parameters::Parameters &p) override;
    void on_reset(const parameters::Parameters &p) override;
};

struct MaxDSigma : Criterion
{
    static inline Float tolerance = std::pow(10., 20.);
    MaxDSigma() :
        Criterion("MaxDSigma")
    {
    }
    void update(const parameters::Parameters &p) override;
};

struct MinDSigma : Criterion
{
    static inline Float tolerance = std::pow(10., -20.);
    MinDSigma() :
        Criterion("MinDSigma")
    {
    }
    void update(const parameters::Parameters &p) override;
};

struct ConditionC : Criterion
{
    static inline Float tolerance = std::pow(10., 14.);
    ConditionC() :
        Criterion("ConditionC")
    {
    }
    void update(const parameters::Parameters &p) override;
};

//! pycma's noeffectaxis: met when m + 0.1 sigma D_i b_i == m in floating point, for the axis
//! i = t mod d of generation t (Hansen 2016, App. B.3)
struct NoEffectAxis : Criterion
{
    //! The step along the axis, in standard deviations
    static inline Float tolerance = 0.1;
    NoEffectAxis() :
        Criterion("NoEffectAxis")
    {
    }
    void update(const parameters::Parameters &p) override;
};

//! pycma's noeffectcoord: met when m_i + 0.2 sigma sqrt(C_ii) == m_i in floating point, for some
//! coordinate i (Hansen 2016, App. B.3)
struct NoEffectCoord : Criterion
{
    //! The step along the coordinate, in standard deviations
    static inline Float tolerance = 0.2;
    NoEffectCoord() :
        Criterion("NoEffectCoord")
    {
    }
    void update(const parameters::Parameters &p) override;
};

/**
 * The medians of the first `split` values of a growing sequence and of the rest, kept up to date in
 * O(log n) per value (B10: Stagnation sorted the whole history four times per generation, O(t) per
 * generation). The medians are exactly those of a sort: of an even count, the mean of the two
 * middle values. NaN orders after every number.
 */
class SplitMedian
{
  public:
    //! The sequence so far
    [[nodiscard]] const std::vector<Float> &values() const { return values_; }

    void push_back(Float x);

    //! Replace the sequence (split back to 0)
    void assign(const std::vector<Float> &values);

    void clear() { assign({}); }

    //! Values [0, split) form the head, [split, n) the tail
    void set_split(size_t split);

    [[nodiscard]] Float head_median() const { return head_.median(); }
    [[nodiscard]] Float tail_median() const { return tail_.median(); }

  private:
    struct NanLast
    {
        bool operator()(const Float a, const Float b) const
        {
            return std::isnan(b) ? !std::isnan(a) : a < b;
        }
    };

    //! A multiset with its median: lower half lo, upper half hi, |lo| - |hi| in {0, 1}
    struct Median
    {
        std::multiset<Float, NanLast> lo, hi;
        void insert(Float x);
        void erase(Float x);
        [[nodiscard]] Float median() const;
        void rebalance();
    };

    std::vector<Float> values_;
    Median head_, tail_;
    size_t split_ = 0;
};

struct Stagnation : Criterion
{
    static inline Float tolerance = 0.3;

    size_t n_stagnation;
    //! The median fitness and the best fitness of every generation since the last reset
    SplitMedian median_fitnesses;
    SplitMedian best_fitnesses;
    Stagnation() :
        Criterion("Stagnation")
    {
    }
    void update(const parameters::Parameters &p) override;
    void on_reset(const parameters::Parameters &p) override;
};

struct TooMuchRepelling : Criterion
{
    static inline Float tolerance = 50;
    TooMuchRepelling() :
        Criterion("TooMuchRepelling")
    {
    }
    void update(const parameters::Parameters &p) override;
};

} // namespace restart