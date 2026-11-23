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

struct MaxSigma : Criterion
{
    static inline Float tolerance = 1e4;
    MaxSigma() :
        Criterion("MaxSigma")
    {
    }
    void update(const parameters::Parameters &p) override;
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

struct NoEffectAxis : Criterion
{
    static inline Float tolerance = 10.0 * std::numeric_limits<Float>::epsilon();
    NoEffectAxis() :
        Criterion("NoEffectAxis")
    {
    }
    void update(const parameters::Parameters &p) override;
};

struct NoEffectCoord : Criterion
{
    static inline Float tolerance = 10.0 * std::numeric_limits<Float>::epsilon();
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