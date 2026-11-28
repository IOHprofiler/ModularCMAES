#pragma once

#include "common.hpp"
#include "population.hpp"

namespace parameters
{
struct Parameters;
struct Modules;
} // namespace parameters

namespace selection
{
struct Elitsm
{
    virtual ~Elitsm() = default;
    virtual void operator()(parameters::Parameters &p) const;
};

struct NoElitsm : Elitsm
{
    void operator()(parameters::Parameters &p) const override {}
};

/**
 * Pairwise selection (Auger, Brockhoff and Hansen 2011, Alg. 1): of each mirrored pair (offspring
 * 2i and 2i + 1) only the better one is ranked. sort() orders the population: the winners of the
 * pairs, the unpaired individuals and the parents of elitism by f, then the losers by f, and sets
 * pop.n_ranked. The losers keep their real f (#75, F9: an f of +inf made MSR, Stagnation and the
 * mean f infinite); they get no negative weights in the active update (Weights.pairwise).
 */
struct Pairwise
{
    virtual ~Pairwise() = default;

    //! Sort pop; its first n_offspring individuals are the offspring, in their pairs
    virtual void sort(Population &pop, size_t n_offspring) const;

    /**
     * The order in which sort() puts the values f, whose first n_offspring are the offspring, in
     * their pairs: the indices into f, written to out. Returns the number of ranked individuals
     * (pop.n_ranked after sort()). XNES ranks the offspring alone with it, before elitism adds the
     * parents (#172).
     */
    virtual size_t
    rank(Eigen::Ref<const Vector> f, size_t n_offspring, std::vector<size_t> &out) const;

    //! sort(p.pop, p.pop.n)
    void operator()(parameters::Parameters &p) const;

  private:
    mutable std::vector<size_t> order;
    mutable std::vector<char> loser;
};

//! No pairwise selection: sort() sorts the population by f
struct NoPairwise : Pairwise
{
    void sort(Population &pop, size_t n_offspring) const override;

    //! The order of f (ties in index order, NaN last, as Population::sort); all are ranked
    size_t
    rank(Eigen::Ref<const Vector> f, size_t n_offspring, std::vector<size_t> &out) const override;
};

struct Strategy
{
    std::shared_ptr<Pairwise> pairwise;
    std::shared_ptr<Elitsm> elitsm;

    Strategy(const parameters::Modules &);

    void select(parameters::Parameters &p) const;
};

} // namespace selection
