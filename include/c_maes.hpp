#pragma once

#include "parameters.hpp"

/**
 * The modular CMA-ES.
 *
 * Two ways to run it:
 *  - with an objective function: run() / operator() until a break condition holds, step() for one
 *    generation, or mutate(), select(), recombine(), adapt() for the steps of a generation;
 *  - ask-tell, for an objective that is evaluated outside: ask() returns the points to evaluate,
 *    tell() takes their objective values. step() and run() are loops over ask() and tell(), so both
 *    ways give the same results.
 *
 * ask() returns the largest batch of points that the algorithm can evaluate in any order: a whole
 * population (with the two TPA points after it), or, with sequential selection, the individuals up
 * to the first one after which it can stop, then one (or a mirrored pair) at a time. A restart asks
 * for its center, then for the points of the hill-valley tests of the repelling archive, one at a
 * time.
 */
struct ModularCMAES
{
    std::shared_ptr<parameters::Parameters> p;

    ModularCMAES(const std::shared_ptr<parameters::Parameters> p) :
        p(p)
    {
    }
    ModularCMAES(const size_t dim) :
        ModularCMAES(std::make_shared<parameters::Parameters>(dim))
    {
    }
    ModularCMAES(const parameters::Settings &settings) :
        ModularCMAES(std::make_shared<parameters::Parameters>(settings))
    {
    }

    void recombine() const;

    void select() const;

    void adapt() const;

    //! Sample and evaluate a generation (after a restart, when one is due)
    void mutate(FunctionType &objective);

    //! One generation; returns false when a break condition holds
    bool step(FunctionType &objective);

    //! Run until a break condition holds
    void operator()(FunctionType &objective);

    bool break_conditions() const;

    /**
     * The favorite solution: the mean of the search distribution as a point of the objective
     * (integer variables rounded; inside the bounds when a bound correction is used), as pycma's
     * xfavorite. With a noisy objective it estimates the optimum better than the best evaluated
     * point, whose value was selected for being low. Not evaluated.
     */
    Vector xfavorite() const;

    /**
     * The points to evaluate next, the columns of a d x k matrix (k >= 1, in the space of the
     * objective). Starts a generation when none is in progress. Until tell(), it returns the same
     * points. The view is valid until the next tell().
     */
    Eigen::Ref<const Matrix> ask();

    /**
     * The objective values of the points of ask(), in the same order. Returns true when this
     * completed a generation (selection, recombination and adaptation are done; break_conditions()
     * tells whether to stop), false when ask() has more points of this generation.
     */
    bool tell(Eigen::Ref<const Vector> f);

    /**
     * The same, for the points X (d x k) instead of the asked ones: a column that differs from the
     * asked point replaces it (a repaired or injected solution). For an individual of the
     * population, its y and z are derived from X with its sigma, as for elitism.
     */
    bool tell(Eigen::Ref<const Vector> f, Eigen::Ref<const Matrix> X);

    //! Whether ask() has points that wait for tell()
    [[nodiscard]] bool has_pending() const { return phase != Phase::IDLE; }

  private:
    enum class Phase
    {
        IDLE,              // no generation in progress
        RESTART_CENTER,    // the center of a restart
        RESTART_REPELLING, // a point of a hill-valley test of the repelling archive
        POPULATION,        // individuals next..next+n_individuals-1, then n_extra extra points
        EXTRA,             // the extra points (TPA) after a population that stopped early
        DONE               // all points of the generation are evaluated
    };

    Phase phase = Phase::IDLE;
    Eigen::Index next = 0;
    Eigen::Index n_individuals = 0;
    Eigen::Index n_extra = 0;
    Eigen::Index n_pending = 0;

    //! The pending points (the first n_pending columns), and the objective values in step()
    Matrix batch;
    Vector values;

    bool tell(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X);
    void reserve_batch(const Eigen::Index k);
    void start_generation();
    void start_population();
    void sample_batch();
    void next_restart_point();
    void inject(const Eigen::Index i, Eigen::Ref<const Vector> x) const;
    void consume(Eigen::Ref<const Vector> f, const Eigen::Ref<const Matrix> *X);
    void evaluate(FunctionType &objective);
};
