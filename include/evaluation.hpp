#pragma once

#include "common.hpp"

/**
 * The types of the evaluation flow (#84). ModularCMAES::advance is the only place that queues points
 * to evaluate, step by step (Step), each request tagged with its source; ask() hands out the queued
 * points, and tell() passes each request's values to ModularCMAES::dispatch, a switch over the
 * sources that calls the module (the one in place at that time). Modules compute their points and
 * receive their values; they never queue points themselves, and none calls the objective.
 */
namespace evaluation
{
/**
 * The steps of a generation, in order (ModularCMAES::advance; C++ only). A step may queue points;
 * the next step runs once they all have their values.
 */
enum class Step
{
    //! A new generation. When the restart criteria fired, the final mean of the run that ends is
    //! queued
    BEGIN,
    //! The restart: the hill-valley points of the repelling archive are queued one at a time, then
    //! the modules are reset for the new run
    RESTART,
    //! A new population: its first batch is queued, with the points that do not depend on its
    //! values (TPA's two)
    SAMPLE,
    //! The next batch of the population is queued until it is complete (sequential selection)
    POPULATION,
    //! Every point of the population has its value: selection
    SELECT,
    //! Recombination and adaptation; the generation is complete
    UPDATE,
};

constexpr int n_steps = static_cast<int>(Step::UPDATE) + 1;

//! Who pushed a request: ModularCMAES::dispatch passes its values on by this tag
enum class Source
{
    //! A batch of the population (Request::index: its first individual)
    POPULATION,
    //! The extra points of the step-size adaptation (TPA)
    EXTRA_POINTS,
    //! The final mean of the run that ends at a restart
    RESTART_CENTER,
    //! A point of a hill-valley test of the repelling archive
    HILL_VALLEY,
};


//! What a module receives: the values of its points, and the points that were evaluated (tell(f,
//! X) may replace the asked ones)
struct Evaluations
{
    Eigen::Ref<const Vector> f;
    Eigen::Ref<const Matrix> X;
    //! Whether X was given by tell(f, X) (a column may then differ from the asked point)
    bool given;
};

//! A queued request: its points are columns begin..begin+size-1 of the batch of ask()
struct Request
{
    Source source;
    Eigen::Index begin;
    Eigen::Index size;
    //! A payload for the module, by source (POPULATION: the first individual of the batch)
    Eigen::Index index;
};

} // namespace evaluation
