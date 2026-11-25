#pragma once

#include <string>
#include <vector>

#include "common.hpp"

namespace parameters
{
struct Parameters;
}

/**
 * Every evaluation of the objective goes through one queue (#84). A module that needs objective
 * values pushes a request: its points, tagged with their source. ask() hands out what is queued;
 * tell() passes each request's values to Parameters::dispatch, a switch over the sources that calls
 * the module through the parameters (the module in place at that time: never a pointer kept since
 * the push). The module may then push follow-up requests. No module calls the objective itself.
 *
 * When a module pushes is fixed by the stages of a generation: ModularCMAES::advance calls the
 * modules at each stage, in order, and waits for tell() whenever the queue is not empty.
 */
namespace evaluation
{
//! The stages of a generation, in order (C++ only: not part of the Python API)
enum class Stage
{
    //! The restart criteria fired: the final mean of the run that ends is evaluated, then the
    //! hill-valley points of the repelling archive, one at a time
    RESTART,
    //! The population, in batches when sequential selection may stop it early, with the points
    //! that do not depend on the population's values (TPA's two) in the first batch
    POPULATION,
    //! Every point of the population has its value; before selection
    POPULATION_EVALUATED,
    //! After selection, before recombination and adaptation
    SELECTED,
};

constexpr int n_stages = static_cast<int>(Stage::SELECTED) + 1;

//! Who pushed a request: Parameters::dispatch passes its values on by this tag
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

//! The name of a source ("population", "extra-points", "restart-center", "hill-valley")
const char *to_string(Source source);

//! What a module receives: the values of its points, and the points that were evaluated (tell(f,
//! X) may replace the asked ones)
struct Evaluations
{
    Eigen::Ref<const Vector> f;
    Eigen::Ref<const Matrix> X;
    //! Whether X was given by tell(f, X) (a column may then differ from the asked point)
    bool given;
};

//! A queued request: its points are columns begin..begin+size-1 of the batch
struct Request
{
    Source source;
    Eigen::Index begin;
    Eigen::Index size;
    //! A payload for the module, by source (POPULATION: the first individual of the batch)
    Eigen::Index index;
};

class EvaluationQueue
{
  public:
    explicit EvaluationQueue(size_t dim = 0) : points(static_cast<Eigen::Index>(dim), 0) {}

    //! Queue the points X (copied), d x k
    void push(Eigen::Ref<const Matrix> X, Source source, Eigen::Index index = 0);

    /**
     * Queue k points that the caller fills: the returned columns. The view is valid until the next
     * push (which may move the buffer); fill it first.
     */
    Eigen::Ref<Matrix> push(Eigen::Index k, Source source, Eigen::Index index = 0);

    [[nodiscard]] bool empty() const { return requests.empty(); }

    //! The number of points that wait for their values
    [[nodiscard]] Eigen::Index size() const { return n_points; }

    //! The pending points, in push order (one per column)
    [[nodiscard]] Eigen::Ref<const Matrix> batch() const { return points.leftCols(n_points); }

    //! The source of every pending request, in push order
    [[nodiscard]] std::vector<std::string> sources() const;

    /**
     * The values f of the pending points (X: the evaluated points, when they differ from the asked
     * ones). Each request's slice goes to Parameters::dispatch, in push order; requests pushed
     * meanwhile form the next batch.
     */
    void answer(parameters::Parameters &p,
                Eigen::Ref<const Vector> f,
                const Eigen::Ref<const Matrix> *X = nullptr);

    //! Evaluate the pending points with an objective function until the queue is empty
    void drain(parameters::Parameters &p, FunctionType &objective);

  private:
    Matrix points;
    Eigen::Index n_points = 0;
    std::vector<Request> requests;

    // The batch being answered: its requests and points stay valid while they are dispatched
    Matrix answering_points;
    std::vector<Request> answering;
};
} // namespace evaluation
