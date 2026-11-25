#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "common.hpp"

namespace parameters
{
struct Parameters;
}

/**
 * Every evaluation of the objective goes through one queue (#84). A component that needs objective
 * values pushes a request (points and a consumer) to the queue; ask() hands out what is queued,
 * tell() passes each request's values to its consumer, which may push follow-up requests. No
 * component calls the objective itself.
 *
 * When a component may push is fixed by the stages of a generation (GenerationStage). The driver
 * (ModularCMAES) runs the stages in order, fires their subscribers, and waits for tell() whenever the
 * queue is not empty.
 */
namespace evaluation
{
/**
 * The stages of a generation, in order. Each is a point at which subscribers may push requests;
 * the next stage starts when the queue is empty.
 */
enum class GenerationStage
{
    //! The restart criteria fired; the run that ends is still intact (the final mean is evaluated,
    //! repelling tests the hill-valley points)
    RUN_ENDING,
    //! A new run is set up (lambda, mean, sigma, matrix); nothing is sampled yet
    RUN_STARTING,
    //! The mean, sigma and matrix of this generation are fixed; nothing is sampled yet
    GENERATION_STARTING,
    //! The population: lambda points, in batches when sequential selection may stop it early.
    //! Its subscribers add points that do not depend on the population's values (TPA's two
    //! points): they join the population's first batch
    POPULATION,
    //! Every point of the population has its value; before selection
    POPULATION_EVALUATED,
    //! After selection, before recombination
    SELECTED,
    //! The mean, sigma and matrix are updated; the generation ends when the queue is empty
    UPDATED,
};

constexpr size_t n_stages = static_cast<size_t>(GenerationStage::UPDATED) + 1;

//! The name of a stage ("RUN_ENDING", ...)
const char *to_string(GenerationStage stage);

//! What a consumer receives: the values of its points, and the points that were evaluated (tell(f,
//! X) may replace the asked ones)
struct Evaluations
{
    Eigen::Ref<const Vector> f;
    Eigen::Ref<const Matrix> X;
    //! Whether X was given by tell(f, X) (a column may then differ from the asked point)
    bool given;
};

using Consumer = std::function<void(parameters::Parameters &, const Evaluations &)>;

struct Request
{
    Eigen::Index begin;   // the first column in the batch
    Eigen::Index size;    // the number of points
    Consumer consume;
    const char *source;   // "population", "restart-center", ... (a string literal; debugging)
};

class EvaluationQueue
{
  public:
    explicit EvaluationQueue(size_t dim = 0) : points(static_cast<Eigen::Index>(dim), 0) {}

    /**
     * Queue k points. Returns their columns, to be filled by the caller before the next ask().
     * Pushed while tell() answers a batch, they go to the next batch.
     */
    Eigen::Ref<Matrix> push(Eigen::Index k, Consumer consume, const char *source);

    //! Queue the points X
    void push(Eigen::Ref<const Matrix> X, Consumer consume, const char *source);

    [[nodiscard]] bool empty() const { return requests.empty(); }

    //! The number of points that wait for their values
    [[nodiscard]] Eigen::Index size() const { return n_points; }

    //! The pending points, in push order (one per column)
    [[nodiscard]] Eigen::Ref<const Matrix> batch() const { return points.leftCols(n_points); }

    //! The source of every pending request, in push order
    [[nodiscard]] std::vector<std::string> sources() const;

    /**
     * The values f of the pending points (X: the evaluated points, when they differ from the asked
     * ones). Each request's slice goes to its consumer, in push order; requests pushed meanwhile
     * form the next batch.
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

    // The batch being answered: its requests and points stay valid while the consumers run
    Matrix answering_points;
    std::vector<Request> answering;
};

/**
 * The subscribers of each stage. A subscriber pushes its requests when its stage is fired, and
 * declares the points it adds to every generation (min_points; e.g. TPA's 2), which the budget
 * check reserves. A subscriber whose number of points varies declares 0 and clips its requests to
 * the budget (Parameters::push_variable refuses more).
 */
class Pipeline
{
  public:
    using Hook = std::function<void(parameters::Parameters &)>;
    using MinPoints = std::function<size_t(const parameters::Parameters &)>;

    struct Subscriber
    {
        std::string name;
        Hook hook;
        MinPoints min_points;
    };

    void on(GenerationStage stage,
            std::string name,
            Hook hook,
            MinPoints min_points = [](const parameters::Parameters &) { return size_t{0}; });

    //! Run the subscribers of a stage, in the order they subscribed
    void fire(GenerationStage stage, parameters::Parameters &p);

    //! The points every generation adds besides the population (the sum of min_points)
    [[nodiscard]] size_t fixed_points(const parameters::Parameters &p) const;

    [[nodiscard]] const std::vector<Subscriber> &subscribers(GenerationStage stage) const
    {
        return stages[static_cast<size_t>(stage)];
    }

  private:
    std::array<std::vector<Subscriber>, n_stages> stages;
};
} // namespace evaluation
