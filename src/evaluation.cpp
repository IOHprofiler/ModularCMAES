#include "evaluation.hpp"

#include <stdexcept>
#include <utility>

namespace evaluation
{
const char *to_string(const GenerationStage stage)
{
    switch (stage)
    {
    case GenerationStage::RUN_ENDING:
        return "RUN_ENDING";
    case GenerationStage::RUN_STARTING:
        return "RUN_STARTING";
    case GenerationStage::GENERATION_STARTING:
        return "GENERATION_STARTING";
    case GenerationStage::POPULATION:
        return "POPULATION";
    case GenerationStage::ALONGSIDE_POPULATION:
        return "ALONGSIDE_POPULATION";
    case GenerationStage::POPULATION_EVALUATED:
        return "POPULATION_EVALUATED";
    case GenerationStage::SELECTED:
        return "SELECTED";
    case GenerationStage::UPDATED:
        return "UPDATED";
    }
    return "unknown";
}

Eigen::Ref<Matrix> EvaluationQueue::push(const Eigen::Index k, Consumer consume, const char *source)
{
    if (k <= 0)
        throw std::invalid_argument("EvaluationQueue::push: a request needs at least one point");
    const Eigen::Index needed = n_points + k;
    if (points.cols() < needed)
        // Grows rarely (a population of a new size); keeps the pending points
        points.conservativeResize(Eigen::NoChange, std::max(needed, 2 * points.cols()));
    requests.push_back(Request{n_points, k, std::move(consume), source});
    n_points = needed;
    return points.middleCols(needed - k, k);
}

void EvaluationQueue::push(Eigen::Ref<const Matrix> X, Consumer consume, const char *source)
{
    if (points.rows() != X.rows())
        points.resize(X.rows(), std::max<Eigen::Index>(X.cols(), points.cols()));
    push(X.cols(), std::move(consume), source) = X;
}

std::vector<std::string> EvaluationQueue::sources() const
{
    std::vector<std::string> out;
    out.reserve(requests.size());
    for (const auto &r : requests)
        out.push_back(r.source);
    return out;
}

void EvaluationQueue::answer(parameters::Parameters &p,
                             Eigen::Ref<const Vector> f,
                             const Eigen::Ref<const Matrix> *X)
{
    if (requests.empty())
        throw std::logic_error("no points are pending: call ask() first");
    if (f.size() != n_points)
        throw std::invalid_argument("expected " + std::to_string(n_points) +
                                    " objective values (the points of ask()), got " +
                                    std::to_string(f.size()));
    if (X != nullptr && (X->rows() != points.rows() || X->cols() != n_points))
        throw std::invalid_argument("expected the points as a " + std::to_string(points.rows()) +
                                    " x " + std::to_string(n_points) + " matrix, got " +
                                    std::to_string(X->rows()) + " x " +
                                    std::to_string(X->cols()));

    // The answered batch moves aside (a swap of buffers: no copy); consumers push to the next one.
    // f is only read while the consumers run, here
    const Eigen::Index n = n_points;
    std::swap(points, answering_points);
    std::swap(requests, answering);
    if (points.rows() != answering_points.rows())
        points.resize(answering_points.rows(), answering_points.cols());
    requests.clear();
    n_points = 0;

    const Eigen::Ref<const Matrix> evaluated =
        X != nullptr ? *X : Eigen::Ref<const Matrix>(answering_points.leftCols(n));
    for (const auto &r : answering)
        r.consume(p,
                  Evaluations{f.segment(r.begin, r.size),
                              evaluated.middleCols(r.begin, r.size),
                              X != nullptr});
    answering.clear();
}

void EvaluationQueue::drain(parameters::Parameters &p, FunctionType &objective)
{
    Vector f;
    while (!empty())
    {
        f.resize(n_points);
        for (Eigen::Index j = 0; j < n_points; ++j)
            f(j) = objective(points.col(j));
        answer(p, f);
    }
}

void Pipeline::on(const GenerationStage stage, std::string name, Hook hook, MinPoints min_points)
{
    stages[static_cast<size_t>(stage)].push_back(
        Subscriber{std::move(name), std::move(hook), std::move(min_points)});
}

void Pipeline::fire(const GenerationStage stage, parameters::Parameters &p)
{
    fired_[static_cast<size_t>(stage)] = true;
    for (const auto &s : stages[static_cast<size_t>(stage)])
        s.hook(p);
}

size_t Pipeline::fixed_points(const parameters::Parameters &p) const
{
    size_t n = 0;
    for (const auto &stage : stages)
        for (const auto &s : stage)
            n += s.min_points(p);
    return n;
}
} // namespace evaluation
