#include "evaluation.hpp"

#include <stdexcept>
#include <utility>

#include "parameters.hpp"

namespace evaluation
{
const char *to_string(const Source source)
{
    switch (source)
    {
    case Source::POPULATION:
        return "population";
    case Source::EXTRA_POINTS:
        return "extra-points";
    case Source::RESTART_CENTER:
        return "restart-center";
    case Source::HILL_VALLEY:
        return "hill-valley";
    }
    return "unknown";
}

Eigen::Ref<Matrix> EvaluationQueue::push(const Eigen::Index k,
                                         const Source source,
                                         const Eigen::Index index)
{
    if (k <= 0)
        throw std::invalid_argument("EvaluationQueue::push: a request needs at least one point");
    const Eigen::Index needed = n_points + k;
    if (points.cols() < needed)
        // Grows rarely (a population of a new size); keeps the pending points
        points.conservativeResize(Eigen::NoChange, std::max(needed, 2 * points.cols()));
    requests.push_back(Request{source, n_points, k, index});
    n_points = needed;
    return points.middleCols(needed - k, k);
}

void EvaluationQueue::push(Eigen::Ref<const Matrix> X,
                           const Source source,
                           const Eigen::Index index)
{
    if (X.rows() != points.rows())
        throw std::invalid_argument("EvaluationQueue::push: points of dimension " +
                                    std::to_string(X.rows()) + ", the queue holds dimension " +
                                    std::to_string(points.rows()));
    push(X.cols(), source, index) = X;
}

std::vector<std::string> EvaluationQueue::sources() const
{
    std::vector<std::string> out;
    out.reserve(requests.size());
    for (const auto &r : requests)
        out.emplace_back(to_string(r.source));
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

    // The answered batch moves aside (a swap of buffers: no copy); the queue is empty while the
    // values are dispatched. A request left from a dispatch that threw is dropped here
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
        p.dispatch(r,
                   Evaluations{f.segment(r.begin, r.size),
                               evaluated.middleCols(r.begin, r.size),
                               X != nullptr});
    answering.clear();
}
} // namespace evaluation
