#include "population.hpp"

void Population::reserve(const size_t columns)
{
    if (capacity() >= columns)
        return;
    const auto c = static_cast<Eigen::Index>(columns);
    X.conservativeResize(Eigen::NoChange, c);
    X_transformed.conservativeResize(Eigen::NoChange, c);
    Y.conservativeResize(Eigen::NoChange, c);
    Z.conservativeResize(Eigen::NoChange, c);
    S.conservativeResize(Eigen::NoChange, c);
    f.conservativeResize(c);
    t.conservativeResize(c);
}

void Population::sort()
{
    const Eigen::Index k = static_cast<Eigen::Index>(n);
    utils::sort_order(f.head(k), order);

    // Gather each array into the scratch buffer, then copy it back
    if (scratch.rows() != X.rows() || scratch.cols() < k)
        scratch.resize(X.rows(), static_cast<Eigen::Index>(capacity()));
    if (scratch_vector.size() < k)
        scratch_vector.resize(static_cast<Eigen::Index>(capacity()));

    // Column by column: an Eigen index view would copy the order vector (an allocation)
    for (Matrix *m : {&X, &X_transformed, &Z, &Y, &S})
    {
        for (Eigen::Index j = 0; j < k; ++j)
            scratch.col(j) = m->col(static_cast<Eigen::Index>(order[j]));
        m->leftCols(k) = scratch.leftCols(k);
    }
    for (Vector *v : {&f, &t})
    {
        for (Eigen::Index j = 0; j < k; ++j)
            scratch_vector(j) = (*v)(static_cast<Eigen::Index>(order[j]));
        v->head(k) = scratch_vector.head(k);
    }
}

Population &Population::operator+=(const Population &other)
{
    // The active columns of other go directly after the active columns of this population
    const auto k = static_cast<Eigen::Index>(n), m = static_cast<Eigen::Index>(other.n);
    reserve(n + other.n);
    X.middleCols(k, m) = other.X.leftCols(m);
    X_transformed.middleCols(k, m) = other.X_transformed.leftCols(m);
    Y.middleCols(k, m) = other.Y.leftCols(m);
    Z.middleCols(k, m) = other.Z.leftCols(m);
    S.middleCols(k, m) = other.S.leftCols(m);
    f.segment(k, m) = other.f.head(m);
    t.segment(k, m) = other.t.head(m);
    n += other.n;
    return *this;
}

void Population::resize_cols(const size_t size)
{
    // Only the number of active columns changes; the buffers keep their capacity (elitism appends
    // the parents every generation, which would otherwise reallocate twice per generation)
    n = std::min(size, n);
}

void Population::keep_only(const std::vector<size_t> &idx)
{
    for (const auto i : idx)
        if (i >= capacity())
            throw std::out_of_range("Population.keep_only: column " + std::to_string(i) + " of " +
                                    std::to_string(capacity()));
    X = X(Eigen::all, idx).eval();
    X_transformed = X_transformed(Eigen::all, idx).eval();
    Z = Z(Eigen::all, idx).eval();
    Y = Y(Eigen::all, idx).eval();
    S = S(Eigen::all, idx).eval();
    f = f(idx).eval();
    t = t(idx).eval();
    n = idx.size();
}

size_t Population::n_finite() const
{
    return (f.head(static_cast<Eigen::Index>(n)).array() != std::numeric_limits<Float>::infinity())
        .cast<size_t>()
        .sum();
}

std::ostream &operator<<(std::ostream &os, const Population &p)
{
    const auto k = static_cast<Eigen::Index>(p.n);
    os << "Population" << "\nx=\n"
       << p.X_transformed.leftCols(k) << "\ny=\n"
       << p.Y.leftCols(k) << "\nf=\n"
       << p.f.head(k).transpose();
    return os;
}
