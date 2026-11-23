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
    const auto idx = utils::sort_indexes(f.head(k).eval());
    X.leftCols(k) = X(Eigen::all, idx).eval();
    X_transformed.leftCols(k) = X_transformed(Eigen::all, idx).eval();
    Z.leftCols(k) = Z(Eigen::all, idx).eval();
    Y.leftCols(k) = Y(Eigen::all, idx).eval();
    S.leftCols(k) = S(Eigen::all, idx).eval();
    f.head(k) = f(idx).eval();
    t.head(k) = t(idx).eval();
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
    // Keeps at least the active columns; shrinking the buffers below their capacity only when
    // they hold more than size columns (elitism appends the parents; WP2 keeps them instead)
    n = std::min(size, n);
    const auto c = static_cast<Eigen::Index>(std::max(n, std::min(size, capacity())));
    X.conservativeResize(d, c);
    X_transformed.conservativeResize(d, c);
    Y.conservativeResize(d, c);
    Z.conservativeResize(d, c);
    S.conservativeResize(d, c);
    f.conservativeResize(c);
    t.conservativeResize(c);
}

void Population::keep_only(const std::vector<size_t> &idx)
{
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
