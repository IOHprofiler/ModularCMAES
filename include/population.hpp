#pragma once

#include "common.hpp"

struct Population
{
    Matrix X;
    Matrix X_transformed;
    Matrix Z;
    Matrix Y;
    Matrix S;
    Vector f;
    Vector t;

    size_t d;
    //! The number of active columns: individuals 0..n-1 belong to the population. The buffers
    //! can hold more (capacity()), e.g. after sequential selection stopped a generation early;
    //! the columns from n on hold no population data and must not be read, except the dropped
    //! points of an elitist selection (n_pool).
    size_t n;
    /**
     * With elitism, the size of the sorted (mu + lambda) pool of which the selection kept the
     * first n (selection::Strategy::select). Columns n..n_pool-1 then hold the points it dropped,
     * the worst of the pool, in their order: the active update of the matrix gives them the
     * negative weights (#161). 0, or at most n, when no point was dropped; reset by every change of
     * the population other than resize_cols.
     */
    size_t n_pool = 0;

    //! The number of points that the elitist selection dropped (columns n..n + n_dropped() - 1)
    [[nodiscard]] size_t n_dropped() const { return n_pool > n ? n_pool - n : 0; }
    /**
     * The individuals 0..n_ranked-1 are ranked by f; the rest are the losers of pairwise selection
     * (Mirror::PAIRWISE), in the order of their f, which get no negative weights (#75, F9). Set by
     * the selection; every n individuals are ranked when it is n or more (the default, and after
     * sort(), keep_only() and +=). ranked() is the number.
     */
    size_t n_ranked = std::numeric_limits<size_t>::max();

    [[nodiscard]] size_t ranked() const { return std::min(n_ranked, n); }

    [[nodiscard]] size_t capacity() const { return static_cast<size_t>(X.cols()); }

    //! Grow the buffers to at least the given number of columns, keeping their contents
    void reserve(const size_t columns);

    //! The buffers start at 0 (not uninitialized memory): the columns that no generation has
    //! written yet are saved with the state, which is then the same for the same run (#91)
    Population(const size_t d, const size_t n) :
        X(Matrix::Zero(d, n)),
        X_transformed(Matrix::Zero(d, n)),
        Z(Matrix::Zero(d, n)),
        Y(Matrix::Zero(d, n)),
        S(Matrix::Zero(d, n)),
        f(Vector::Constant(n, std::numeric_limits<Float>::infinity())),
        t(Vector::Zero(n)),
        d(d),
        n(n)
    {
    }

    Population(const Matrix &X,
               const Matrix &Z,
               const Matrix &Y,
               const Vector &f,
               const Matrix &S) :
        X(X),
        X_transformed(X),
        Z(Z),
        Y(Y),
        S(S),
        f(f),
        t(Vector::Zero(f.rows())),
        d(X.rows()),
        n(X.cols())
    {
    }

    Population() :
        Population(0, 0)
    {
    }

    //! Sort the active columns by f (ties in index order, NaN last), without allocating
    void sort();

    //! Put the active columns in the given order (a permutation of 0..n-1), without allocating
    //! once the buffers have the capacity; n_ranked is left to the caller
    void reorder(const std::vector<size_t> &order);

    Population &operator+=(const Population &other);

    void resize_cols(const size_t size);

    void keep_only(const std::vector<size_t> &idx);

    size_t n_finite() const;

  private:
    // Reused by sort(): the order of the columns and a buffer to gather them into
    std::vector<size_t> order;
    Matrix scratch;
    Vector scratch_vector;
};

std::ostream &operator<<(std::ostream &os, const Population &dt);