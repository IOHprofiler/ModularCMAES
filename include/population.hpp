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
    size_t n;

    Population(const size_t d, const size_t n)
        : X(d, n), X_transformed(d, n), Z(d, n), Y(d, n), S(d, n), f(Vector::Constant(n, std::numeric_limits<Float>::infinity())), t(n), d(d), n(n) {}

    Population(const Matrix &X, const Matrix &Z, const Matrix &Y, const Vector &f, const Matrix &S)
        : X(X), X_transformed(X), Z(Z), Y(Y), S(S), f(f),  t(f.rows()), d(X.rows()), n(X.cols()) {}

    Population() : Population(0, 0) {}

    void sort();

    Population &operator+=(const Population &other);

    void resize_cols(const size_t size);

    void keep_only(const std::vector<size_t> &idx);

    size_t n_finite() const;
};

std::ostream &operator<<(std::ostream &os, const Population &dt);