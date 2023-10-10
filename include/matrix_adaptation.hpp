#pragma once

#include "common.hpp"
#include "modules.hpp"
#include "mutation.hpp"
#include "population.hpp"
#include "settings.hpp"
#include "stats.hpp"
#include "weights.hpp"

namespace matrix_adaptation
{
    struct Adaptation
    {
        Vector m, m_old, dm, ps;
        double dd;
        double chiN;

        Adaptation(const size_t dim, const Vector &x0) : m(x0), m_old(dim), dm(Vector::Zero(dim)),
                                                         ps(Vector::Zero(dim)), dd(static_cast<double>(dim)),
                                                         chiN(sqrt(dd) * (1.0 - 1.0 / (4.0 * dd) + 1.0 / (21.0 * pow(dd, 2.0)))) {}

        virtual void adapt_evolution_paths(const Population &pop, const parameters::Weights &w, const std::shared_ptr<mutation::Strategy> &mutation, const parameters::Stats &stats, const size_t mu, const size_t lambda) = 0;
        virtual bool adapt_matrix(const parameters::Weights &w, const parameters::Modules &m, const Population &pop, const size_t mu, const parameters::Settings &settings) = 0;
        virtual void restart(const parameters::Settings &settings) = 0;
        virtual void scale_mutation_steps(Population &pop) = 0;
    };

    struct CovarianceAdaptation : Adaptation
    {
        Vector pc, d;
        Matrix B, C;
        Matrix inv_root_C;
        bool hs = true;

        CovarianceAdaptation(const size_t dim, const Vector &x0) : Adaptation(dim, x0), pc(Vector::Zero(dim)),
                                                         d(Vector::Ones(dim)),
                                                         B(Matrix::Identity(dim, dim)), C(Matrix::Identity(dim, dim)),
                                                         inv_root_C(Matrix::Identity(dim, dim))

        {
        }

        void scale_mutation_steps(Population &pop) override;

        void adapt_evolution_paths(const Population &pop, const parameters::Weights &w, const std::shared_ptr<mutation::Strategy> &mutation, const parameters::Stats &stats, const size_t mu, const size_t lambda) override;

        void adapt_covariance_matrix(const parameters::Weights &w, const parameters::Modules &m, const Population &pop, const size_t mu);

        bool perform_eigendecomposition(const parameters::Settings &settings);

        bool adapt_matrix(const parameters::Weights &w, const parameters::Modules &m, const Population &pop, const size_t mu, const parameters::Settings &settings) override;

        void restart(const parameters::Settings &settings) override;
    };

    struct MatrixAdaptation : Adaptation {
        Matrix M;
        MatrixAdaptation(const size_t dim, const Vector &x0) : Adaptation(dim, x0), M(Matrix::Identity(dim, dim)) {}

        void adapt_evolution_paths(const Population &pop, const parameters::Weights &w, const std::shared_ptr<mutation::Strategy> &mutation, const parameters::Stats &stats, const size_t mu, const size_t lambda) override;
        bool adapt_matrix(const parameters::Weights &w, const parameters::Modules &m, const Population &pop, const size_t mu, const parameters::Settings &settings)  override;
        void restart(const parameters::Settings &settings)  override;
        void scale_mutation_steps(Population &pop)  override;
    };



    inline std::shared_ptr<Adaptation> get(const parameters::Modules &m, const size_t dim, const Vector &x0)
    {
        using namespace parameters;
        switch (m.matrix_adaptation)
        {
        case MatrixAdaptationType::MATRIX:
            return std::make_shared<MatrixAdaptation>(dim, x0);
        default:
        case MatrixAdaptationType::COVARIANCE:
            return std::make_shared<CovarianceAdaptation>(dim, x0);
        }
    }
} // namespace parameters
