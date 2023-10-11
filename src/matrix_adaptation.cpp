#include "matrix_adaptation.hpp"

namespace matrix_adaptation
{
    using namespace parameters;


    void CovarianceAdaptation::adapt_evolution_paths(const Population &pop, const Weights &w, const std::shared_ptr<mutation::Strategy> &mutation, const Stats &stats, const size_t mu, const size_t lambda)
    {
        dm = (m - m_old) / mutation->sigma;
        ps = (1.0 - mutation->cs) * ps + (sqrt(mutation->cs * (2.0 - mutation->cs) * w.mueff) * inv_root_C * dm);

        const double actual_ps_length = ps.norm() / sqrt(1.0 - pow(1.0 - mutation->cs, 2.0 * (stats.evaluations / lambda)));
        const double expected_ps_length = (1.4 + (2.0 / (dd + 1.0))) * chiN;

        hs = actual_ps_length < expected_ps_length;
        pc = (1.0 - w.cc) * pc + (hs * sqrt(w.cc * (2.0 - w.cc) * w.mueff)) * dm;
    }

    void CovarianceAdaptation::adapt_covariance_matrix(const Weights &w, const Modules &m, const Population &pop, const size_t mu)
    {
        const auto rank_one = w.c1 * pc * pc.transpose();
        const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);
        const auto old_C = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * w.positive.sum())) * C;

        Matrix rank_mu;
        if (m.active)
        {
            auto weights = w.weights.topRows(pop.Y.cols());
            auto neg_scaler = dd / (inv_root_C * pop.Y).colwise().norm().array().pow(2).transpose();
            auto w2 = (weights.array() < 0).select(weights.array() * neg_scaler, weights);
            
            rank_mu = w.cmu * ((pop.Y.array().rowwise() * w2.array().transpose()).matrix() * pop.Y.transpose());
        }
        else
        {
            rank_mu = w.cmu * ((pop.Y.leftCols(mu).array().rowwise() * w.positive.array().transpose()).matrix() * pop.Y.leftCols(mu).transpose());
        }

        C = old_C + rank_one + rank_mu;

        C = C.triangularView<Eigen::Upper>().toDenseMatrix() +
            C.triangularView<Eigen::StrictlyUpper>().toDenseMatrix().transpose();
    }

    bool CovarianceAdaptation::perform_eigendecomposition(const Settings &settings)
    {
        Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(C);
        if (eigensolver.info() != Eigen::Success)
        {
            if (settings.verbose)
            {
                std::cout << "Eigensolver failed, we need to restart reason:"
                          << eigensolver.info() << std::endl;
            }
            return false;
        }
        d = eigensolver.eigenvalues();

        if (d.minCoeff() < 0.0)
        {
            if (settings.verbose)
            {
                std::cout << "Negative eigenvalues after decomposition, we need to restart.\n";
            }
            return false;
        }

        d = d.cwiseSqrt();
        B = eigensolver.eigenvectors();
        inv_root_C = (B * d.cwiseInverse().asDiagonal()) * B.transpose();
        return true;
    }

    bool CovarianceAdaptation::adapt_matrix(const Weights &w, const Modules &m, const Population &pop, const size_t mu, const Settings &settings)
    {
        adapt_covariance_matrix(w, m, pop, mu);
        return perform_eigendecomposition(settings);
    }

    void CovarianceAdaptation::restart(const Settings &settings)
    {
        B = Matrix::Identity(settings.dim, settings.dim);
        C = Matrix::Identity(settings.dim, settings.dim);
        inv_root_C = Matrix::Identity(settings.dim, settings.dim);
        d.setOnes();
        m = settings.x0.value_or(Vector::Zero(settings.dim));
        m_old.setZero();
        dm.setZero();
        pc.setZero();
        ps.setZero();
    }

    void CovarianceAdaptation::scale_mutation_steps(Population &pop)
    {
        pop.Y = B * (d.asDiagonal() * pop.Z);
    }

    void CovarianceAdaptation::invert_mutation_steps(Population &pop, size_t n_offspring) {
        // z = diag(1 / D) * (B^{-1} * pop.Y)
        // y = (x - m) / sigma
    }

    void MatrixAdaptation::adapt_evolution_paths(const Population &pop, const Weights &w, const std::shared_ptr<mutation::Strategy> &mutation, const Stats &stats, const size_t mu, const size_t lambda)
    {
        dm = (m - m_old) / mutation->sigma;

        const auto dz = (pop.Z.leftCols(mu).array().rowwise() * w.positive.array().transpose()).rowwise().sum().matrix();
        ps = (1.0 - mutation->cs) * ps + (sqrt(mutation->cs * (2.0 - mutation->cs) * w.mueff) * dz);
    }

    bool MatrixAdaptation::adapt_matrix(const Weights &w, const Modules &m, const Population &pop, const size_t mu, const Settings &settings)
    {
        const auto old_M = (1 - 0.5 * w.c1 - 0.5 * w.cmu) * M;
        const auto scaled_ps = (0.5 * w.c1) * (M * ps) * ps.transpose();

        Matrix new_M;
        if (m.active)
        {
            // TODO: Check if we can do this like this
            const auto scaled_weights = (0.5 * w.cmu * w.weights.topRows(pop.Y.cols())).array().transpose();
            new_M = (pop.Y.array().rowwise() * scaled_weights).matrix() * pop.Z.transpose();
        }
        else
        {
            const auto scaled_weights = (0.5 * w.cmu * w.positive).array().transpose();
            new_M = (pop.Y.leftCols(mu).array().rowwise() * scaled_weights).matrix() * pop.Z.leftCols(mu).transpose();
        }

        M = old_M + scaled_ps + new_M;
        return true;
    }

    void MatrixAdaptation::restart(const Settings &settings)
    {
        ps.setZero();
        m = settings.x0.value_or(Vector::Zero(settings.dim));
        m_old.setZero();
        dm.setZero();
    }

    void MatrixAdaptation::scale_mutation_steps(Population &pop)
    {
        pop.Y = M * pop.Z;
    }
    
    void MatrixAdaptation::invert_mutation_steps(Population &pop, size_t n_offspring) {
        // z = M^{-1}y
        // y = (x - m) / sigma
    }
}