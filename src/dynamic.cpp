#include "parameters.hpp"

namespace parameters
{
    Dynamic::Dynamic(const size_t dim, const Vector& x0) : m(x0), m_old(dim), dm(Vector::Zero(dim)), pc(Vector::Zero(dim)),
                                         ps(Vector::Zero(dim)), d(Vector::Ones(dim)),
                                         B(Matrix::Identity(dim, dim)), C(Matrix::Identity(dim, dim)),
                                         inv_root_C(Matrix::Identity(dim, dim)), 
                                         dd(static_cast<double>(dim)),
                                         chiN(sqrt(dd) * (1.0 - 1.0 / (4.0 * dd) + 1.0 / (21.0 * pow(dd, 2.0))))
    {
    }

    void Dynamic::adapt_evolution_paths(const Weights &w, const std::shared_ptr<mutation::Strategy> &mutation, const Stats &stats, const size_t lambda)
    {
        dm = (m - m_old) / mutation->sigma;
        ps = (1.0 - mutation->cs) * ps + (sqrt(mutation->cs * (2.0 - mutation->cs) * w.mueff) * inv_root_C * dm);

        const double actual_ps_length = ps.norm() / sqrt(1.0 - pow(1.0 - mutation->cs, 2.0 * (stats.evaluations / lambda)));
        const double expected_ps_length = (1.4 + (2.0 / (dd + 1.0))) * chiN;

        hs = actual_ps_length < expected_ps_length;
        pc = (1.0 - w.cc) * pc + (hs * sqrt(w.cc * (2.0 - w.cc) * w.mueff)) * dm;
    }

    void Dynamic::adapt_covariance_matrix(const Weights &w, const Modules &m, const Population &pop, const size_t mu)
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

    bool Dynamic::perform_eigendecomposition(const Settings &settings)
    {
        Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(C);
        if (eigensolver.info() != Eigen::Success)
        {
            if (settings.verbose){
                // TODO: check why this sometimes happens on the first eval (sphere 60d)
                std::cout << "Eigensolver failed, we need to restart reason:"
                        << eigensolver.info() << std::endl;
            }
            return false;
        }
        d = eigensolver.eigenvalues();
       
        if (d.minCoeff() < 0.0){
            if (settings.verbose) {
                std::cout << "Negative eigenvalues after decomposition, we need to restart.\n";
            }
            return false;
        }              
        
        d = d.cwiseSqrt();
        B = eigensolver.eigenvectors();
        inv_root_C = (B * d.cwiseInverse().asDiagonal()) * B.transpose();
        return true;
    }
}