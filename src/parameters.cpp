#include "parameters.hpp"

namespace parameters
{
    Parameters::Parameters(const size_t dim, const Modules& m) : 
        dim(dim), 
        lambda(4 + std::floor(3 * std::log(dim))),
        mu(lambda / 2),
        modules(m), 
        dynamic(dim),
        weights(dim, mu, lambda, modules),
        pop(dim, lambda),
        old_pop(dim, lambda),        
        sampler(sampling::get(dim, modules, lambda)),
        mutation(mutation::get(modules,           
            mu, weights.mueff,
            static_cast<double>(dim),
            2.0 // sigma
        )),
        selection(std::make_shared<selection::Strategy>(modules)),
        restart(restart::get(modules.restart_strategy, 
            static_cast<double>(dim), 
            static_cast<double>(lambda), 
            static_cast<double>(mu), 
            stats.budget)
        ),
        bounds(bounds::get(dim, modules.bound_correction))
    {
        // Ensure proper initialization of pop.s
        mutation->sample_sigma(pop);

        if (modules.mirrored == sampling::Mirror::PAIRWISE and lambda % 2 != 0)
            lambda++;

        if (mu > lambda)
            mu = lambda / 2;

    }

    Parameters::Parameters(const size_t dim) : Parameters(dim, {}) {}
        
    void Parameters::perform_restart(const std::optional<double>& sigma) {
        weights = Weights(dim, mu, lambda, modules);
        sampler = sampling::get(dim, modules, lambda);

        pop = Population(dim, lambda); 
        old_pop = Population(dim, lambda);

        mutation = mutation::get(modules, mu, weights.mueff,
            static_cast<double>(dim), 
            sigma.value_or(mutation->sigma0)
        );
        
        mutation->sample_sigma(pop);
        
        dynamic.B = Matrix::Identity(dim, dim);
        dynamic.C = Matrix::Identity(dim, dim);
        dynamic.inv_root_C = Matrix::Identity(dim, dim);
        dynamic.d.setOnes();
        dynamic.m.setZero(); // = Vector::Random(dim) * 5;
        dynamic.m_old.setZero();
        dynamic.dm.setZero();
        dynamic.pc.setZero();
        dynamic.ps.setZero();
    }

    bool Parameters::invalid_state() const {
        const bool sigma_out_of_bounds = 1e-8 > mutation->sigma or mutation->sigma > 1e4;

        if(sigma_out_of_bounds) {
            std::cout << "sigma " << mutation->sigma << " restarting\n";
        }
        return sigma_out_of_bounds;
    }

    void Parameters::adapt()
    {

        dynamic.adapt_evolution_paths(weights, mutation, stats, lambda);
        mutation->adapt(weights, dynamic, pop, old_pop, stats, lambda);

        dynamic.adapt_covariance_matrix(weights, modules, pop, mu);
        
        if (!dynamic.perform_eigendecomposition(stats) or invalid_state())
            perform_restart();
        
        old_pop = pop;
        restart->evaluate(*this);
        
        stats.t++;
    }   
}

std::ostream &operator<<(std::ostream &os, const parameters::Stats &s)
{
    return os
           << "Stats"
           << " t=" << s.t
           << " evals=" << s.evaluations
           << " xopt=("
           << s.xopt.transpose()
           << ") fopt="
           << s.fopt;
}
