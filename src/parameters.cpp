#include "parameters.hpp"

namespace parameters
{
    Parameters::Parameters(const Settings& settings) : 
        lambda(settings.lambda0),
        mu(settings.mu0),
        settings(settings),
        dynamic(settings.dim, settings.x0.value_or(Vector::Zero(settings.dim))),
        weights(settings.dim, settings.mu0, settings.lambda0, settings),
        pop(settings.dim, settings.lambda0),
        old_pop(settings.dim, settings.lambda0),        
        sampler(sampling::get(settings.dim, settings.modules, settings.lambda0)),
        mutation(mutation::get(settings.modules,           
            settings.mu0, weights.mueff,
            static_cast<double>(settings.dim),
            settings.sigma0,
            settings.cs
        )),
        selection(std::make_shared<selection::Strategy>(settings.modules)), 
        restart(restart::get(settings.modules.restart_strategy, 
            static_cast<double>(settings.dim), 
            static_cast<double>(settings.lambda0), 
            static_cast<double>(settings.mu0), 
            settings.budget)
        ),
        bounds(bounds::get(settings.modules.bound_correction, settings.lb, settings.ub))
    {
    } 

    Parameters::Parameters(const size_t dim) : Parameters(Settings(dim,  {})) {}
        
    void Parameters::perform_restart(const std::optional<double>& sigma) {
        weights = Weights(settings.dim, mu, lambda, settings);
        sampler = sampling::get(settings.dim, settings.modules, lambda);

        pop = Population(settings.dim, lambda); 
        old_pop = Population(settings.dim, lambda);

        mutation = mutation::get(settings.modules, mu, weights.mueff,
            static_cast<double>(settings.dim), 
            sigma.value_or(settings.sigma0),
            settings.cs
        );
        dynamic.B = Matrix::Identity(settings.dim, settings.dim);
        dynamic.C = Matrix::Identity(settings.dim, settings.dim);
        dynamic.inv_root_C = Matrix::Identity(settings.dim, settings.dim);
        dynamic.d.setOnes();
        dynamic.m = settings.x0.value_or(Vector::Zero(settings.dim));
        dynamic.m_old.setZero();
        dynamic.dm.setZero();
        dynamic.pc.setZero();
        dynamic.ps.setZero();
        restart->criteria = restart::RestartCriteria(settings.dim, lambda, stats.t);
    }

    bool Parameters::invalid_state() const {
        const bool sigma_out_of_bounds = 1e-16 > mutation->sigma or mutation->sigma > 1e4;

        if(sigma_out_of_bounds && settings.verbose) {
            std::cout << "sigma out of bounds: " << mutation->sigma << " restarting\n";
        }
        return sigma_out_of_bounds;
    }

    void Parameters::adapt()
    {

        dynamic.adapt_evolution_paths(weights, mutation, stats, lambda);
        mutation->adapt(weights, dynamic, pop, old_pop, stats, lambda);

        dynamic.adapt_covariance_matrix(weights, settings.modules, pop, mu);
        
        if (!dynamic.perform_eigendecomposition(settings) or invalid_state())
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
