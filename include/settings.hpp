#pragma once

#include <stdexcept>
#include <string>

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
struct Settings
{
    size_t dim;
    Modules modules;

    std::optional<Float> target;
    std::optional<size_t> max_generations;
    size_t budget;

    Float sigma0;
    size_t lambda0;
    size_t mu0;

    std::optional<Vector> x0;

    Vector lb;
    Vector ub;
    Indices integer_variables;
    Vector db;
    Vector center;
    Float diameter;
    Float volume;
    bool has_bounds;

    std::optional<Float> cs;
    std::optional<Float> cc;
    std::optional<Float> cmu;
    std::optional<Float> c1;
    std::optional<Float> damps;
    std::optional<Float> acov;

    bool verbose;
    bool one_plus_one;

    //! The seed of the optimizer's own random generator. Without one, the optimizer starts from
    //! the global generator (rng::set_seed) when it is created (rng::fork).
    std::optional<int> seed = std::nullopt;

    //! Opt-in: draw the samples with xoshiro256++ and a ziggurat (sampling::FastGaussian) instead
    //! of mt19937 and the inverse normal CDF, for the default UNIFORM + GAUSSIAN sampler. Faster,
    //! but a different random stream than the default.
    bool fast_sampling = false;

    Settings(size_t dim,
             std::optional<Modules> mod = std::nullopt,
             std::optional<Float> target = std::nullopt,
             std::optional<size_t> max_generations = std::nullopt,
             std::optional<size_t> budget = std::nullopt,
             std::optional<Float> sigma = std::nullopt,
             std::optional<size_t> mu = std::nullopt,
             std::optional<size_t> lambda = std::nullopt,
             std::optional<Vector> x0 = std::nullopt,
             std::optional<Vector> lb = std::nullopt,
             std::optional<Vector> ub = std::nullopt,
             std::optional<Indices> integer_variables = std::nullopt,
             std::optional<Float> cs = std::nullopt,
             std::optional<Float> cc = std::nullopt,
             std::optional<Float> cmu = std::nullopt,
             std::optional<Float> c1 = std::nullopt,
             std::optional<Float> damps = std::nullopt,
             std::optional<Float> acov = std::nullopt,
             bool verbose = false,
             bool always_compute_eigv = false) :
        dim(checked_dim(dim)),
        modules(mod.value_or(Modules())),
        target(target),
        max_generations(max_generations),
        budget(budget.value_or(dim * 1e4)),
        sigma0(sigma.value_or(2.0)),
        lambda0(lambda.value_or(4 + std::floor(3 * std::log(dim)))),
        mu0(mu.value_or(lambda0 / 2)),
        x0(x0),
        lb(lb.value_or(Vector::Ones(dim) * -5)),
        ub(ub.value_or(Vector::Ones(dim) * 5)),
        integer_variables(integer_variables.value_or(Indices{})),
        cs(cs),
        cc(cc),
        cmu(cmu),
        c1(c1),
        damps(damps),
        acov(acov),
        verbose(verbose),
        one_plus_one(false)
    {
        // Invalid values raise std::invalid_argument (ValueError in Python) here, instead of a
        // crash or a silent misbehaviour later (#113)
        require(lambda0 >= 1, "lambda0 must be at least 1");
        require(!mu || *mu >= 1, "mu0 must be at least 1");
        require(std::isfinite(sigma0) && sigma0 > 0,
                "sigma0 must be positive and finite, got " + std::to_string(sigma0));
        require_length("lb", this->lb);
        require_length("ub", this->ub);
        require((this->lb.array() <= this->ub.array()).all(),
                "lb must not exceed ub (and neither may be NaN)");
        if (this->x0)
        {
            require_length("x0", *this->x0);
            require(this->x0->allFinite(), "x0 must be finite");
        }

        if (modules.mirrored == Mirror::PAIRWISE and lambda0 % 2 != 0)
            lambda0++;

        if (modules.ssa == StepSizeAdaptation::SA ||
            modules.matrix_adaptation == MatrixAdaptationType::CMSA)
        {
            mu0 = mu.value_or(std::max<size_t>(1, lambda0 / 4));
        }

        if (mu0 > lambda0)
        {
            mu0 = lambda0 / 2;
        }

        if (modules.ssa != StepSizeAdaptation::CSA and
            modules.matrix_adaptation == MatrixAdaptationType::COVARIANCE and
            not always_compute_eigv)
        {
            modules.matrix_adaptation = MatrixAdaptationType::COVARIANCE_NO_EIGV;
        }

        // xNES's active form is its utilities (summing to zero) over all lambda points; CMA-ES's
        // negative weights, scaled for CMA-ES, made the natural gradient shrink sigma even on a
        // linear function (dev/overhaul-2.0.md F2). So NATURAL_GRADIENT with active uses the utilities.
        if (modules.matrix_adaptation == MatrixAdaptationType::NATURAL_GRADIENT && modules.active &&
            modules.weights == RecombinationWeights::DEFAULT)
        {
            modules.weights = RecombinationWeights::UTILITY;
        }

        // The utilities of NES/xNES are negative for the worse half: every matrix update uses all
        // of them, as the active update does
        if (modules.weights == RecombinationWeights::UTILITY)
        {
            modules.active = true;
        }

        if (modules.matrix_adaptation == MatrixAdaptationType::NONE)
        {
            modules.active = false;
        }

        if (lambda0 == 1)
        {
            mu0 = 1;
            one_plus_one = true;
            modules.elitist = true;
            modules.active = false;
            modules.sequential_selection = false;
            modules.weights = RecombinationWeights::EQUAL;
            modules.ssa = StepSizeAdaptation::SR;

            if (modules.restart_strategy == RestartStrategyType::BIPOP ||
                modules.restart_strategy == RestartStrategyType::IPOP)
                modules.restart_strategy = RestartStrategyType::RESTART;
        }

        // The first generation must fit: lambda0 and the points that every generation evaluates
        // besides the population, TPA's two (Parameters::reserved_points), and the evaluation of
        // the run's center of mass that ends every run (#111)
        const size_t first_generation = lambda0 + (modules.ssa == StepSizeAdaptation::TPA ? 2 : 0);
        require(this->budget >= first_generation + 1,
                "budget " + std::to_string(this->budget) +
                    " is smaller than the first generation and the evaluation of the final "
                    "center of mass (" +
                    std::to_string(first_generation + 1) + " evaluations)");

        update_bounds();
    }

    //! The length of a vector of the dimension (lb, ub, x0); std::invalid_argument otherwise
    void require_length(const char *name, const Vector &v) const
    {
        require(static_cast<size_t>(v.size()) == dim,
                std::string(name) + " must have length dim = " + std::to_string(dim) + ", got " +
                    std::to_string(v.size()));
    }

    //! Recompute the values derived from lb and ub (after a change of the bounds)
    void update_bounds()
    {
        db = ub - lb;
        center = lb + (db * 0.5);
        diameter = db.norm();
        volume = db.prod();
        has_bounds = true;
        if (!std::isfinite(diameter))
        {
            has_bounds = false;
            diameter = 10;
            volume = pow(diameter, dim);
            center.setConstant(0);
        }
    }

  private:
    static size_t checked_dim(const size_t dim)
    {
        require(dim >= 1, "dim must be at least 1");
        return dim;
    }

    static void require(const bool condition, const std::string &message)
    {
        if (!condition)
            throw std::invalid_argument("Settings: " + message);
    }
};

} // namespace parameters