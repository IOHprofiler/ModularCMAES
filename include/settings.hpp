#pragma once

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
/**
 * The settings of a run. The constructor fills in the defaults, adjusts the modules so that they
 * fit together and checks every value (std::invalid_argument).
 *
 * After construction, the set_* methods change one field (the Python setters call them, #136).
 * A setter checks only its own field (its length, NaN, sign or range) and recomputes the values
 * derived from it, so the fields may be set in any order and through invalid intermediate states.
 * The checks between fields (require_consistent: lb <= ub, the integer variables, mu0 <= lambda0,
 * the budget, ...) run in the constructor and again when an optimizer is created from the
 * settings (Parameters calls require_valid). Assigning a field directly skips the setter, not
 * require_valid; it does not recompute the derived values.
 */
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

    /**
     * The initial covariance matrix C of the search distribution (warm starting, #89): the first
     * generation samples N(m, sigma0^2 C0). None (the default): C = I. Set with set_C0, which
     * checks it. Each matrix adaptation starts its sampling factor from it
     * (matrix_adaptation::Adaptation::start_from); SEPARABLE takes its diagonal, NONE rejects it.
     * A restart starts from it again (Parameters::finish_restart).
     */
    std::optional<Matrix> C0 = std::nullopt;

    /**
     * Set C0, or remove it (nullopt). std::invalid_argument unless C0 is a (dim, dim) finite
     * matrix, symmetric up to rounding (||C0 - C0^T||_F <= 1e-10 ||C0||_F; its symmetric part
     * (C0 + C0^T) / 2 is kept) and positive definite, or when matrix_adaptation is NONE.
     */
    void set_C0(std::optional<Matrix> c0)
    {
        if (c0)
        {
            require(modules.matrix_adaptation != MatrixAdaptationType::NONE,
                    "C0 cannot be used with matrix_adaptation NONE, which keeps C = I");
            const auto rows = static_cast<size_t>(c0->rows());
            const auto cols = static_cast<size_t>(c0->cols());
            require(rows == dim && cols == dim,
                    "C0 must be a (dim, dim) = (" + std::to_string(dim) + ", " +
                        std::to_string(dim) + ") matrix, got (" + std::to_string(c0->rows()) +
                        ", " + std::to_string(c0->cols()) + ")");
            require(c0->allFinite(), "C0 must be finite");
            require((*c0 - c0->transpose()).norm() <= 1e-10 * c0->norm(), "C0 must be symmetric");
            Matrix symmetric = (*c0 + c0->transpose()) * 0.5;
            const Eigen::SelfAdjointEigenSolver<Matrix> eigen(symmetric, Eigen::EigenvaluesOnly);
            require(eigen.info() == Eigen::Success, "C0: the eigenvalue solver failed");
            std::ostringstream smallest;
            smallest << eigen.eigenvalues().minCoeff();
            require(eigen.eigenvalues().minCoeff() > 0.0 &&
                        Eigen::LLT<Matrix>(symmetric).info() == Eigen::Success,
                    "C0 must be positive definite, its smallest eigenvalue is " + smallest.str());
            c0 = std::move(symmetric);
        }
        C0 = std::move(c0);
    }

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
        // modules and lambda0 are initialized by now. At least 1: lambda0 = 1 with PAIRWISE
        // becomes lambda0 = 2 below and gave mu0 = 0, NaN weights (#136). With SA or CMSA mu0 is
        // computed again below, after PAIRWISE has made lambda0 even.
        mu0(mu.value_or(default_mu0())),
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
        // crash or a silent misbehaviour later (#113): first each field, as its setter does
        require_lambda0(lambda0);
        if (mu)
            require_mu0(*mu);
        require_sigma0(sigma0);
        require_budget(this->budget);
        require_lb(this->lb);
        require_ub(this->ub);
        require_x0(this->x0);
        mu0_given = mu.has_value();

        if (modules.mirrored == Mirror::PAIRWISE and lambda0 % 2 != 0)
            lambda0++;

        if (modules.ssa == StepSizeAdaptation::SA ||
            modules.matrix_adaptation == MatrixAdaptationType::CMSA)
        {
            mu0 = mu.value_or(default_mu0());
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

        // Then the checks between fields, on the adjusted values
        require_consistent();

        update_bounds();
    }

    // The setters (#136): each checks its own field only, and recomputes what is derived from it;
    // nothing changes when the check fails. The checks between fields wait for require_valid.

    //! lambda0 (at least 1). mu0 follows it (default_mu0) until mu0 is given, to the constructor
    //! or to set_mu0. The constructor's other rules for lambda0 (PAIRWISE makes it even, 1 makes a
    //! (1+1)-ES) are not applied: require_consistent rejects what they would have changed.
    void set_lambda0(const size_t value)
    {
        require_lambda0(value);
        lambda0 = value;
        if (!mu0_given)
            mu0 = default_mu0();
    }

    //! mu0 (at least 1); from now on it no longer follows lambda0
    void set_mu0(const size_t value)
    {
        require_mu0(value);
        mu0 = value;
        mu0_given = true;
    }

    void set_budget(const size_t value)
    {
        require_budget(value);
        budget = value;
    }

    void set_sigma0(const Float value)
    {
        require_sigma0(value);
        sigma0 = value;
    }

    void set_x0(const std::optional<Vector> &value)
    {
        require_x0(value);
        x0 = value;
    }

    //! lb; recomputes db, center, diameter, volume and has_bounds
    void set_lb(const Vector &value)
    {
        require_lb(value);
        lb = value;
        update_bounds();
    }

    //! ub; recomputes db, center, diameter, volume and has_bounds
    void set_ub(const Vector &value)
    {
        require_ub(value);
        ub = value;
        update_bounds();
    }

    // The checks of one field, shared by the constructor, the setters and require_valid;
    // std::invalid_argument naming the field otherwise

    static void require_lambda0(const size_t value)
    {
        require(value >= 1, "lambda0 must be at least 1");
    }

    static void require_mu0(const size_t value) { require(value >= 1, "mu0 must be at least 1"); }

    static void require_sigma0(const Float value)
    {
        require(std::isfinite(value) && value > 0,
                "sigma0 must be positive and finite, got " + str(value));
    }

    //! At least 2, the smallest budget of any settings: the (1+1)-ES's first point and the final
    //! center of mass. require_consistent checks it against the first generation.
    static void require_budget(const size_t value)
    {
        require(value >= 2,
                "budget must be at least 2 (a first generation and the evaluation of the final "
                "center of mass), got " +
                    std::to_string(value));
    }

    //! The length, and in each coordinate neither NaN nor +inf, which leaves no value (#133)
    void require_lb(const Vector &value) const
    {
        require_length("lb", value);
        for (Eigen::Index i = 0; i < value.size(); ++i)
            if (std::isnan(value(i)) || value(i) == std::numeric_limits<Float>::infinity())
                fail("lb[" + std::to_string(i) + "] = " + str(value(i)) +
                     " is not allowed: a lower bound is neither NaN nor +inf");
    }

    //! The length, and in each coordinate neither NaN nor -inf, which leaves no value (#133)
    void require_ub(const Vector &value) const
    {
        require_length("ub", value);
        for (Eigen::Index i = 0; i < value.size(); ++i)
            if (std::isnan(value(i)) || value(i) == -std::numeric_limits<Float>::infinity())
                fail("ub[" + std::to_string(i) + "] = " + str(value(i)) +
                     " is not allowed: an upper bound is neither NaN nor -inf");
    }

    void require_x0(const std::optional<Vector> &value) const
    {
        if (!value)
            return;
        require_length("x0", *value);
        require(value->allFinite(), "x0 must be finite");
    }

    //! The length of a vector of the dimension (lb, ub, x0)
    void require_length(const char *name, const Vector &v) const
    {
        require(static_cast<size_t>(v.size()) == dim,
                std::string(name) + " must have length dim = " + std::to_string(dim) + ", got " +
                    std::to_string(v.size()));
    }

    /**
     * No coordinate of the box is empty (#133); std::invalid_argument otherwise. A coordinate with
     * lb_i == ub_i, finite, is a fixed variable and is accepted. Empty: lb_i > ub_i, a NaN bound,
     * lb_i = +inf, ub_i = -inf (this includes lb_i = ub_i = +inf or -inf, whose width is NaN).
     * The messages are built only for a coordinate that fails.
     */
    static void require_nonempty_box(const Vector &lb, const Vector &ub)
    {
        for (Eigen::Index i = 0; i < lb.size(); ++i)
        {
            if (!(lb(i) <= ub(i)))
                fail("lb must not exceed ub (and neither may be NaN): lb[" + std::to_string(i) +
                     "] = " + str(lb(i)) + ", ub[" + std::to_string(i) + "] = " + str(ub(i)));
            if (lb(i) == std::numeric_limits<Float>::infinity() ||
                ub(i) == -std::numeric_limits<Float>::infinity())
                fail("the box is empty in coordinate " + std::to_string(i) +
                     ": lb = " + str(lb(i)) + ", ub = " + str(ub(i)) +
                     " (a bound of +inf for lb, or -inf for ub, leaves no value)");
        }
    }

    /**
     * The checks between fields (#133, #136), on fields that passed their own checks: the box
     * (require_nonempty_box), the integer variables in it, mu0 <= lambda0, the budget against the
     * first generation, and lambda0 against one_plus_one and PAIRWISE. Each message names the
     * fields and their conflicting values. The constructor makes mu0, lambda0, the modules and
     * one_plus_one consistent; only setters (and changes of the modules in place) can break the
     * last four checks.
     */
    void require_consistent() const
    {
        require_nonempty_box(lb, ub);
        // Every integer variable has an integer value in [lb, ub]: [ceil(lb), floor(ub)] is not
        // empty (#79 F21). An index out of range is reported when the Parameters are created
        // (mapping::CoordinateMapping).
        for (const auto idx : integer_variables)
            if (idx >= 0 && static_cast<size_t>(idx) < dim &&
                !(std::ceil(lb(idx)) <= std::floor(ub(idx))))
            {
                const auto i = std::to_string(idx);
                fail("integer variable " + i + " has no integer value in [lb[" + i + "], ub[" + i +
                     "]] = [" + str(lb(idx)) + ", " + str(ub(idx)) + "]");
            }
        require(mu0 <= lambda0,
                "mu0 = " + std::to_string(mu0) + " exceeds lambda0 = " + std::to_string(lambda0));
        // The first generation must fit: lambda0 and the points that every generation evaluates
        // besides the population, TPA's two (Parameters::reserved_points), and the evaluation of
        // the run's center of mass that ends every run (#111)
        const bool tpa = modules.ssa == StepSizeAdaptation::TPA;
        const size_t needed = lambda0 + (tpa ? 2 : 0) + 1;
        require(budget >= needed,
                "budget " + std::to_string(budget) +
                    " is smaller than the first generation and the evaluation of the final "
                    "center of mass: lambda0 = " +
                    std::to_string(lambda0) + (tpa ? ", TPA's 2 points" : "") + " and 1, " +
                    std::to_string(needed) + " evaluations");
        require(!one_plus_one || lambda0 == 1,
                "lambda0 = " + std::to_string(lambda0) +
                    ", but these settings were created with lambda0 = 1, a (1+1)-ES "
                    "(one_plus_one, with the modules adjusted for it): create new Settings for "
                    "another lambda0");
        require(one_plus_one || lambda0 != 1,
                "lambda0 = 1 makes a (1+1)-ES (one_plus_one), for which the constructor adjusts "
                "the modules: give lambda0 = 1 to the constructor");
        require(modules.mirrored != Mirror::PAIRWISE || lambda0 % 2 == 0,
                "lambda0 = " + std::to_string(lambda0) +
                    " is odd, but mirrored PAIRWISE needs an even lambda0 (the constructor "
                    "increases an odd lambda0 by one)");
    }

    //! Every check: each field (as its setter does) and require_consistent. An optimizer runs it
    //! when it is created from the settings (Parameters).
    void require_valid() const
    {
        require_lambda0(lambda0);
        require_mu0(mu0);
        require_sigma0(sigma0);
        require_budget(budget);
        require_lb(lb);
        require_ub(ub);
        require_x0(x0);
        require_consistent();
    }

    //! The default mu0 of the current lambda0 and modules, the constructor's rule: lambda0 / 2, or
    //! lambda0 / 4 with ssa SA or matrix_adaptation CMSA; at least 1
    [[nodiscard]] size_t default_mu0() const
    {
        const bool quarter = modules.ssa == StepSizeAdaptation::SA ||
                             modules.matrix_adaptation == MatrixAdaptationType::CMSA;
        return std::max<size_t>(1, quarter ? lambda0 / 4 : lambda0 / 2);
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
    //! Whether mu0 was given (to the constructor or set_mu0); until then it follows lambda0
    bool mu0_given = false;

    static size_t checked_dim(const size_t dim)
    {
        require(dim >= 1, "dim must be at least 1");
        return dim;
    }

    [[noreturn]] static void fail(const std::string &message)
    {
        throw std::invalid_argument("Settings: " + message);
    }

    static void require(const bool condition, const std::string &message)
    {
        if (!condition)
            fail(message);
    }

    //! A number for a message: 0.6, not std::to_string's 0.600000
    static std::string str(const Float value)
    {
        std::ostringstream ss;
        ss << value;
        return ss.str();
    }
};

} // namespace parameters