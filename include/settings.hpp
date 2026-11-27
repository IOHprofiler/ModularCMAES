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
 * The checks between fields (require_consistent: lb <= ub, c1 + cmu <= 1, the integer variables, mu0 <= lambda0,
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
        // crash or a silent misbehaviour later (#113): first each field, as its setter does
        require_lambda0(lambda0);
        if (mu)
            require_mu0(*mu);
        require_sigma0(sigma0);
        require_budget(this->budget);
        require_lb(this->lb);
        require_ub(this->ub);
        require_target(this->target);
        require_x0(this->x0);
        require_max_generations(this->max_generations);
        require_rates(this->cs, this->cc, this->cmu, this->c1, this->damps, this->acov);
        mu0_given = mu.has_value();

        // The (1+1)-ES has no mirrored pairs to select from: PAIRWISE becomes NONE (the
        // maintainer's decision on #136, F17 of #75). It made lambda0 = 1 even, 2, with
        // mu0 = 1 / 2 = 0 and NaN weights. Before the rule below, which would make lambda0 even;
        // the other adjustments of the (1+1)-ES are at the end. MIRRORED stays, without effect: the
        // pairs are formed within a generation (#75).
        if (lambda0 == 1 and modules.mirrored == Mirror::PAIRWISE)
            modules.mirrored = Mirror::NONE;

        // The default mu0 follows the even lambda0 (#75: it was half the odd one, 3 of 8 at d = 3);
        // a given mu0 is kept, and checked against lambda0 (require_consistent)
        if (modules.mirrored == Mirror::PAIRWISE and lambda0 % 2 != 0)
        {
            lambda0++;
            mu0 = mu.value_or(default_mu0());
        }

        if (modules.ssa == StepSizeAdaptation::SA ||
            modules.matrix_adaptation == MatrixAdaptationType::CMSA)
        {
            mu0 = mu.value_or(default_mu0());
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

        // CMSA has no active variant (Beyer & Sendhoff 2008): its update with negative weights, not
        // rescaled and not in the decay, made C indefinite, with an UnableToAdapt restart every few
        // generations (#73, F6). With UTILITY, CMSA uses the positive utilities.
        if (modules.matrix_adaptation == MatrixAdaptationType::NONE ||
            modules.matrix_adaptation == MatrixAdaptationType::CMSA)
        {
            modules.active = false;
        }

        // The (1+1)-ES (PAIRWISE is NONE already, above)
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

        // Only CSA needs C^-1/2, from the eigendecomposition. After the (1+1)-ES's adjustments, so
        // that the step-size rule that runs decides (#131: with lambda0 = 1 every ssa runs SR, but
        // the default CSA kept the eigendecomposition)
        if (modules.ssa != StepSizeAdaptation::CSA and
            modules.matrix_adaptation == MatrixAdaptationType::COVARIANCE and
            not always_compute_eigv)
        {
            modules.matrix_adaptation = MatrixAdaptationType::COVARIANCE_NO_EIGV;
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

    //! target: None (no target) or a number, not NaN
    void set_target(const std::optional<Float> &value)
    {
        require_target(value);
        target = value;
    }

    //! max_generations: None (no limit) or at least 1 (#147)
    void set_max_generations(const std::optional<size_t> &value)
    {
        require_max_generations(value);
        max_generations = value;
    }

    //! The learning rates and the damping: None (the module's default) or a number in the range
    //! of the field (#147; require_cs ... require_acov). The sum c1 + cmu waits for
    //! require_consistent.
    void set_cs(const std::optional<Float> &value)
    {
        require_cs(value);
        cs = value;
    }
    void set_cc(const std::optional<Float> &value)
    {
        require_cc(value);
        cc = value;
    }
    void set_cmu(const std::optional<Float> &value)
    {
        require_cmu(value);
        cmu = value;
    }
    void set_c1(const std::optional<Float> &value)
    {
        require_c1(value);
        c1 = value;
    }
    void set_damps(const std::optional<Float> &value)
    {
        require_damps(value);
        damps = value;
    }
    void set_acov(const std::optional<Float> &value)
    {
        require_acov(value);
        acov = value;
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
        update_bounds(); // center depends on sigma0 in coordinates with one infinite bound
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

    //! NaN never stops a run (a comparison with NaN is false): rejected (the maintainer's
    //! decision on #136). None means no target; +-inf are allowed.
    static void require_target(const std::optional<Float> &value)
    {
        require(!value || !std::isnan(*value), "target must not be NaN (None: no target)");
    }

    //! At least 1: 0 stops the run before its first generation (#147)
    static void require_max_generations(const std::optional<size_t> &value)
    {
        require(!value || *value >= 1,
                "max_generations must be at least 1 (None: no limit), got " +
                    std::to_string(value.value_or(0)));
    }

    // The learning rates (#147). None keeps the module's default. The bounds are Hansen, "The CMA
    // Evolution Strategy: A Tutorial" (2016, arXiv:1604.00772v2), list of symbols: c_c <= 1,
    // c_sigma < 1, c_1 <= 1 - c_mu, c_mu <= 1 - c_1; Eq. (15): 0 < c_mu <= 1 (c_mu = 0: no
    // learning); Table 1: alpha_cov = 2. The tutorial gives d_sigma only as ~ 1; d_sigma > 0 is
    // ours: it divides the step-size update, and a negative one makes sigma grow.
    // cs = 1 is allowed although the tutorial writes c_sigma < 1: cs (2 - cs) = 1 and the run
    // works. cs > 1 makes cs (2 - cs) smaller than 1 and, above 2, negative: NaN or a diverging
    // sigma. c1 and cmu are at most 1 because the other is not negative; their sum is checked in
    // require_consistent.

    //! cs in (0, 1]
    static void require_cs(const std::optional<Float> &value)
    {
        require(!value || (*value > 0 && *value <= 1),
                "cs must be in (0, 1] (None: the default), got " + str(value.value_or(0)));
    }

    //! cc in (0, 1]
    static void require_cc(const std::optional<Float> &value)
    {
        require(!value || (*value > 0 && *value <= 1),
                "cc must be in (0, 1] (None: the default), got " + str(value.value_or(0)));
    }

    //! cmu in [0, 1]; 0 switches the rank-mu update off
    static void require_cmu(const std::optional<Float> &value)
    {
        require(!value || (*value >= 0 && *value <= 1),
                "cmu must be in [0, 1] (None: the default), got " + str(value.value_or(0)));
    }

    //! c1 in [0, 1]; 0 switches the rank-one update off
    static void require_c1(const std::optional<Float> &value)
    {
        require(!value || (*value >= 0 && *value <= 1),
                "c1 must be in [0, 1] (None: the default), got " + str(value.value_or(0)));
    }

    //! damps > 0 (not NaN; +inf switches the step-size update off)
    static void require_damps(const std::optional<Float> &value)
    {
        require(!value || *value > 0,
                "damps must be positive (None: the default), got " + str(value.value_or(0)));
    }

    //! acov > 0 and finite: the factor alpha_cov of c1 and cmu (Hansen 2016, (57), (58))
    static void require_acov(const std::optional<Float> &value)
    {
        require(!value || (*value > 0 && std::isfinite(*value)),
                "acov must be positive and finite (None: the default 2), got " +
                    str(value.value_or(0)));
    }

    static void require_rates(const std::optional<Float> &cs,
                              const std::optional<Float> &cc,
                              const std::optional<Float> &cmu,
                              const std::optional<Float> &c1,
                              const std::optional<Float> &damps,
                              const std::optional<Float> &acov)
    {
        require_cs(cs);
        require_cc(cc);
        require_cmu(cmu);
        require_c1(c1);
        require_damps(damps);
        require_acov(acov);
    }

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
     * fields and their conflicting values. mu0 > lambda0 is rejected, not adjusted (the
     * maintainer's decision on #136; a (1+1)-ES still sets mu0 = 1). The constructor makes lambda0,
     * the modules and one_plus_one consistent; only setters (and changes of the modules in place)
     * can break the checks of lambda0 against one_plus_one and PAIRWISE.
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
        // C stays positive definite only if c1 + cmu <= 1 (Hansen 2016, c_1 <= 1 - c_mu). Checked
        // when both are given: a default is computed from mu_eff (Weights), which is not known
        // here, and changes with the population size at a restart.
        require(!(c1 && cmu) || *c1 + *cmu <= 1,
                "c1 + cmu must be at most 1 to keep C positive definite, got c1 = " +
                    str(*c1) + ", cmu = " + str(*cmu));
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

    /**
     * Whether x lies in the domain (#168): inside [lb, ub] when a bound correction is used. With
     * bound_correction NONE the objective is evaluated outside the box, which does not constrain
     * the points: every point is in the domain. Only a point in the domain becomes global_best.
     */
    [[nodiscard]] bool in_domain(Eigen::Ref<const Vector> x) const
    {
        if (modules.bound_correction == CorrectionMethod::NONE)
            return true;
        return !((x.array() < lb.array()) || (x.array() > ub.array())).any();
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
        require_target(target);
        require_x0(x0);
        require_max_generations(max_generations);
        require_rates(cs, cc, cmu, c1, damps, acov);
        require_consistent();
    }

    //! The default mu0 of the current lambda0 and modules: default_mu(lambda0)
    [[nodiscard]] size_t default_mu0() const { return default_mu(lambda0); }

    //! The default number of parents of a population of lambda with the current modules, the
    //! constructor's rule: lambda / 2, or lambda / 4 with ssa SA or matrix_adaptation CMSA; at
    //! least 1
    [[nodiscard]] size_t default_mu(const size_t lambda) const
    {
        const bool quarter = modules.ssa == StepSizeAdaptation::SA ||
                             modules.matrix_adaptation == MatrixAdaptationType::CMSA;
        return std::max<size_t>(1, quarter ? lambda / 4 : lambda / 2);
    }

    /**
     * The number of parents of a run whose population a restart set to lambda (IPOP, BIPOP, and
     * the cap of the first generation at the budget; #150): default_mu(lambda), as pycma's fmin,
     * whose every run takes the default CMA_mu, popsize // 2 (Hansen 2009: mu = floor(lambda / 2)).
     * When mu0 was given (to the constructor or set_mu0), its ratio to lambda0 instead:
     * mu0 lambda / lambda0, rounded half up, between 1 and lambda (the maintainer's decision on
     * #150; pycma keeps a given CMA_mu as the same number in every run).
     */
    [[nodiscard]] size_t mu_for(const size_t lambda) const
    {
        if (!mu0_given)
            return std::min(default_mu(lambda), lambda);
        const size_t mu = (2 * mu0 * lambda + lambda0) / (2 * lambda0);
        return std::clamp<size_t>(mu, 1, lambda);
    }

    //! Recompute the values derived from lb, ub and sigma0 (after a change of one of them).
    //! center, per coordinate (#134): two finite bounds, their midpoint; two infinite bounds, 0;
    //! one infinite bound, 0 if the half-line holds it, else the finite bound moved inward by
    //! sigma0 (so the first mean lies in the box).
    void update_bounds()
    {
        db = ub - lb;
        diameter = db.norm();
        volume = db.prod();
        has_bounds = true;
        center.resize(dim);
        for (Eigen::Index i = 0; i < center.size(); ++i)
        {
            const bool lo = std::isfinite(lb(i)), hi = std::isfinite(ub(i));
            if (lo && hi)
                center(i) = lb(i) + db(i) * 0.5;
            else if (lo)
                center(i) = lb(i) <= 0 ? 0.0 : lb(i) + sigma0;
            else if (hi)
                center(i) = ub(i) >= 0 ? 0.0 : ub(i) - sigma0;
            else
                center(i) = 0.0;
        }
        if (!std::isfinite(diameter))
        {
            has_bounds = false;
            diameter = 10;
            volume = pow(diameter, dim);
        }
    }

  private:
    friend struct serialization::Access;
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