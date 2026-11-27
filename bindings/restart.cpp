// Bindings of the restart criteria and the restart strategies (submodule restart).
#include "common.hpp"

#include <nanobind/stl/bind_vector.h>
#include <nanobind/trampoline.h>

namespace modcma_python
{
namespace
{
//! Python subclasses of restart.Criterion: update (required) and on_reset are called from C++
struct PyCriterion : restart::Criterion
{
    NB_TRAMPOLINE(restart::Criterion, 2);

    void update(const parameters::Parameters &p) override { NB_OVERRIDE_PURE(update, p); }

    void on_reset(const parameters::Parameters &p) override { NB_OVERRIDE(on_reset, p); }
};
} // namespace

void define_restart_criteria(nb::module_ &main)
{
    auto m = submodule(main, "restart");
    m.doc() =
        "Restart criteria and restart strategies.\n\n"
        "After every generation, the optimizer updates the criteria in `Parameters.criteria`; "
        "when one of them is met, the next generation starts a restart (or, with "
        "RestartStrategy.STOP, the run ends). The restart strategy (`Parameters.restart_strategy`, "
        "chosen by `Modules.restart_strategy`) sets the population size and the step size of "
        "the new run. The criteria can be changed and extended: remove items, change the "
        "tolerances (static, shared by all instances of a class), or subclass `Criterion` in "
        "Python.";
    using namespace restart;

    nb::class_<Criterion, PyCriterion>(
        m,
        "Criterion",
        "A restart criterion: a condition on the state of the optimizer, checked after every "
        "generation.\n\n"
        "Subclass it in Python to define your own: call `super().__init__(name)`, override "
        "`update` (required) to set `self.met`, and optionally `on_reset`. Then add an instance "
        "to `Parameters.criteria.items`:\n\n"
        "    class MyCriterion(restart.Criterion):\n"
        "        def __init__(self):\n"
        "            super().__init__(\"MyCriterion\")\n\n"
        "        def update(self, parameters):\n"
        "            self.met = parameters.stats.t - self.last_restart > 100\n\n"
        "    cma.p.criteria.items.append(MyCriterion())")
        .def(nb::init<std::string>(),
             nb::arg("name"),
             "A criterion called name (shown by `Criteria.reason`), not met.")
        .def("on_reset",
             &Criterion::on_reset,
             nb::arg("parameters"),
             "Called by `reset`, at the start and at every restart; override it to reset the "
             "state of a criterion (the default does nothing).")
        .def("update",
             &Criterion::update,
             nb::arg("parameters"),
             "Called after every generation with the `Parameters` of the optimizer; sets `met`. "
             "Python subclasses must override it.")
        .def("reset",
             &Criterion::reset,
             nb::arg("parameters"),
             "Start over: sets `last_restart` to the current generation, `met` to False and "
             "calls `on_reset`.")
        .def_rw("met", &Criterion::met, "Whether the criterion is met (set by `update`).")
        .def_rw("name", &Criterion::name, "The name of the criterion.")
        .def_rw("last_restart",
                &Criterion::last_restart,
                "The generation (`Stats.t`) of the last reset, at the start or a restart.")
        .def(
            "__repr__",
            [](Criterion &self) {
                return "<" + self.name + " met: " + std::to_string(self.met) + ">";
            },
            "<name met: 0 or 1>.");

    nb::class_<ExceededMaxIter, Criterion>(
        m,
        "ExceededMaxIter",
        "Met when the run has lasted `max_iter` generations, so a run takes at most `max_iter` "
        "generations (at least one).\n\n"
        "As Hansen (2009), \"MaxIter ... the maximal number of iterations in each run\", and "
        "pycma's maxiter (countiter >= maxiter). `update` counts the generation it is called "
        "at: the optimizer updates the criteria at the end of a generation, before `Stats.t` "
        "counts it. Before 2.0 a run took max_iter + 2 generations (#139).")
        .def(nb::init<>(),
             "The criterion; `max_iter` is set by the first reset (or, added to a running "
             "optimizer, before its first update).")
        .def_rw("max_iter",
                &ExceededMaxIter::max_iter,
                "The most generations of one run, set at every reset to "
                "100 + 50 (d + 3)^2 / sqrt(lambda), or to the smaller `Strategy.max_iter` of the "
                "restart strategy (BIPOP's small runs).");

    nb::class_<NoImprovement, Criterion>(
        m,
        "NoImprovement",
        "Met when the best fitness of a generation has not changed over the last `n_bin` "
        "generations.\n\n"
        "Checked from n_bin generations after a restart on: met when the range (max - min) of "
        "the best fitness of the last n_bin generations is at most `NoImprovement.tolerance`.")
        .def(nb::init<>(),
             "The criterion; `n_bin` is set by the first reset (or, added to a running "
             "optimizer, before its first update).")
        .def_rw("n_bin",
                &NoImprovement::n_bin,
                "The number of generations compared, set at every reset to "
                "10 + ceil(30 d / lambda).")
        .def_rw("best_fitnesses",
                &NoImprovement::best_fitnesses,
                "The best fitness of every generation since the last reset.")
        .def_rw_static("tolerance",
                       &NoImprovement::tolerance,
                       "The largest range of the best fitness that counts as no improvement "
                       "(default 1e-20).");

    nb::class_<MaxSigma, Criterion>(
        m,
        "MaxSigma",
        "Met when sigma max(D) has grown by more than `MaxSigma.tolerance` since the start of "
        "the run: TolXUp of Hansen (2016, App. B.3), \"usually indicates a far too small initial "
        "sigma, or divergent behavior\".\n\n"
        "D are the axis lengths of the distribution, the square roots of the eigenvalues of C, "
        "from `Adaptation.metric()`, for COVARIANCE, SEPARABLE and NONE. The others, which "
        "would decompose their matrix every generation, use sqrt(max C_ii) "
        "(`Adaptation.coordinate_wise_variances`) instead of max(D): for them the test is "
        "pycma's tolfacupx, sigma sqrt(C_ii) > 1e3 sigma0 for some i, with 1e4. max C_ii is at "
        "most the largest eigenvalue, so it fires later than with the eigenvalues. Before 2.0 "
        "the criterion compared sigma with 1e4, whatever sigma0 and C.")
        .def(nb::init<>(), "The criterion; `start` is set by the first reset.")
        .def_rw("start",
                &MaxSigma::start,
                "sigma max(D) at the start of the run, set at every reset.")
        .def_rw_static("tolerance",
                       &MaxSigma::tolerance,
                       "The largest growth of sigma max(D) (default 1e4). Also the upper limit of "
                       "`constants.clip_sigma`, there as a step size.");

    nb::class_<MinSigma, Criterion>(
        m, "MinSigma", "Met when the step size sigma is smaller than `MinSigma.tolerance`.")
        .def(nb::init<>(), "The criterion.")
        .def_rw_static("tolerance",
                       &MinSigma::tolerance,
                       "The smallest step size (default 1e-20); also the lower limit of "
                       "`constants.clip_sigma`.");

    nb::class_<UnableToAdapt, Criterion>(
        m,
        "UnableToAdapt",
        "Met when the matrix adaptation failed (e.g. its eigendecomposition or Cholesky "
        "factorization) or sigma is not finite.\n\n"
        "Always among the criteria, also with RestartStrategy.NONE.")
        .def(nb::init<>(), "The criterion.");

    nb::class_<FlatFitness, Criterion>(
        m,
        "FlatFitness",
        "Met when the fitness landscape looks flat: too often, the best individual of a "
        "generation has the same fitness as the one at rank `flat_fitness_index`.\n\n"
        "Of the last d generations (from d generations after a restart on), counts those in "
        "which the two fitness values are equal; met when more than `max_flat_fitness` are. "
        "Only used when lambda > 1.")
        .def(nb::init<>(),
             "The criterion; its parameters are set by the first reset (or, added to a running "
             "optimizer, before its first update).")
        .def_rw("max_flat_fitness",
                &FlatFitness::max_flat_fitness,
                "The most flat generations among the last d, set at every reset to ceil(d / 3).")
        .def_rw("flat_fitness_index",
                &FlatFitness::flat_fitness_index,
                "The rank compared with the best, set at every reset to round(0.1 + lambda / 4).")
        .def_rw("flat_fitnesses",
                &FlatFitness::flat_fitnesses,
                "For each of the last d generations (generation t at t mod d), 1 when it was "
                "flat.");

    nb::class_<TolX, Criterion>(
        m,
        "TolX",
        "Met when the search distribution has become very small in every coordinate.\n\n"
        "Met when every entry of `tolx_vector` (sigma times the standard deviations sqrt(C_ii) "
        "and the absolute values of the evolution path p_c) is smaller than tolerance times the "
        "step size at the start of the run (Hansen 2016, App. B.3; pycma's tolx).\n\n"
        "For every matrix adaptation: C_ii from `Adaptation.coordinate_wise_variances`, and "
        "p_c where the adaptation has one (COVARIANCE, COVARIANCE_NO_EIGV, SEPARABLE, "
        "CHOLESKY).")
        .def(nb::init<>(), "The criterion.")
        .def_rw("tolx_vector",
                &TolX::tolx_vector,
                "The values compared: sigma sqrt(C_ii) (d values), then sigma |p_c| (d more, for "
                "the adaptations with p_c).")
        .def_rw_static("tolerance", &TolX::tolerance, "The tolerance (default 1e-11).");

    nb::class_<MaxDSigma, Criterion>(
        m,
        "MaxDSigma",
        "Met when sigma has grown too much relative to the step size at the start of the run: "
        "sigma / sigma_start > tolerance * max(D), with D the square roots of the eigenvalues "
        "of C (TolUpSigma of Hansen 2009; pycma's tolupsigma).\n\n"
        "sigma_start is the sigma0 of the run, as in Hansen (2009) and pycma (es.sigma0, which "
        "its fmin sets for each run): Settings.sigma0, or the step size of the restart, smaller "
        "in the small runs of BIPOP. Before 2.0 it was Settings.sigma0 in every run (#139).\n\n"
        "D from `Adaptation.metric()` for COVARIANCE, SEPARABLE and NONE. The others use "
        "sqrt(max C_ii) (`Adaptation.coordinate_wise_variances`) instead of max(D): free, and it "
        "follows the scale of C. max C_ii is at most the largest eigenvalue, so the criterion "
        "fires earlier than with the eigenvalues; no source does this.")
        .def(nb::init<>(), "The criterion; `sigma_start` is set by the first reset.")
        .def_rw("sigma_start",
                &MaxDSigma::sigma_start,
                "The step size at the start of the run, set at every reset.")
        .def_rw_static("tolerance", &MaxDSigma::tolerance, "The factor (default 1e20).");

    nb::class_<MinDSigma, Criterion>(
        m,
        "MinDSigma",
        "Met when sigma has shrunk too much relative to the step size at the start of the run: "
        "sigma / sigma_start < tolerance * min(D), with D the square roots of the eigenvalues of "
        "C. modcma's own counterpart of `MaxDSigma`; sigma_start as there (#139).\n\n"
        "D from `Adaptation.metric()` for COVARIANCE, SEPARABLE and NONE. The others use "
        "sqrt(min C_ii) (`Adaptation.coordinate_wise_variances`) instead of min(D). min C_ii is "
        "at least the smallest eigenvalue, so the criterion fires earlier than with the "
        "eigenvalues; no source does this.")
        .def(nb::init<>(), "The criterion; `sigma_start` is set by the first reset.")
        .def_rw("sigma_start",
                &MinDSigma::sigma_start,
                "The step size at the start of the run, set at every reset.")
        .def_rw_static("tolerance", &MinDSigma::tolerance, "The factor (default 1e-20).");

    nb::class_<ConditionC, Criterion>(
        m,
        "ConditionC",
        "Met when the condition number of the covariance matrix, max(D)^2 / min(D)^2, exceeds "
        "the tolerance (pycma's tolconditioncov).\n\n"
        "D from `Adaptation.metric()`. Created by default for COVARIANCE, SEPARABLE and NONE, "
        "which keep the metric; added to the others, it decomposes their matrix once per update "
        "of it, O(d^3).")
        .def(nb::init<>(), "The criterion.")
        .def_rw_static(
            "tolerance", &ConditionC::tolerance, "The largest condition number (default 1e14).");

    nb::class_<NoEffectAxis, Criterion>(
        m,
        "NoEffectAxis",
        "Met when a step of 0.1 standard deviations along a principal axis of C does not "
        "change the mean (pycma's noeffectaxis; Hansen 2016, App. B.3).\n\n"
        "In generation t, axis i = t mod d: met when m + 0.1 sigma D_i b_i == m in floating "
        "point (b_i the eigenvector, D_i the square root of its eigenvalue, from "
        "`Adaptation.metric()`). Created by default for COVARIANCE, SEPARABLE and NONE, which "
        "keep the metric; added to the others, it decomposes their matrix once per update of "
        "it, O(d^3).")
        .def(nb::init<>(), "The criterion.")
        .def_rw_static("tolerance",
                       &NoEffectAxis::tolerance,
                       "The step along the axis, in standard deviations (default 0.1, pycma). "
                       "Before 2.0, an absolute threshold of the step (10 times the machine "
                       "epsilon).");

    nb::class_<NoEffectCoord, Criterion>(
        m,
        "NoEffectCoord",
        "Met when a step of 0.2 standard deviations in some coordinate does not change the "
        "mean: m_i + 0.2 sigma sqrt(C_ii) == m_i in floating point (pycma's noeffectcoord; "
        "Hansen 2016, App. B.3).\n\n"
        "For every matrix adaptation: C_ii from `Adaptation.coordinate_wise_variances`.")
        .def(nb::init<>(), "The criterion.")
        .def_rw_static("tolerance",
                       &NoEffectCoord::tolerance,
                       "The step along the coordinate, in standard deviations (default 0.2, "
                       "pycma). Before 2.0, an absolute threshold of the step (10 times the "
                       "machine epsilon).");

    nb::class_<Stagnation, Criterion>(
        m,
        "Stagnation",
        "Met when neither the best nor the median fitness of the generations improves any "
        "more.\n\n"
        "From `n_stagnation` generations after a restart on, the history of each is split at "
        "tolerance (0.3) of its length; met when the median of the later part is not lower than "
        "the median of the earlier part, for both the best and the median fitness.")
        .def(nb::init<>(),
             "The criterion; `n_stagnation` is set by the first reset (or, added to a running "
             "optimizer, before its first update).")
        .def_rw("n_stagnation",
                &Stagnation::n_stagnation,
                "The generations before the criterion is checked, set at every reset to "
                "100 + 100 d^1.5 / lambda.")
        // lists, as before; assigning one rebuilds the running medians
        .def_prop_rw(
            "median_fitnesses",
            [](const Stagnation &s) { return s.median_fitnesses.values(); },
            [](Stagnation &s, const std::vector<Float> &v) { s.median_fitnesses.assign(v); },
            "The median fitness of every generation since the last reset (a copy; assign a "
            "list to replace it).")
        .def_prop_rw(
            "best_fitnesses",
            [](const Stagnation &s) { return s.best_fitnesses.values(); },
            [](Stagnation &s, const std::vector<Float> &v) { s.best_fitnesses.assign(v); },
            "The best fitness of every generation since the last reset (a copy; assign a list "
            "to replace it).")
        .def_rw_static("tolerance",
                       &Stagnation::tolerance,
                       "The fraction of the history that forms the earlier part (default 0.3).");

    nb::class_<TooMuchRepelling, Criterion>(
        m,
        "TooMuchRepelling",
        "Met when repelling rejected too many samples: on average at least tolerance "
        "rejections per individual in a generation.\n\n"
        "Used when `Modules.repelling_type` is not NONE.")
        .def(nb::init<>(), "The criterion.")
        .def_rw_static("tolerance",
                       &TooMuchRepelling::tolerance,
                       "The rejections per individual (default 50).");

    nb::bind_vector<vCriteria>(m,
                               "CriteriaVector",
                               "The list of restart criteria of `Criteria.items` (a mutable "
                               "sequence of `Criterion` objects).");

    nb::class_<Criteria>(
        m,
        "Criteria",
        "The restart criteria of an optimizer (`Parameters.criteria`).\n\n"
        "The defaults: `UnableToAdapt` always; with a restart strategy other than NONE also "
        "MinSigma, MaxSigma, ExceededMaxIter, NoImprovement, Stagnation, MinDSigma, MaxDSigma, "
        "FlatFitness (for lambda > 1), TolX and NoEffectCoord; ConditionC and NoEffectAxis with "
        "COVARIANCE, SEPARABLE and NONE (the matrix adaptations that keep their metric); with "
        "repelling also TooMuchRepelling.")
        .def_prop_rw(
            "items",
            [](Criteria &self) -> vCriteria & { return self.items; },
            [](Criteria &self, vCriteria v) { self.items = std::move(v); },
            nb::rv_policy::reference_internal,
            "The criteria, a list-like `CriteriaVector`; append, remove or assign a list of "
            "`Criterion` objects. An empty list means no restarts.")
        .def("reset",
             &Criteria::reset,
             nb::arg("parameters"),
             "Reset every criterion (`Criterion.reset`).")
        .def("update",
             &Criteria::update,
             nb::arg("parameters"),
             "Update every criterion (`Criterion.update`).")
        .def("reason",
             &Criteria::reason,
             "The names of the criteria that are met, each followed by \"; \" (empty when "
             "none is).")
        .def("any", &Criteria::any, "Whether any criterion is met.");
}

void define_restart_strategy(nb::module_ &main)
{
    auto m = submodule(main, "restart");
    using namespace restart;

    nb::class_<Strategy>(
        m,
        "Strategy",
        "A restart strategy: sets mu, lambda and sigma of the next run "
        "(`Parameters.restart_strategy`).\n\n"
        "This base class, used for RestartStrategy.NONE, STOP and RESTART, keeps mu and lambda "
        "and restarts with sigma0.")
        .def_rw("max_iter",
                &Strategy::max_iter,
                "The most generations of the current run: `ExceededMaxIter` uses the smaller of "
                "this and its own limit, at its reset. 2^64 - 1 (no limit) except in BIPOP's "
                "small runs.")
        .def("update",
             &Strategy::update,
             nb::arg("parameters"),
             "Prepare a restart: may change `Parameters.mu` and `Parameters.lamb`, and "
             "returns the step size of the next run (here sigma0).");

    nb::class_<IPOP, Strategy>(
        m,
        "IPOP",
        "IPOP (Auger & Hansen 2005): every restart multiplies lambda by `ipop_factor`, up to "
        "`max_lambda_factor` * lambda0 (2^9 lambda0, the framework paper).\n\n"
        "mu is the default of the new lambda, lambda // 2 (lambda // 4 with ssa SA or "
        "matrix_adaptation CMSA), as pycma's fmin (popsize // 2) and the framework paper (Alg. "
        "1); with a Settings.mu0 that was given, mu0 / lambda0 of the new lambda, rounded half "
        "up (at least 1). Before 2.0 mu kept the ratio mu / lambda of the previous run (#150).")
        .def_rw("ipop_factor",
                &IPOP::ipop_factor,
                "The factor of the population size per restart (default 2; lambda * ipop_factor "
                "is rounded).")
        .def_rw("max_lambda_factor",
                &IPOP::max_lambda_factor,
                "The largest population size as a multiple of lambda0 (default 512 = 2^9, the same "
                "for BIPOP's large regime).");

    nb::class_<BIPOP, Strategy>(
        m,
        "BIPOP",
        "BIPOP (Hansen 2009): restarts alternate between a large and a small population "
        "regime.\n\n"
        "Each regime counts the evaluations of its runs; the first run counts as small (pycma's "
        "fmin; Hansen 2009 disregards it). A restart uses the regime that has used fewer "
        "evaluations, the large one on a tie, so the first restart is large. A large run "
        "doubles lambda_large, the population of the latest large run (lambda_init before the "
        "first), up to `max_lambda_factor` * lambda_init (2^9 lambda_init, as IPOP), and "
        "restarts with sigma0. A small run draws lambda_small = floor(lambda_init (lambda_large "
        "/ lambda_init)^(u^2)) (rounded up to even for PAIRWISE, at least 2), between "
        "lambda_init and the latest large population, and restarts with sigma0 * 10^(-2u'), u "
        "and u' uniform in [0, 1); it takes at most half the evaluations of the large regime so "
        "far (`Strategy.max_iter`, which `ExceededMaxIter` enforces). mu follows lambda as in "
        "`IPOP`: lambda // 2 (pycma's fmin, Hansen 2009), or the ratio of a given mu0 (#150; "
        "before 2.0, lambda mu0 / lambda0, truncated).\n\n"
        "Up to the limit, the population sizes are those of pycma 4.6.0's `fmin` "
        "(cma/evolution_strategy.py:5362-5381) given an integer popsize (its default, 4 + 3 ln "
        "d, is not rounded there) and without PAIRWISE. Hansen (2009) writes the same formula "
        "with lambda_l / 2 in place of lambda_l, lambda_l the latest large population: half of "
        "pycma's range. pycma has no limit on the population, but its `fmin` stops after "
        "`restarts` large runs; with 9 (restarts=True), as Hansen (2009), its largest population "
        "is 2^9 lambda_def, this limit. modcma keeps restarting at the limit until the budget "
        "is used.")
        .def("large",
             &BIPOP::large,
             "Whether the next run is large: the large regime has used no more evaluations than "
             "the small one.")
        .def_rw("lambda_init", &BIPOP::lambda_init, "The initial lambda.")
        .def_rw("lambda_large",
                &BIPOP::lambda_large,
                "lambda of the latest large run (0 before the first restart, then 2 "
                "lambda_init, doubled at each large run up to max_lambda_factor * lambda_init).")
        .def_rw("lambda_small",
                &BIPOP::lambda_small,
                "lambda of the last small-regime draw (one at every restart), in [lambda_init, "
                "lambda_large]: up to the latest large population.")
        .def_rw("max_lambda_factor",
                &BIPOP::max_lambda_factor,
                "The largest population of the large regime as a multiple of lambda_init "
                "(default 512 = 2^9, as `IPOP.max_lambda_factor`).")
        .def_rw("used_small",
                &BIPOP::used_small,
                "The evaluations used by the small regime, the first run included.")
        .def_rw("used_large", &BIPOP::used_large, "The evaluations used by the large regime.")
        .def_rw("large_run",
                &BIPOP::large_run,
                "Whether the current run belongs to the large regime (False for the first run).")
        .def_ro("used_budget",
                &BIPOP::used_budget,
                "The evaluations used up to the last restart.");
}
} // namespace modcma_python
