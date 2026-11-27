// Bindings of the settings, the parameters, the population, the optimizer and the ES baselines.
#include "common.hpp"

namespace modcma_python
{
namespace
{
//! Getter of a population array: a view of the active columns 0..n-1 (bound with
//! rv_policy::reference_internal, so the view keeps the population alive)
auto active_columns(Matrix Population::*member)
{
    return [member](Population &p) {
        Matrix &m = p.*member;
        const auto n = std::min<Eigen::Index>(static_cast<Eigen::Index>(p.n), m.cols());
        return Eigen::Map<Matrix>(m.data(), m.rows(), n);
    };
}

auto active_entries(Vector Population::*member)
{
    return [member](Population &p) {
        Vector &v = p.*member;
        const auto n = std::min<Eigen::Index>(static_cast<Eigen::Index>(p.n), v.size());
        return Eigen::Map<Vector>(v.data(), n);
    };
}

constexpr auto view = nb::rv_policy::reference_internal;

template <typename Buffer> auto assign_buffer(Buffer Population::*member)
{
    return [member](Population &p, const Buffer &value) { p.*member = value; };
}

//! Getter of a vector of Settings derived from lb and ub: a read-only view (bound with
//! rv_policy::reference_internal); set lb or ub instead (#136)
auto derived_vector(Vector parameters::Settings::*member)
{
    return [member](const parameters::Settings &s) {
        const Vector &v = s.*member;
        return Eigen::Map<const Vector>(v.data(), v.size());
    };
}

std::optional<Indices> to_indices(const std::optional<std::vector<int>> &idx)
{
    if (!idx)
        return std::nullopt;
    return Eigen::Map<const Indices>(idx->data(), static_cast<Eigen::Index>(idx->size()));
}

//! A matrix from Python (a 2-d numpy array, a list of lists, ...) for Settings.C0; None stays None
std::optional<Matrix> to_matrix(nb::handle value)
{
    if (value.is_none())
        return std::nullopt;
    using RowMatrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    static const nb::object asarray = nb::module_::import_("numpy").attr("asarray");
    const nb::object a = asarray(value, nb::arg("dtype") = "float64");
    if (nb::cast<size_t>(a.attr("ndim")) != 2)
        throw nb::value_error("Settings: C0 must be a 2-d (dim, dim) matrix");
    return Matrix(nb::cast<RowMatrix>(a));
}
} // namespace

void define_population(nb::module_ &main)
{
    nb::class_<Population>(
        main,
        "Population",
        "The individuals of one generation, as the columns of (d, capacity) buffers.\n\n"
        "The first n columns are the active individuals. Reading an array returns a view of the "
        "active columns that shares memory with the population: X, X_transformed, Z, Y and S are "
        "(d, n) matrices, f and t are (n,) vectors. Assigning an array replaces the whole buffer "
        "and leaves n unchanged. The buffers can hold more than n columns, e.g. after sequential "
        "selection stopped a generation early; those columns hold no population data.\n\n"
        "A view is valid until the next generation: the optimizer then swaps pop and old_pop, "
        "and a restart or a larger population replaces the buffers (the view would read freed "
        "memory). Copy an array (`.copy()`) to keep its values.")
        .def(nb::init<size_t, size_t>(),
             nb::arg("dimension"),
             nb::arg("n"),
             "A population of n individuals in the given dimension.\n\n"
             "f is +inf for every individual; the other arrays are not initialised.")
        .def(nb::init<Matrix, Matrix, Matrix, Vector, Matrix>(),
             nb::arg("X"),
             nb::arg("Z"),
             nb::arg("Y"),
             nb::arg("f"),
             nb::arg("S"),
             "A population from its arrays.\n\n"
             "Parameters\n"
             "----------\n"
             "X : numpy.ndarray\n"
             "    (d, n) the points; X_transformed is a copy of X.\n"
             "Z : numpy.ndarray\n"
             "    (d, n) the samples of the sampler.\n"
             "Y : numpy.ndarray\n"
             "    (d, n) the samples transformed by the matrix adaptation.\n"
             "f : numpy.ndarray\n"
             "    (n,) the objective values.\n"
             "S : numpy.ndarray\n"
             "    (d, n) the step sizes.\n\n"
             "t (the generation of each individual) is not initialised.")
        .def("sort",
             &Population::sort,
             "Sort the active individuals by f, ascending, in every array.\n\n"
             "Ties keep their order and NaN values go last.")
        .def("resize_cols",
             &Population::resize_cols,
             nb::arg("size"),
             "Reduce the number of active individuals to min(size, n).\n\n"
             "The buffers keep their capacity; this never adds individuals.")
        .def("keep_only",
             &Population::keep_only,
             nb::arg("idx"),
             "Keep only the given columns, in the given order.\n\n"
             "Every buffer becomes its columns idx (indices into the buffers, 0..capacity-1) and n "
             "becomes len(idx).\n\n"
             "Parameters\n"
             "----------\n"
             "idx : list of int\n"
             "    The columns to keep.")
        .def_prop_ro("n_finite",
                     &Population::n_finite,
                     "The number of active individuals whose f is not +inf.\n\n"
                     "Unevaluated individuals, and the worse individual of each pair under "
                     "pairwise selection, have f = +inf.")
        .def(
            "__add__",
            [](const Population &self, const Population &other) {
                Population result = self;
                result += other;
                return result;
            },
            nb::arg("other"),
            "A new population with the active individuals of this one, then those of other.")
        .def("__iadd__",
             &Population::operator+=,
             nb::arg("other"),
             nb::rv_policy::none,
             "Append the active individuals of other to this population (in place).")
        // The arrays are views of the active columns 0..n-1 (fewer than the buffers hold after
        // sequential selection stopped a generation early); assigning replaces the buffer.
        .def_prop_rw("X",
                     active_columns(&Population::X),
                     assign_buffer(&Population::X),
                     view,
                     "(d, n) array: the sampled points, X = m + S * Y (elementwise), after bound "
                     "correction.\n\n"
                     "Points of the search space before the integer variables are rounded (see "
                     "X_transformed).")
        .def_prop_rw("X_transformed",
                     active_columns(&Population::X_transformed),
                     assign_buffer(&Population::X_transformed),
                     view,
                     "(d, n) array: the points that are evaluated.\n\n"
                     "X with the integer variables rounded (CoordinateMapping); equal to X "
                     "without integer variables.")
        .def_prop_rw("Z",
                     active_columns(&Population::Z),
                     assign_buffer(&Population::Z),
                     view,
                     "(d, n) array: the samples drawn from the sampler (standard normal by "
                     "default), lengthened by threshold convergence when it is on.")
        .def_prop_rw("Y",
                     active_columns(&Population::Y),
                     assign_buffer(&Population::Y),
                     view,
                     "(d, n) array: the samples Z transformed by the matrix adaptation (the "
                     "shape of the search distribution), before scaling by the step size.")
        .def_prop_rw("S",
                     active_columns(&Population::S),
                     assign_buffer(&Population::S),
                     view,
                     "(d, n) array: the step size of every individual and coordinate.\n\n"
                     "Every entry is sigma, unless sample_sigma is on (each individual draws its "
                     "own step sizes around sigma) or the lower bound of an integer variable "
                     "raises it.")
        .def_prop_rw("f",
                     active_entries(&Population::f),
                     assign_buffer(&Population::f),
                     view,
                     "(n,) array: the objective values; +inf before an individual is evaluated.")
        .def_prop_rw("t",
                     active_entries(&Population::t),
                     assign_buffer(&Population::t),
                     view,
                     "(n,) array: the generation (Stats.t) in which each individual was sampled.")
        .def_rw("d", &Population::d, "The dimension: the number of rows of the arrays.")
        .def_prop_rw(
            "n",
            [](const Population &p) { return p.n; },
            [](Population &p, const size_t n) {
                if (n > p.capacity())
                    throw nb::value_error(("Population.n: at most the capacity, " +
                                           std::to_string(p.capacity()))
                                              .c_str());
                p.n = n;
            },
            "The number of active individuals: the columns 0..n-1 of the buffers (at most "
            "capacity; otherwise ValueError).")
        .def_prop_ro("capacity",
                     &Population::capacity,
                     "The number of columns of the buffers (at least n).");
}

void define_parameters(nb::module_ &main)
{
    auto m = main.def_submodule(
        "parameters",
        "The configuration and the bookkeeping of the optimizer.\n\n"
        "Modules (the module choices) and Settings (dimension, budget, bounds, strategy "
        "parameters) configure a run; Weights holds the recombination weights and learning "
        "rates, Stats and Solution the progress of a run, CoordinateMapping the rounding of "
        "integer variables. The Parameters class itself is modcma.c_maes.Parameters.");
    using namespace parameters;

    nb::class_<Modules>(
        m,
        "Modules",
        "The module configuration: one choice per module.\n\n"
        "A bool attribute turns a module on or off; an enum attribute selects one option of an "
        "enum in modcma.c_maes.options. The defaults, Modules(), give the standard CMA-ES with "
        "cumulative step-size adaptation (CSA) and no restarts. Settings may adjust some choices "
        "so that they fit together (see Settings).")
        .def(nb::init<>(), "The default configuration: CMA-ES with CSA, every bool module off.")
        .def_prop_rw("elitist",
                     bool_getter(&Modules::elitist),
                     bool_setter(&Modules::elitist),
                     "bool, default False: (mu + lambda) selection.\n\n"
                     "The parents of the previous generation compete with the offspring. Can "
                     "speed up convergence on unimodal functions, at the cost of diversity.")
        .def_prop_rw("active",
                     bool_getter(&Modules::active),
                     bool_setter(&Modules::active),
                     "bool, default False: active covariance matrix update.\n\n"
                     "The worst individuals enter the update of the matrix with negative weights "
                     "(Jastrebski and Arnold). Settings turns it off for matrix_adaptation NONE "
                     "and for a (1+1)-ES.")
        .def_prop_rw("orthogonal",
                     bool_getter(&Modules::orthogonal),
                     bool_setter(&Modules::orthogonal),
                     "bool, default False: orthogonal sampling.\n\n"
                     "The samples of a generation are made orthogonal to each other (by a QR "
                     "decomposition), keeping their lengths (Wang et al.).")
        .def_prop_rw("sequential_selection",
                     bool_getter(&Modules::sequential_selection),
                     bool_setter(&Modules::sequential_selection),
                     "bool, default False: sequential selection (Brockhoff et al.).\n\n"
                     "The evaluation of a generation stops at the first individual, from index "
                     "mu on, that improves on the best of the previous generation; the rest of "
                     "the generation is not evaluated.")
        .def_prop_rw("threshold_convergence",
                     bool_getter(&Modules::threshold_convergence),
                     bool_setter(&Modules::threshold_convergence),
                     "bool, default False: threshold convergence (Piad-Morffis et al.).\n\n"
                     "A sample z shorter than the threshold t is lengthened to 2 t - ||z||. "
                     "t = init_threshold E||z|| ((budget - evaluations) / budget)^decay_factor "
                     "(`ThresholdConvergence`), computed once per generation: it decreases as the "
                     "budget is used, which prolongs the exploration phase.")
        .def_prop_rw("sample_sigma",
                     bool_getter(&Modules::sample_sigma),
                     bool_setter(&Modules::sample_sigma),
                     "bool, default False: every individual samples its own step sizes.\n\n"
                     "The step sizes are drawn from a log-normal distribution around sigma, "
                     "with learning rate Weights.beta (self-adaptation).")
        .def_rw("weights",
                &Modules::weights,
                "options.RecombinationWeights, default DEFAULT: the recombination weights.\n\n"
                "See modcma.c_maes.options.RecombinationWeights for the choices.")
        .def_rw("sampler",
                &Modules::sampler,
                "options.BaseSampler, default UNIFORM: the generator of the base points in "
                "[0, 1)^d.\n\n"
                "Pseudo-random or a low-discrepancy sequence; sample_transformation turns the "
                "points into samples of the search distribution. See "
                "modcma.c_maes.options.BaseSampler.")
        .def_rw("mirrored",
                &Modules::mirrored,
                "options.Mirror, default NONE: mirrored sampling.\n\n"
                "Every second sample is the mirror image of the one before; PAIRWISE also "
                "selects only the better point of each pair. See modcma.c_maes.options.Mirror.")
        .def_rw("ssa",
                &Modules::ssa,
                "options.StepSizeAdaptation, default CSA: the step-size adaptation.\n\n"
                "See modcma.c_maes.options.StepSizeAdaptation for the choices.")
        .def_rw("bound_correction",
                &Modules::bound_correction,
                "options.CorrectionMethod, default NONE: the correction of points outside the "
                "bounds lb and ub.\n\n"
                "See modcma.c_maes.options.CorrectionMethod for the choices.")
        .def_rw("restart_strategy",
                &Modules::restart_strategy,
                "options.RestartStrategy, default NONE: what happens when a restart criterion "
                "holds.\n\n"
                "Restart (with the same, an increasing or an alternating population size) or "
                "stop. See modcma.c_maes.options.RestartStrategy.")
        .def_rw("repelling_type",
                &Modules::repelling_type,
                "options.RepellingType, default NONE: repelling restarts.\n\n"
                "Restarts avoid the regions of the optima found by earlier runs (taboo "
                "points). See modcma.c_maes.options.RepellingType.")
        .def_rw("repelling_metric",
                &Modules::repelling_metric,
                "options.RepellingMetric, default EUCLIDEAN: the distance between a sample and "
                "a taboo point of repelling restarts.\n\n"
                "See modcma.c_maes.options.RepellingMetric for the choices.")
        .def_rw("matrix_adaptation",
                &Modules::matrix_adaptation,
                "options.MatrixAdaptationType, default COVARIANCE: the adaptation of the shape "
                "of the search distribution.\n\n"
                "Settings replaces COVARIANCE by COVARIANCE_NO_EIGV when ssa is not CSA (unless "
                "always_compute_eigv). See modcma.c_maes.options.MatrixAdaptationType.")
        .def_rw("center_placement",
                &Modules::center_placement,
                "options.CenterPlacement, default X0: where the mean of the search distribution "
                "starts after a restart.\n\n"
                "See modcma.c_maes.options.CenterPlacement for the choices.")
        .def_rw("sample_transformation",
                &Modules::sample_transformation,
                "options.SampleTranformerType, default GAUSSIAN: the distribution of the "
                "samples.\n\n"
                "Transforms the base points of the sampler into samples of this distribution. "
                "See modcma.c_maes.options.SampleTranformerType.")
        .def(
            "__repr__",
            [](Modules &mod) { return to_string(mod); },
            "The module choices as text.");

    nb::class_<Solution>(m, "Solution", "A point with its objective value and when it was found.")
        .def(nb::init<>(), "An empty solution: x is empty, y is +inf, t and e are 0.")
        .def_prop_rw(
            "x",
            [](const Solution &s) { return s.x; },
            [](Solution &s, const Vector &x) { s.x = x; },
            "(d,) array: the point, as evaluated (integer variables rounded). A copy: it keeps its "
            "values when the optimizer finds a better point, and changing it does not change the "
            "optimizer.")
        .def_rw("y", &Solution::y, "The objective value of x; +inf for an empty solution.")
        .def_rw("t", &Solution::t, "The generation in which x was found.")
        .def_rw("e", &Solution::e, "The number of evaluations when x was found.")
        .def("__repr__", &Solution::repr, "x and y as text.");

    nb::class_<Center, Solution>(
        m,
        "Center",
        "The record of a finished run: its evaluated center of mass, with the distribution the "
        "run ended with, and why it ended (#111).\n\n"
        "Every run ends with an evaluation of its center of mass (`ModularCMAES.center_of_mass`), "
        "at a restart and when the optimizer stops: x and y are that point and its value, t the "
        "last generation of the run and e the evaluations after it. The run ended with the "
        "distribution N(x, sigma^2 C), C = B diag(d)^2 B^T, with d and B in `metric`. Every run "
        "is recorded, converged or not: `std` and `reasons` tell how it ended.")
        .def(nb::init<>(), "An empty record: x is empty, y is +inf, no metric, no reasons.")
        .def_ro("sigma", &Center::sigma, "The step size at the end of the run.")
        .def_ro("metric",
                &Center::metric,
                "The shape of the distribution at the end of the run, "
                "`matrix_adaptation.Metric` (the run's last `Adaptation.metric()`).")
        .def_ro("reasons",
                &Center::reasons,
                "list of str: why the run ended: the names of the restart criteria that were met, "
                "and \"budget\", \"target\" or \"max_generations\" when one of those ended it.")
        .def_prop_ro(
            "std",
            [](const Center &c) { return c.max_std(); },
            "sigma * max(metric.d): the longest semi-axis of the run's one-sigma ellipsoid, the "
            "radius of the smallest ball that contains it. How far the run converged (NaN for an "
            "empty record).")
        .def(
            "__repr__",
            [](const Center &c) {
                return repr("Center", [&](std::stringstream &ss) {
                    ss << " x: (" << c.x.transpose() << ") y: " << c.y << " std: " << c.max_std();
                });
            },
            "x, y and std as text.");

    nb::class_<Stats>(m, "Stats", "The progress of a run: counters and the best solutions.")
        .def(nb::init<>(), "Empty statistics: no generation, no evaluation, no solution.")
        .def_rw("t",
                &Stats::t,
                "The generation counter, incremented at the end of every generation; restarts do "
                "not reset it.")
        .def_rw("t_run",
                &Stats::t_run,
                "The generations of the current run: reset at a restart. h_sigma's bias "
                "correction counts these.")
        .def_rw("evaluations",
                &Stats::evaluations,
                "The number of objective function evaluations so far.\n\n"
                "Includes the extra points of TPA, the center of mass that ends every run and the "
                "hill-valley tests of repelling restarts.")
        .def_rw("current_avg",
                &Stats::current_avg,
                "The mean objective value of the population after the last selection (+inf "
                "before the first).")
        .def_rw("solutions",
                &Stats::solutions,
                "list of Solution: the best solution of every finished run, appended when it "
                "ends, at a restart and when the optimizer stops (a copy). Its center of mass, "
                "evaluated after it, can be better: see centers.")
        .def_rw("centers",
                &Stats::centers,
                "list of Center: the record of every finished run, appended when it ends, at a "
                "restart and when the optimizer stops: its evaluated center of mass, with sigma, "
                "the metric and why it ended (a copy). One per run, as solutions.")
        .def_rw("current_best",
                &Stats::current_best,
                "The best solution of the current run (since the last restart).")
        .def_rw("global_best",
                &Stats::global_best,
                "The best solution found so far, over all runs.")
        .def_rw("has_improved",
                &Stats::has_improved,
                "Whether the last selection improved current_best.")
        .def_rw("success_ratio",
                &Stats::success_ratio,
                "The smoothed rate of successful generations.\n\n"
                "A generation is a success when its best individual is better than the best of "
                "the run so far (with elitism, the parent); counted once per generation. "
                "An exponential moving average with weight 1/12, starting at 2/11. Used by the "
                "success rule step-size adaptation (SR) and by the update of the evolution path "
                "in the (1+1) configuration (lambda0 = 1).")
        .def_rw("last_update",
                &Stats::last_update,
                "The generation of the last update of the matrix of the matrix adaptation.")
        .def_rw("n_updates",
                &Stats::n_updates,
                "The number of updates of the matrix of the matrix adaptation so far.")
        .def("__repr__", [](Stats &stats) {
            return repr("Stats", [&](std::stringstream &ss) {
                ss << " t: " << stats.t << " e: " << stats.evaluations;
                ss << " best: " << stats.global_best << " improved: " << stats.has_improved;
            });
        },
        "The generation, the evaluations, the best solution and has_improved as text.");

    nb::class_<Weights>(
        m,
        "Weights",
        "The recombination weights and the learning rates of a run.\n\n"
        "Computed when the Parameters are created, and again at every restart with the current "
        "mu and lambda. A learning rate given in the Settings is used as is; the defaults below "
        "use d for the dimension.")
        .def(nb::init<Settings, Float>(),
             nb::arg("Settings"),
             nb::arg("expected_length_z"),
             "The weights and learning rates for settings.mu0 and settings.lambda0.\n\n"
             "Parameters\n"
             "----------\n"
             "Settings : Settings\n"
             "    The settings: dimension, modules and the learning rates given there.\n"
             "expected_length_z : float\n"
             "    The expected length of a sample of the sampler, E||z|| (Parameters pass the "
             "sampler's expected_length()).")
        .def_rw("mueff",
                &Weights::mueff,
                "The variance effective selection mass of the positive weights, "
                "1 / sum(positive**2).")
        .def_rw("mueff_mean",
                &Weights::mueff_mean,
                "The variance effective selection mass of the weights of the mean, "
                "1 / sum(w**2): `mueff`, except for UTILITY, where the mean uses all lambda "
                "utilities. Normalizes the evolution paths.")
        .def_rw("utility",
                &Weights::utility,
                "Whether the weights are the NES utilities (`RecombinationWeights.UTILITY`).")
        .def_rw("pairwise",
                &Weights::pairwise,
                "Whether the selection is pairwise (`Mirror.PAIRWISE`): the losers of the mirrored "
                "pairs (f = +inf, sorted last) then get no negative weights in the active update, "
                "as in pycma.")
        .def_rw("mueff_neg",
                &Weights::mueff_neg,
                "The variance effective selection mass of the negative weights, "
                "sum(negative)**2 / sum(negative**2).")
        .def_rw("c1",
                &Weights::c1,
                "The learning rate of the rank-one update of the matrix.\n\n"
                "Default acov / ((d + 1.3)**2 + mueff); 2 / (d**2 + 6) for a (1+1)-ES "
                "(lambda0 = 1).")
        .def_rw("cmu",
                &Weights::cmu,
                "The learning rate of the rank-mu update of the matrix.\n\n"
                "Default min(1 - c1, acov (mueff - 2 + 1 / mueff) / ((d + 2)**2 + acov mueff / "
                "2)), multiplied by (d + 2) / 3 for SEPARABLE; 2 / (d**2 + 6) for lambda0 = 1.")
        .def_rw("cc",
                &Weights::cc,
                "The learning rate of the evolution path pc.\n\n"
                "Default (4 + mueff / d) / (d + 4 + 2 mueff / d); 2 / (d + 2) for a (1+1)-ES; "
                "with NATURAL_GRADIENT cs / 2, or 1 / (4 d**1.5) for a (1+1)-ES.")
        .def_rw("cs",
                &Weights::cs,
                "The learning rate of the step-size adaptation.\n\n"
                "Default by ssa: CSA (mueff + 2) / (d + mueff + 5) (Hansen, The CMA Evolution "
                "Strategy: A Tutorial, arXiv:1604.00772v2, Table 1, Eq. (55); pycma uses d + "
                "mueff + 3 and caps cs at 1/2); SR 2 / (d + 2); XNES 0.01; MXNES 1; LPXNES 9 mueff "
                "/ (10 sqrt(d)); otherwise (TPA, MSR, PSR, SA) 0.3. With matrix_adaptation "
                "NATURAL_GRADIENT (any ssa): (9 + 3 ln(d)) / (5 d sqrt(d)).")
        .def_rw("damps",
                &Weights::damps,
                "The damping of the step-size update.\n\n"
                "Default for CSA 1 + 2 max(0, sqrt((mueff - 1) / (d + 1)) - 1) + cs (Hansen, "
                "arXiv:1604.00772v2, Table 1, Eq. (55)); for SR 1 + d / 2; for MSR max(1, 2 - 2 / "
                "d); otherwise (PSR among them) 1.")
        .def_rw("sqrt_cc_mueff",
                &Weights::sqrt_cc_mueff,
                "sqrt(cc (2 - cc) mueff_mean): the normalisation of the update of the path pc "
                "(mueff_mean = mueff, or 1 / sum u_i^2 for the utilities of UTILITY).")
        .def_rw("sqrt_cs_mueff",
                &Weights::sqrt_cs_mueff,
                "sqrt(cs (2 - cs) mueff_mean): the normalisation of the update of the path ps "
                "(mueff_mean = mueff, or 1 / sum u_i^2 for the utilities of UTILITY).")
        .def_rw("lazy_update_interval",
                &Weights::lazy_update_interval,
                "The number of generations between two eigendecompositions of the covariance "
                "matrix (COVARIANCE), 1 / (c1 + cmu) / d / 10; the matrix itself is updated "
                "every generation.\n\n"
                "Below 1, the matrix is decomposed every generation.")
        .def_rw("expected_length_z",
                &Weights::expected_length_z,
                "The expected length of a sample of the sampler, E||z||.")
        .def_rw("expected_length_ps",
                &Weights::expected_length_ps,
                "(1.4 + 2 / (d + 1)) expected_length_z: the threshold on the length of ps above "
                "which the update of the path pc stalls (h_sigma).")
        .def_rw("beta",
                &Weights::beta,
                "The learning rate of the sampled step sizes (sample_sigma).\n\n"
                "1 / sqrt(2 mueff); log(2) / (sqrt(d) log(d)) with LPXNES.")
        .def_rw("weights",
                &Weights::weights,
                "(lambda,) array: all weights by rank, positive followed by negative.")
        .def_rw("positive",
                &Weights::positive,
                "(mu,) array: the weights of the mu best individuals, normalised to sum 1.")
        .def_rw("negative",
                &Weights::negative,
                "(lambda - mu,) array: the weights of the other individuals, for the active "
                "update.\n\n"
                "Scaled so that their absolute values sum to min(1 + c1 / cmu, 1 + 2 mueff_neg / "
                "(mueff + 2), (1 - c1 - cmu) / (d cmu)) (Hansen 2016); with UTILITY the "
                "utilities are used as they are (they sum to zero, as in NES).")
        .def("for_population",
             &Weights::for_population,
             nb::arg("n"),
             "The weights for a population of n evaluated individuals.\n\n"
             "After sequential selection stopped a generation early (mu < n < lambda), the "
             "positive weights are unchanged and the negative weights of ranks mu..n-1 are scaled "
             "so that their sum equals the sum of all lambda - mu negative weights. For n >= "
             "lambda, these are the weights themselves.\n\n"
             "Parameters\n"
             "----------\n"
             "n : int\n"
             "    The number of evaluated individuals.\n\n"
             "Returns\n"
             "-------\n"
             "numpy.ndarray\n"
             "    (min(n, lambda),) the weights.")
        .def("__repr__", [](Weights &w) {
            return repr("Weights", [&](std::stringstream &ss) {
                ss << " mueff: " << w.mueff << " mueff_neg: " << w.mueff_neg;
                ss << " c1: " << w.c1 << " cmu: " << w.cmu << " cc: " << w.cc;
                ss << " weights: " << w.weights.transpose();
                ss << " positive: " << w.positive.transpose();
                ss << " negative: " << w.negative.transpose();
            });
        },
        "The selection masses, learning rates and weights as text.");

    // dim and modules are positional; everything else is keyword-only (mu0 before lambda0, as
    // everywhere in modcma). A positional call with more arguments raises a TypeError instead of
    // silently assigning values to the wrong parameters.
    nb::class_<Settings>(
        m,
        "Settings",
        "The settings of a run: dimension, modules, stopping conditions, search space and "
        "strategy parameters.\n\n"
        "A strategy parameter left None gets its default when the Parameters are created (see "
        "Weights). The constructor computes the derived values (db, center, diameter, volume, "
        "has_bounds) and adjusts the modules.\n\n"
        "Setting a field afterwards checks that field alone (ValueError for a wrong length, NaN, "
        "a sign or a range) and recomputes what is derived from it: lb and ub the derived values "
        "of the box, lambda0 the default mu0. The fields can therefore be set in any order. The "
        "checks between fields (lb <= ub, the integer variables in the box, mu0 <= lambda0, the "
        "budget against the first generation, lambda0 with the modules) run in the constructor "
        "and again when an optimizer is created from the settings (ModularCMAES(settings), "
        "Parameters(settings)): ValueError naming the fields. Changing the modules afterwards, "
        "in place, does not adjust them.")
        .def(
            "__init__",
            [](Settings *self,
               size_t dim,
               std::optional<Modules> modules,
               std::optional<Float> target,
               size_to max_generations,
               size_to budget,
               std::optional<Float> sigma0,
               std::optional<size_t> mu0,
               std::optional<size_t> lambda0,
               std::optional<std::vector<Float>> x0,
               std::optional<std::vector<Float>> lb,
               std::optional<std::vector<Float>> ub,
               std::optional<std::vector<int>> integer_variables,
               std::optional<Float> cs,
               std::optional<Float> cc,
               std::optional<Float> cmu,
               std::optional<Float> c1,
               std::optional<Float> damps,
               std::optional<Float> acov,
               bool verbose,
               bool always_compute_eigv,
               std::optional<int> seed,
               bool fast_sampling,
               nb::handle C0) {
                new (self) Settings(dim,
                                    modules,
                                    target,
                                    max_generations,
                                    budget,
                                    sigma0,
                                    mu0,
                                    lambda0,
                                    to_vector(x0),
                                    to_vector(lb),
                                    to_vector(ub),
                                    to_indices(integer_variables),
                                    cs,
                                    cc,
                                    cmu,
                                    c1,
                                    damps,
                                    acov,
                                    verbose,
                                    always_compute_eigv);
                self->seed = seed;
                self->fast_sampling = fast_sampling;
                self->set_C0(to_matrix(C0));
            },
            // array_like for the vectors (numpy arrays, lists, ...), as accepted at run time
            nb::sig("def __init__(self, dim: int, modules: Modules | None = None, *, "
                    "target: float | None = None, max_generations: int | None = None, "
                    "budget: int | None = None, sigma0: float | None = None, "
                    "mu0: int | None = None, lambda0: int | None = None, "
                    "x0: numpy.typing.ArrayLike | None = None, "
                    "lb: numpy.typing.ArrayLike | None = None, "
                    "ub: numpy.typing.ArrayLike | None = None, "
                    "integer_variables: numpy.typing.ArrayLike | None = None, "
                    "cs: float | None = None, cc: float | None = None, cmu: float | None = None, "
                    "c1: float | None = None, damps: float | None = None, "
                    "acov: float | None = None, verbose: bool = False, "
                    "always_compute_eigv: bool = False, seed: int | None = None, "
                    "fast_sampling: bool = False, "
                    "C0: numpy.typing.ArrayLike | None = None) -> None"),
            nb::arg("dim"),
            nb::arg("modules") = nb::none(),
            nb::kw_only(),
            nb::arg("target") = nb::none(),
            nb::arg("max_generations") = nb::none(),
            nb::arg("budget") = nb::none(),
            nb::arg("sigma0") = nb::none(),
            nb::arg("mu0") = nb::none(),
            nb::arg("lambda0") = nb::none(),
            nb::arg("x0") = nb::none(),
            nb::arg("lb") = nb::none(),
            nb::arg("ub") = nb::none(),
            nb::arg("integer_variables") = nb::none(),
            nb::arg("cs") = nb::none(),
            nb::arg("cc") = nb::none(),
            nb::arg("cmu") = nb::none(),
            nb::arg("c1") = nb::none(),
            nb::arg("damps") = nb::none(),
            nb::arg("acov") = nb::none(),
            nb::arg("verbose") = false,
            nb::arg("always_compute_eigv") = false,
            nb::arg("seed") = nb::none(),
            nb::arg("fast_sampling") = false,
            nb::arg("C0") = nb::none(),
            "Settings for a problem of dimension dim.\n\n"
            "dim and modules are positional, every other argument is keyword-only. None means "
            "a default computed from the dimension and the modules.\n\n"
            "The constructor makes the configuration consistent:\n\n"
            "- with mirrored PAIRWISE, an odd lambda0 above 1 is increased by one;\n"
            "- with ssa SA or matrix_adaptation CMSA, the default mu0 is lambda0 // 4;\n"
            "- when ssa is not CSA, matrix_adaptation COVARIANCE becomes COVARIANCE_NO_EIGV "
            "(unless always_compute_eigv);\n"
            "- with matrix_adaptation NONE, active is turned off;\n"
            "- lambda0 = 1 gives a (1+1)-ES (one_plus_one): mu0 = 1, elitist, ssa SR, EQUAL "
            "weights, no active update, no sequential selection, mirrored PAIRWISE becomes "
            "NONE, and IPOP or BIPOP become RESTART;\n"
            "- when the box lb..ub is not finite, has_bounds is False, diameter 10, volume "
            "10**dim; center is computed per coordinate (see `center`).\n\n"
            "Invalid values raise ValueError, naming the argument: dim 0, lambda0 0, mu0 0, a "
            "mu0 above lambda0 (except for lambda0 = 1, which uses mu0 = 1), a NaN target, a "
            "sigma0 that is not positive and finite, x0, lb or ub of a length other "
            "than dim, a non-finite x0, NaN in lb or ub, +inf in lb or -inf in ub, lb above "
            "ub, an integer variable without an integer value in [lb, ub], a budget smaller "
            "than the first generation and the evaluation of the final center of mass, or a C0 "
            "that is not a symmetric positive definite (dim, dim) matrix (or with "
            "matrix_adaptation NONE).\n\n"
            "Parameters\n"
            "----------\n"
            "dim : int\n"
            "    The dimension of the search space.\n"
            "modules : Modules, optional\n"
            "    The module configuration (copied). Default Modules(): CMA-ES with CSA.\n"
            "target : float, optional\n"
            "    Stop when the best objective value is at most target; not NaN. Default None: "
            "no target.\n"
            "max_generations : int, optional\n"
            "    Stop after this many generations, at least 1. Default None: no limit.\n"
            "budget : int, optional\n"
            "    The maximum number of evaluations, default 10000 * dim; at least the first "
            "generation (lambda0, and TPA's 2 points) and the final center of mass. A generation "
            "runs only when it leaves one evaluation: the run stops when the evaluations so far "
            "plus lambda (and TPA's 2 points) plus 1 would exceed it, and every run ends with "
            "the evaluation of its center of mass (#111).\n"
            "sigma0 : float, optional\n"
            "    The initial step size, default 2.0, positive and finite; also the step size "
            "after a restart (the small-population runs of BIPOP use sigma0 * 10**(-2 u), u "
            "uniform in [0, 1)).\n"
            "mu0 : int, optional\n"
            "    The number of parents, default lambda0 // 2 (lambda0 // 4 with ssa SA or "
            "matrix_adaptation CMSA); at least 1 and at most lambda0.\n"
            "lambda0 : int, optional\n"
            "    The number of offspring per generation, default 4 + floor(3 ln(dim)); at least 1 "
            "(1 is the (1+1)-ES). Many modules work poorly with 2 or 3 (see the compatibility "
            "page).\n"
            "x0 : array_like, optional\n"
            "    (dim,) the initial mean of the search distribution. Default None: the center of "
            "the bounds (the zero vector when the box is not finite); after a restart, "
            "center_placement X0 uses the same point.\n"
            "lb : array_like, optional\n"
            "    (dim,) the lower bounds, default -5 in every coordinate.\n"
            "ub : array_like, optional\n"
            "    (dim,) the upper bounds, default 5 in every coordinate.\n"
            "integer_variables : sequence of int, optional\n"
            "    The indices of the variables that take integer values: they are rounded before "
            "evaluation (a value in [lb, ub] to the nearest integer in [lb, ub]) and their step "
            "size has a lower bound. Each needs an integer value in [lb, ub]. Default: none.\n"
            "cs : float, optional\n"
            "    The learning rate of the step-size adaptation, in (0, 1]. Default depends on ssa (see "
            "Weights.cs); for CSA (mueff + 2) / (dim + mueff + 5) (Hansen, The CMA Evolution "
            "Strategy: A Tutorial, arXiv:1604.00772v2, Table 1, Eq. (55)).\n"
            "cc : float, optional\n"
            "    The learning rate of the evolution path pc, in (0, 1]. Default (4 + mueff / dim) / (dim + "
            "4 + 2 mueff / dim) (see Weights.cc).\n"
            "cmu : float, optional\n"
            "    The learning rate of the rank-mu update, in [0, 1]. Default min(1 - c1, acov (mueff - 2 + "
            "1 / mueff) / ((dim + 2)**2 + acov mueff / 2)) (see Weights.cmu).\n"
            "c1 : float, optional\n"
            "    The learning rate of the rank-one update, in [0, 1]; with cmu, c1 + cmu <= 1. Default acov / ((dim + 1.3)**2 + "
            "mueff) (see Weights.c1).\n"
            "damps : float, optional\n"
            "    The damping of the step-size update, positive. Default for CSA 1 + 2 max(0, sqrt((mueff - "
            "1) / (dim + 1)) - 1) + cs (see Weights.damps).\n"
            "acov : float, optional\n"
            "    The factor alpha_cov in the defaults of c1 and cmu, positive, default 2.\n"
            "verbose : bool, default False\n"
            "    Print a message at every restart and when a decomposition of the matrix "
            "fails.\n"
            "always_compute_eigv : bool, default False\n"
            "    Keep matrix_adaptation COVARIANCE, with its eigendecomposition, when ssa is not "
            "CSA.\n"
            "seed : int, optional\n"
            "    The seed of the optimizer's own random generator. Default None: the optimizer "
            "starts from a copy of the global generator (set_seed) when it is created.\n"
            "fast_sampling : bool, default False\n"
            "    Sample with xoshiro256++ and a ziggurat instead of mt19937 and the inverse "
            "normal CDF (the default UNIFORM + GAUSSIAN sampler only): faster, but a different "
            "random stream.\n"
            "C0 : array_like, optional\n"
            "    (dim, dim) the initial covariance matrix C, symmetric positive definite: the "
            "first generation samples N(x0, sigma0**2 C0). Default None: the identity. See the "
            "attribute C0; modcma.warm_start estimates x0, sigma0 and C0 from the solutions of "
            "a related task.")
        // The read-only properties name their getter: Python's "property 'db' of 'Settings'
        // object has no setter" then names the field (nanobind leaves the name empty)
        .def_ro("dim",
                &Settings::dim,
                nb::name("dim"),
                "The dimension of the search space (read-only).")
        .def_rw("seed",
                &Settings::seed,
                "The seed of the optimizer's own random generator, or None.\n\n"
                "None: the optimizer starts from a copy of the global generator (set_seed) "
                "when it is created. Set it before the optimizer is created.")
        .def_rw("fast_sampling",
                &Settings::fast_sampling,
                "Opt-in: sample with xoshiro256++ and a ziggurat, default False.\n\n"
                "Instead of mt19937 and the inverse normal CDF (the default UNIFORM + GAUSSIAN "
                "sampler only). Faster, but a different random stream. Set it before the "
                "optimizer is created.")
        .def_prop_rw(
            "C0",
            [](const Settings &s) { return s.C0; },
            [](Settings &s, nb::handle C0) { s.set_C0(to_matrix(C0)); },
            nb::for_setter(nb::sig("def C0(self, arg: numpy.typing.ArrayLike | None, /) -> None")),
            nb::for_setter(nb::arg("value").none()),
            "(dim, dim) array: the initial covariance matrix C, or None (default) for the "
            "identity.\n\n"
            "The first generation samples N(x0, sigma0**2 C0) (warm starting, Nomura et al. "
            "2021; see modcma.warm_start). Each matrix adaptation starts its matrix from a factor "
            "F of C0, F F^T = C0: COVARIANCE from the eigendecomposition, COVARIANCE_NO_EIGV, "
            "CHOLESKY and CMSA from the lower Cholesky factor, MATRIX and NATURAL_GRADIENT from "
            "the symmetric square root. SEPARABLE uses only the diagonal of C0; "
            "matrix_adaptation NONE rejects C0 (ValueError). Every restart starts from C0 "
            "again, as from sigma0 and x0.\n\n"
            "Setting it checks it: a (dim, dim) finite matrix, symmetric up to rounding (a "
            "relative difference ||C0 - C0^T|| <= 1e-10 ||C0||, Frobenius norm; its symmetric "
            "part (C0 + C0^T) / 2 is stored) and positive definite, otherwise ValueError. Set it "
            "before the optimizer is created; the getter returns a copy.")
        .def_ro("modules",
                &Settings::modules,
                nb::name("modules"),
                "The module configuration, after the adjustments of the constructor.\n\n"
                "The attribute is read-only. Its fields can still be changed in place, but the "
                "adjustments of the constructor are not applied again, and an optimizer that "
                "exists keeps the modules it was created with.")
        .def_prop_rw(
            "target",
            [](const Settings &s) { return s.target; },
            [](Settings &s, const std::optional<Float> value) { s.set_target(value); },
            "Stop when the best objective value is at most target; None (default): no "
            "target. NaN raises ValueError: it would never stop the run.")
        .def_prop_rw(
            "max_generations",
            [](const Settings &s) { return s.max_generations; },
            [](Settings &s, const std::optional<size_t> value) { s.set_max_generations(value); },
            "Stop after this many generations, at least 1 (ValueError); None (default): no "
            "limit.")
        .def_prop_rw(
            "budget",
            [](const Settings &s) { return s.budget; },
            [](Settings &s, const size_t value) { s.set_budget(value); },
            "The maximum number of evaluations, default 10000 * dim; at least 2 "
            "(ValueError).\n\n"
            "A generation runs only when it leaves one evaluation: the run stops when the "
            "evaluations so far plus lambda (and TPA's 2 points) plus 1 would exceed it, "
            "and every run ends with the evaluation of its center of mass (#111). The "
            "budget must hold the first generation (lambda0, and TPA's 2 points) and the "
            "final center of mass: the constructor checks it, and after a setter the "
            "optimizer does when it is created (ValueError).")
        .def_prop_rw(
            "sigma0",
            [](const Settings &s) { return s.sigma0; },
            [](Settings &s, const Float value) { s.set_sigma0(value); },
            "The initial step size, default 2.0; positive and finite (ValueError). Also "
            "the step size after a restart (smaller for the small-population runs of "
            "BIPOP). Setting it recomputes `center`, which depends on it for a coordinate "
            "with one infinite bound.")
        .def_prop_rw(
            "mu0",
            [](const Settings &s) { return s.mu0; },
            [](Settings &s, const size_t value) { s.set_mu0(value); },
            "The initial number of parents, at least 1 (ValueError); default lambda0 // 2 "
            "(lambda0 // 4 with ssa SA or matrix_adaptation CMSA).\n\n"
            "Until mu0 is given, to the constructor or by setting it, it follows lambda0: setting "
            "lambda0 recomputes it. A mu0 above lambda0 raises ValueError: in the constructor, "
            "and after a setter when the optimizer is created. Parameters.mu is the current "
            "value, which IPOP and BIPOP restarts change.")
        .def_prop_rw(
            "lambda0",
            [](const Settings &s) { return s.lambda0; },
            [](Settings &s, const size_t value) { s.set_lambda0(value); },
            "The initial number of offspring per generation, at least 1 (ValueError); default 4 "
            "+ floor(3 ln(dim)).\n\n"
            "Setting it recomputes mu0, unless mu0 was given. The setter does not apply the "
            "adjustments of the constructor; the optimizer raises ValueError when it is created "
            "with an odd lambda0 and mirrored PAIRWISE (the constructor adds one), with lambda0 "
            "= 1 set after construction or another lambda0 for settings created with lambda0 = 1 "
            "(the (1+1)-ES, for which the constructor adjusts the modules), or with a budget "
            "that does not hold the first generation. Parameters.lamb is the current value, "
            "which IPOP and BIPOP restarts change.")
        .def_prop_rw(
            "x0",
            [](const Settings &s) { return s.x0; },
            [](Settings &s, const std::optional<std::vector<Float>> &x0) {
                s.set_x0(to_vector(x0));
            },
            // array_like, as the constructor
            nb::for_setter(nb::sig("def x0(self, arg: numpy.typing.ArrayLike | None, /) -> None")),
            "(dim,) array: the initial mean of the search distribution, or None (default) for "
            "`center` (the midpoint of the bounds; see `center` for infinite bounds). A length other than dim, or a value that is not "
            "finite, raises ValueError.")
        .def_prop_rw(
            "lb",
            [](const Settings &s) { return s.lb; },
            [](Settings &s, const std::vector<Float> &lb) { s.set_lb(to_vector(lb)); },
            nb::for_setter(nb::sig("def lb(self, arg: numpy.typing.ArrayLike, /) -> None")),
            "(dim,) array: the lower bounds, default -5; -inf for none. lb_i == ub_i, finite, "
            "fixes the variable. Setting it updates `db`, `center`, `diameter`, `volume` and "
            "`has_bounds`; ValueError for a length other than dim, NaN or +inf. Its relation to "
            "`ub` (lb <= ub, the integer variables in the box) is checked when the optimizer is "
            "created, so lb and ub can be set in any order.")
        .def_prop_rw(
            "ub",
            [](const Settings &s) { return s.ub; },
            [](Settings &s, const std::vector<Float> &ub) { s.set_ub(to_vector(ub)); },
            nb::for_setter(nb::sig("def ub(self, arg: numpy.typing.ArrayLike, /) -> None")),
            "(dim,) array: the upper bounds, default 5; inf for none. lb_i == ub_i, finite, "
            "fixes the variable. Setting it updates `db`, `center`, `diameter`, `volume` and "
            "`has_bounds`; ValueError for a length other than dim, NaN or -inf. Its relation to "
            "`lb` (lb <= ub, the integer variables in the box) is checked when the optimizer is "
            "created, so lb and ub can be set in any order.")
        .def_prop_ro("db",
                     derived_vector(&Settings::db),
                     view,
                     nb::name("db"),
                     "(dim,) array: the widths of the box, ub - lb (read-only: derived from lb "
                     "and ub).")
        .def_prop_ro("center",
                     derived_vector(&Settings::center),
                     view,
                     nb::name("center"),
                     "(dim,) array: where the first run starts without x0, per coordinate: the "
                     "midpoint (lb + ub) / 2 for two finite bounds; 0 for two infinite bounds; "
                     "for one infinite bound, 0 if the half-line holds it, otherwise the finite "
                     "bound moved inward by sigma0 (lb + sigma0, or ub - sigma0), so the start "
                     "lies in the box (read-only: derived from lb, ub and sigma0; x0 sets the "
                     "starting point).")
        .def_ro("diameter",
                &Settings::diameter,
                nb::name("diameter"),
                "The length of the diagonal of the box, ||ub - lb||; 10 when the box is not "
                "finite (then volume is 10^dim). Read-only: derived from lb and ub.")
        .def_ro("volume",
                &Settings::volume,
                nb::name("volume"),
                "The volume of the box, prod(ub - lb); 10**dim when the box is not finite. "
                "Read-only: derived from lb and ub.")
        .def_prop_ro(
            "has_bounds",
            bool_getter(&Settings::has_bounds),
            nb::name("has_bounds"),
            "Whether the box lb..ub is finite (`diameter`, `volume` and `center` depend on "
            "it); read-only, derived from lb and ub. Bound correction does not depend on "
            "it: it corrects every finite bound.")
        .def_ro("integer_variables",
                &Settings::integer_variables,
                nb::name("integer_variables"),
                "(k,) int array: the indices of the integer variables (read-only).")
        .def_prop_rw(
            "cs",
            [](const Settings &s) { return s.cs; },
            [](Settings &s, const std::optional<Float> value) { s.set_cs(value); },
            "The learning rate cs given to the constructor, or None for the default; in (0, 1] "
            "(ValueError).\n\n"
            "The value in use is Parameters.weights.cs.")
        .def_prop_rw(
            "cc",
            [](const Settings &s) { return s.cc; },
            [](Settings &s, const std::optional<Float> value) { s.set_cc(value); },
            "The learning rate cc given to the constructor, or None for the default; in (0, 1] "
            "(ValueError).\n\n"
            "The value in use is Parameters.weights.cc.")
        .def_prop_rw(
            "cmu",
            [](const Settings &s) { return s.cmu; },
            [](Settings &s, const std::optional<Float> value) { s.set_cmu(value); },
            "The learning rate cmu given to the constructor, or None for the default; in [0, 1] "
            "(ValueError). With c1 given, c1 + cmu must be at most 1 (ValueError when the "
            "optimizer is created).\n\n"
            "The value in use is Parameters.weights.cmu.")
        .def_prop_rw(
            "c1",
            [](const Settings &s) { return s.c1; },
            [](Settings &s, const std::optional<Float> value) { s.set_c1(value); },
            "The learning rate c1 given to the constructor, or None for the default; in [0, 1] "
            "(ValueError). With cmu given, c1 + cmu must be at most 1 (ValueError when the "
            "optimizer is created).\n\n"
            "The value in use is Parameters.weights.c1.")
        .def_prop_rw(
            "damps",
            [](const Settings &s) { return s.damps; },
            [](Settings &s, const std::optional<Float> value) { s.set_damps(value); },
            "The damping given to the constructor, or None for the default; positive "
            "(ValueError).\n\n"
            "The value in use is Parameters.weights.damps.")
        .def_prop_rw(
            "acov",
            [](const Settings &s) { return s.acov; },
            [](Settings &s, const std::optional<Float> value) { s.set_acov(value); },
            "The factor alpha_cov of the defaults of c1 and cmu, or None for the default 2; "
            "positive and finite (ValueError).")
        .def_prop_rw("verbose",
                     bool_getter(&Settings::verbose),
                     bool_setter(&Settings::verbose),
                     "Print a message at every restart and when a decomposition of the matrix "
                     "fails.")
        .def_ro("one_plus_one",
                &Settings::one_plus_one,
                nb::name("one_plus_one"),
                "Whether lambda0 is 1: a (1+1)-ES (read-only; see the constructor).")
        .def("__repr__", [](Settings &s) {
            return repr("Settings", [&](std::stringstream &ss) {
                ss << " dim: " << s.dim << " modules: " << to_string(s.modules);
                ss << " target: " << to_string(s.target);
                ss << " max_generations: " << to_string(s.max_generations);
                ss << " budget: " << s.budget << " sigma0: " << s.sigma0;
                ss << " mu0: " << s.mu0 << " lambda0: " << s.lambda0;
                ss << " x0: " << to_string(s.x0);
                ss << " lb: " << s.lb.transpose() << " ub: " << s.ub.transpose();
                ss << " integer_variables: " << s.integer_variables.transpose();
                ss << " cs: " << to_string(s.cs) << " cc: " << to_string(s.cc);
                ss << " cmu: " << to_string(s.cmu) << " c1: " << to_string(s.c1);
                ss << " verbose: " << s.verbose;
                const auto dim = std::to_string(s.dim);
                ss << " C0: " << (s.C0 ? "(" + dim + ", " + dim + ") array" : "None");
            });
        },
        "The settings as text.");

    nb::class_<mapping::CoordinateMapping>(
        m,
        "CoordinateMapping",
        "Maps a sample to the point that is evaluated: the integer variables are rounded.\n\n"
        "Rounding is to the nearest integer, halfway cases away from zero. A coordinate inside "
        "the bounds [lb, ub] is rounded to the nearest integer in [ceil(lb), floor(ub)], so it "
        "stays inside; a coordinate outside them is rounded to the nearest integer (#79).")
        .def(
            "__init__",
            [](mapping::CoordinateMapping *self, std::vector<int> idx, size_t d) {
                new (self) mapping::CoordinateMapping(*to_indices(idx), d);
            },
            nb::arg("integer_variables"),
            nb::arg("d"),
            "A mapping that rounds the given variables.\n\n"
            "Parameters\n"
            "----------\n"
            "integer_variables : list of int\n"
            "    The indices of the integer variables, each in 0..d-1 (otherwise ValueError).\n"
            "d : int\n"
            "    The dimension.")
        .def(
            "transform",
            [](const mapping::CoordinateMapping &self,
               const Vector &x,
               const std::optional<Vector> &lb,
               const std::optional<Vector> &ub) {
                const auto n = x.size();
                return self.transform(
                    x,
                    lb.value_or(Vector::Constant(n, -std::numeric_limits<Float>::infinity())),
                    ub.value_or(Vector::Constant(n, std::numeric_limits<Float>::infinity())));
            },
            nb::arg("x"),
            nb::arg("lb") = nb::none(),
            nb::arg("ub") = nb::none(),
            "A copy of the point x with the integer variables rounded.\n\n"
            "Parameters\n"
            "----------\n"
            "x : numpy.ndarray\n"
            "    The point, shape (d,).\n"
            "lb, ub : numpy.ndarray, optional\n"
            "    The bounds, shape (d,); an integer variable of x inside them is rounded to the "
            "nearest integer inside them. Default None: no bound.\n\n"
            "Returns\n"
            "-------\n"
            "numpy.ndarray\n"
            "    The rounded point, shape (d,).");

    nb::class_<Parameters>(
        main,
        "Parameters",
        "The state of an optimizer: settings, module objects, populations and statistics.\n\n"
        "ModularCMAES works on a Parameters object (ModularCMAES.p). Every module choice of "
        "Settings.modules is an object here (adaptation, mutation, selection, sampler, bounds, "
        "...), which can be inspected and, for experiments, changed. Every Parameters object has "
        "its own random generator: seeded with Settings.seed, or a copy of the global generator "
        "(set_seed) at creation.")
        .def(nb::init<size_t>(),
             nb::arg("dimension"),
             "Parameters with the default settings for this dimension (Settings(dimension)).")
        .def(nb::init<Settings>(),
             nb::arg("settings"),
             "Parameters for the given settings (copied).")
        .def("adapt",
             &Parameters::adapt,
             "The adaptation at the end of a generation.\n\n"
             "Updates the evolution paths, the step size (clipped to the tolerances of the "
             "MinSigma and MaxSigma restart criteria when constants.clip_sigma is set), the "
             "matrix of the matrix adaptation and the restart criteria, then increments "
             "stats.t. ModularCMAES.adapt calls it.")
        .def("resize_population",
             &Parameters::resize_population,
             nb::arg("new_mu"),
             nb::arg("new_lambda"),
             "Change the population size now.\n\n"
             "Sets mu and lamb and resets the sampler and the weights. The populations keep their "
             "contents: the next generation compares with this one (elitism, MSR, PSR, "
             "sequential selection) and reserves the columns it needs. The restart strategies do "
             "not call it: they change mu and lamb, which the reset of the restart then "
             "uses.\n\n"
             "Parameters\n"
             "----------\n"
             "new_mu : int\n"
             "    The number of parents.\n"
             "new_lambda : int\n"
             "    The number of offspring per generation.")
        .def_rw("settings", &Parameters::settings, "The Settings of this optimizer.")
        .def_rw("mu",
                &Parameters::mu,
                "The current number of parents: settings.mu0 at first, changed by IPOP and "
                "BIPOP restarts.")
        .def_rw("lamb",
                &Parameters::lambda,
                "The current number of offspring per generation (lambda): settings.lambda0 at "
                "first, changed by IPOP and BIPOP restarts.")
        // shared_ptr<Adaptation> comes back as its most-derived bound type
        .def_rw("adaptation",
                &Parameters::adaptation,
                "The matrix adaptation, a matrix_adaptation.Adaptation of the type chosen by "
                "modules.matrix_adaptation.\n\n"
                "Holds the mean m, the evolution paths and the matrix of the search "
                "distribution.")
        .def_rw("criteria",
                &Parameters::criteria,
                "The restart criteria (restart.Criteria); its list items can be changed from "
                "Python.")
        .def_rw("stats", &Parameters::stats, "The statistics of the run (Stats).")
        .def_rw("weights",
                &Parameters::weights,
                "The recombination weights and learning rates in use (Weights).")
        .def_rw("pop", &Parameters::pop, "The population of the current generation.")
        .def_rw("old_pop",
                &Parameters::old_pop,
                "The population of the previous generation (the parents for elitism).")
        .def_rw("sampler",
                &Parameters::sampler,
                "The sampler of the samples z (sampling.Sampler), built from modules.sampler, "
                "sample_transformation, mirrored and orthogonal.")
        .def_rw("mutation",
                &Parameters::mutation,
                "The mutation (mutation.Strategy, by modules.ssa): the step size sigma and its "
                "adaptation, threshold convergence, sequential selection and sampled step "
                "sizes.")
        .def_rw("selection",
                &Parameters::selection,
                "The selection (selection.Strategy): pairwise selection and elitism.")
        .def_rw("restart_strategy",
                &Parameters::restart_strategy,
                "The restart strategy (restart.Strategy, by modules.restart_strategy): the "
                "population size and step size of the next run.")
        .def_rw("repelling",
                &Parameters::repelling,
                "The repelling restarts (repelling.Repelling, by modules.repelling_type): the "
                "archive of taboo points.")
        .def_rw("bounds",
                &Parameters::bounds,
                "The bound correction (bounds.BoundCorrection, by modules.bound_correction).")
        .def_rw("center_placement",
                &Parameters::center_placement,
                "The placement of the mean after a restart (center.Placement, by "
                "modules.center_placement).")
        .def_rw("coordinate_mapping",
                &Parameters::coordinate_mapping,
                "The rounding of the integer variables (CoordinateMapping).");
}

namespace
{
using RowMatrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//! Objective values from Python (an array, a sequence or a number) as a vector of float64
Vector to_values(nb::handle f)
{
    static const nb::object asarray = nb::module_::import_("numpy").attr("asarray");
    return nb::cast<Vector>(asarray(f, nb::arg("dtype") = "float64").attr("reshape")(-1));
}

//! Points from Python, (k, d) (or (d,) for one point), as the columns of a d x k matrix
Matrix to_points(nb::handle X)
{
    static const nb::object asarray = nb::module_::import_("numpy").attr("asarray");
    nb::object a = asarray(X, nb::arg("dtype") = "float64");
    const auto ndim = nb::cast<size_t>(a.attr("ndim"));
    if (ndim == 0 || ndim > 2)
        throw nb::value_error("tell(f, X): X must be a (k, d) array of points");
    if (ndim == 1)
        a = a.attr("reshape")(1, -1);
    return nb::cast<RowMatrix>(a).transpose();
}

//! ask() as a new (k, d) array: row i is point i
RowMatrix ask(ModularCMAES &c) { return c.ask().transpose(); }

bool tell(ModularCMAES &c, nb::handle f, nb::handle X)
{
    const Vector values = to_values(f);
    if (X.is_none())
        return c.tell(values);
    const Matrix points = to_points(X);
    return c.tell(values, points);
}

//! One generation; a vectorized objective gets every batch of ask() as a (k, d) array
bool step(ModularCMAES &c, nb::handle objective, const bool vectorized)
{
    if (!vectorized)
    {
        auto f = nb::cast<FunctionType>(objective);
        return c.step(f);
    }
    while (!c.tell(to_values(objective(nb::cast(ask(c), nb::rv_policy::move)))))
        ;
    return !c.break_conditions();
}

void run(ModularCMAES &c, nb::handle objective, const bool vectorized)
{
    if (!vectorized)
    {
        auto f = nb::cast<FunctionType>(objective);
        c(f);
        return;
    }
    // As ModularCMAES::operator(): the break conditions are checked before every generation
    if (c.break_conditions())
        return;
    while (step(c, objective, true))
        ;
}
} // namespace

void define_cmaes(nb::module_ &m)
{
    nb::class_<ModularCMAES>(
        m,
        "ModularCMAES",
        "The modular CMA-ES.\n\n"
        "Create it from Settings (or from Parameters), then run it in one of two ways:\n\n"
        "- with an objective function: run() (or calling the object) until a break condition "
        "holds, step() for one generation, or mutate(), select(), recombine() and adapt() for "
        "the steps of a generation;\n"
        "- ask-tell, for an objective that is evaluated outside the optimizer (in parallel, on "
        "a cluster): ask() returns the points to evaluate, tell() takes their objective "
        "values.\n\n"
        "step() and run() are loops over ask() and tell(), so both ways give the same results, "
        "bit for bit. The state, including the statistics and the best solution "
        "(p.stats.global_best), is in p.")
        .def(nb::init<std::shared_ptr<parameters::Parameters>>(),
             nb::arg("parameters"),
             "An optimizer that works on the given Parameters (shared, not copied: p is "
             "parameters).")
        .def(nb::init<size_t>(),
             nb::arg("dimension"),
             "An optimizer with the default settings for this dimension: the same as "
             "ModularCMAES(Settings(dimension)).")
        .def(nb::init<parameters::Settings>(),
             nb::arg("settings"),
             "An optimizer for the given Settings (copied).")
        .def(
            "__init__",
            [](ModularCMAES *, const nb::callable &, const nb::args &, const nb::kwargs &) {
                throw nb::type_error(
                    "ModularCMAES(fitness_func, d, ...) is the signature of the pure-Python "
                    "implementation, which modcma 2.0 no longer contains. Use "
                    "modcma.fmin(func, x0, sigma0, budget), or "
                    "modcma.ModularCMAES(modcma.Settings(d, ...)).run(func); see "
                    "https://modularcmaes.readthedocs.io/en/latest/migration/. The pure-Python "
                    "implementation remains in modcma 1.x: pip install \"modcma<2\".");
            },
            nb::arg("fitness_func"),
            nb::arg("args"),
            nb::arg("kwargs"),
            "The signature of modcma 1.x (the pure-Python implementation): raises TypeError.")
        .def_prop_ro("center_of_mass",
                     &ModularCMAES::center_of_mass,
                     "The center of mass, (d,) array: the mean of the search distribution as a "
                     "point of the objective (integer variables rounded; inside the bounds when a "
                     "bound correction is used), pycma's `xfavorite`. The current mean: during a "
                     "run it moves with every generation.\n\n"
                     "Every run ends with its evaluation, recorded in `p.stats.centers` (#111). "
                     "With a noisy objective the value of `p.stats.global_best` is biased low (it "
                     "was selected for being low); the mean averages over the population and, "
                     "with a large population, is often closer to the optimum.")
        .def("recombine",
             &ModularCMAES::recombine,
             "Move the mean to the weighted mean of the mu best individuals.\n\n"
             "m_old = m, then m = m_old + sum_k positive[k] (X[:, k] - m_old) over the first mu "
             "columns of the (sorted) population; with the utility weights of NES/xNES, over "
             "every individual. The third step of a generation, after select(). Raises "
             "RuntimeError while the points of a generation wait for their values (after ask(), "
             "or an interrupted mutate() or step()).")
        .def(
            "mutate",
            [](ModularCMAES &c, FunctionType objective) { c.mutate(objective); },
            nb::arg("objective"),
            "Sample and evaluate a generation: the first step of a generation.\n\n"
            "Restarts first when a restart criterion holds (the run that ends evaluates its "
            "center of mass with objective). Then select(), recombine() and adapt() complete the "
            "generation. Every call samples a new generation, as in 1.x; only a generation that "
            "ask() started is completed instead (its points evaluated with objective). When the "
            "optimizer stops (stop_conditions()), it evaluates the final center of mass instead "
            "of a generation, and select(), recombine() and adapt() then do nothing: the 1.x "
            "loop `while not es.break_conditions(): es.mutate(f); es.select(); ...` ends as "
            "run() does.\n\n"
            "Parameters\n"
            "----------\n"
            "objective : callable\n"
            "    The objective function, called with a (d,) array, returns a float.")
        .def("select",
             &ModularCMAES::select,
             "Select the individuals of the evaluated generation: the second step.\n\n"
             "Applies pairwise selection (mirrored PAIRWISE) and elitism (the parents join the "
             "offspring), sorts the population by f, keeps lambda individuals and updates the "
             "statistics (current_avg, current_best, global_best). Raises RuntimeError while the "
             "points of a generation wait for their values (after ask(), or an interrupted "
             "mutate() or step()).")
        .def("adapt",
             &ModularCMAES::adapt,
             "Adapt the search distribution: the last step of a generation.\n\n"
             "Updates the evolution paths, the step size, the matrix and the restart criteria, "
             "and increments p.stats.t (Parameters.adapt). Raises RuntimeError while the points "
             "of a generation wait for their values (after ask(), or an interrupted mutate() or "
             "step()).")
        .def("step",
             &step,
             nb::sig("def step(self, objective: collections.abc.Callable[[numpy.typing.NDArray["
                     "numpy.float64]], typing.Any], *, vectorized: bool = False) -> bool"),
             nb::arg("objective"),
             nb::kw_only(),
             nb::arg("vectorized") = false,
             "Run one generation (with a restart first, when one is due).\n\n"
             "When the optimizer stops (stop_conditions()), the call evaluates the final center "
             "of mass instead, one evaluation, and returns False.\n\n"
             "Parameters\n"
             "----------\n"
             "objective : callable\n"
             "    The objective function. Called with a (d,) array, it returns a float; with "
             "vectorized=True it is called with every batch of points as a (k, d) array and "
             "returns k values.\n"
             "vectorized : bool, default False\n"
             "    Whether objective evaluates a batch of points in one call.\n\n"
             "Returns\n"
             "-------\n"
             "bool\n"
             "    False when the optimizer has stopped (see break_conditions), True "
             "otherwise.")
        .def("__call__",
             &run,
             nb::sig("def __call__(self, objective: collections.abc.Callable[[numpy.typing."
                     "NDArray[numpy.float64]], typing.Any], *, vectorized: bool = False) -> None"),
             nb::arg("objective"),
             nb::kw_only(),
             nb::arg("vectorized") = false,
             "Run until a break condition holds: the same as run().")
        .def("run",
             &run,
             nb::sig("def run(self, objective: collections.abc.Callable[[numpy.typing.NDArray["
                     "numpy.float64]], typing.Any], *, vectorized: bool = False) -> None"),
             nb::arg("objective"),
             nb::kw_only(),
             nb::arg("vectorized") = false,
             "Run generations while no break condition holds.\n\n"
             "The break conditions are checked before every generation, the first one too: the "
             "run never exceeds the budget, and an optimizer that has stopped does not run again. "
             "Every run ends with the evaluation of its center of mass, the last one too (#111). "
             "The result is in p.stats (global_best, evaluations, centers). A vectorized "
             "objective saves a Python call per evaluation.\n\n"
             "Parameters\n"
             "----------\n"
             "objective : callable\n"
             "    The objective function. Called with a (d,) array, it returns a float; with "
             "vectorized=True it is called with every batch of points as a (k, d) array and "
             "returns k values.\n"
             "vectorized : bool, default False\n"
             "    Whether objective evaluates a batch of points in one call.")
        .def("ask",
             &ask,
             "The points to evaluate next, a new (k, d) array (row i is point i).\n\n"
             "Starts a generation when none is in progress; until tell(), it returns the same "
             "points. k >= 1 is the largest batch that can be evaluated in any order: usually "
             "lambda (lambda + 2 with TPA step-size adaptation). With sequential selection, the "
             "points up to the first one after which the generation may stop, then one point at "
             "a time (a pair with pairwise mirroring). Every run ends with its center of mass, "
             "one point: a restart asks for it first and, with repelling restarts, for the "
             "points of the hill-valley tests, one at a time; when the optimizer stops "
             "(stop_conditions()), the last ask() returns it, and break_conditions() holds after "
             "its tell().")
        .def("tell",
             &tell,
             nb::sig("def tell(self, f: numpy.typing.ArrayLike, X: numpy.typing.ArrayLike | None "
                     "= None) -> bool"),
             nb::arg("f"),
             nb::arg("X") = nb::none(),
             "Give the objective values of the points of ask().\n\n"
             "Completing a generation runs select(), recombine() and adapt(); then "
             "break_conditions() tells whether to stop.\n\n"
             "Parameters\n"
             "----------\n"
             "f : array_like\n"
             "    (k,) the objective values of the k points of ask(), in the same order.\n"
             "X : array_like, optional\n"
             "    (k, d), or (d,) for one point: the points that were evaluated, when they "
             "differ from the asked ones (repaired or injected solutions). A row that differs "
             "from the asked point replaces it; for an individual of the population, its y and "
             "z are derived from the row with its step size, as for elitism.\n\n"
             "Returns\n"
             "-------\n"
             "bool\n"
             "    True when this completed a generation (or the final center of mass, when the "
             "optimizer stops), False when ask() has more points of this generation.\n\n"
             "Raises\n"
             "------\n"
             "RuntimeError\n"
             "    No points are pending (call ask() first).\n"
             "ValueError\n"
             "    f or X does not match the points of ask().")
        .def_prop_ro("has_pending",
                     &ModularCMAES::has_pending,
                     "Whether ask() has points that wait for tell().")
        .def("break_conditions",
             &ModularCMAES::break_conditions,
             "Whether the optimizer has stopped; meaningful between generations (after tell() "
             "returned True).\n\n"
             "True when stop_conditions() hold and the final center of mass has been evaluated "
             "(the last ask() and tell(), or step()). False again when the stop conditions no "
             "longer hold, after the budget was raised: the run continues.")
        .def("stop_conditions",
             &ModularCMAES::stop_conditions,
             "Whether the optimizer should stop: then the next ask() (or step()) evaluates the "
             "final center of mass, and break_conditions() holds after it.\n\n"
             "True when one of these holds: p.stats.t reached settings.max_generations; the "
             "best objective value is at most settings.target; the next generation (lambda "
             "points, and the 2 extra points of TPA) would leave no evaluation of "
             "settings.budget for the final center of mass; a restart is due, but its center, "
             "the smallest first generation of the new run and its center of mass would exceed "
             "settings.budget; or restart_strategy is STOP and a restart criterion holds.")
        .def_ro("p",
                &ModularCMAES::p,
                "The Parameters: the state of the optimizer (settings, modules, population, "
                "statistics).");
}

void define_es(nb::module_ &main)
{
    auto m = main.def_submodule(
        "es",
        "Two simple evolution strategies, as baselines.\n\n"
        "OnePlusOneES is a (1+1)-ES with a success rule, MuCommaLambdaES a (mu, lambda)-ES with "
        "self-adapted coordinate-wise step sizes. Of the Modules they use only the sampler "
        "(sampler, sample_transformation, mirrored, orthogonal) and bound_correction, with the "
        "fixed box [-5, 5]^d. They draw from the global random generator (set_seed).");
    using namespace es;
    nb::class_<OnePlusOneES>(
        m,
        "OnePlusOneES",
        "A (1+1)-ES with a success rule for the step size.\n\n"
        "Every step samples x1 = x + sigma z, evaluates it and keeps it when it is better than "
        "x. sigma is multiplied by exp(0.8)**decay after a success and by exp(-0.2)**decay "
        "otherwise, with decay = 1 / sqrt(d + 1).")
        .def(nb::init<size_t, Vector, Float, Float, size_t, Float, parameters::Modules>(),
             nb::arg("d"),
             nb::arg("x0"),
             nb::arg("f0"),
             nb::arg("sigma0") = 1.0,
             nb::arg("budget") = 10'000,
             nb::arg("target") = 1e-8,
             nb::arg("modules") = parameters::Modules(),
             "A (1+1)-ES that starts from x0.\n\n"
             "Parameters\n"
             "----------\n"
             "d : int\n"
             "    The dimension.\n"
             "x0 : numpy.ndarray\n"
             "    (d,) the starting point.\n"
             "f0 : float\n"
             "    The objective value of x0 (not evaluated by the ES; it counts as the first "
             "evaluation, t = 1).\n"
             "sigma0 : float, default 1.0\n"
             "    The initial step size.\n"
             "budget : int, default 10000\n"
             "    The maximum number of evaluations, including x0.\n"
             "target : float, default 1e-8\n"
             "    Stop when f is at most target.\n"
             "modules : Modules, default Modules()\n"
             "    The sampler and the bound correction (RESAMPLE resamples up to 5 d times).")
        .def(
            "__call__",
            [](OnePlusOneES &es, FunctionType objective) { es(objective); },
            "Run steps until t reaches budget or f is at most target.\n\n"
            "The argument is the objective function, called with a (d,) array.")
        .def(
            "step",
            [](OnePlusOneES &es, FunctionType objective) { es.step(objective); },
            "One step: sample, evaluate, select and adapt sigma (one evaluation).\n\n"
            "The argument is the objective function, called with a (d,) array.")
        .def("sample",
             &OnePlusOneES::sample,
             "A candidate x + sigma z, bound corrected; not evaluated.")
        .def_rw("d", &OnePlusOneES::d, "The dimension.")
        .def_rw("sigma", &OnePlusOneES::sigma, "The step size.")
        .def_rw("decay",
                &OnePlusOneES::decay,
                "The exponent of the step-size update, 1 / sqrt(d + 1).")
        .def_rw("x", &OnePlusOneES::x, "(d,) array: the current (best) point.")
        .def_rw("f", &OnePlusOneES::f, "The objective value of x.")
        .def_rw("t", &OnePlusOneES::t, "The number of evaluations, including x0.")
        .def_rw("budget", &OnePlusOneES::budget, "The maximum number of evaluations.")
        .def_rw("target", &OnePlusOneES::target, "Stop when f is at most target.")
        .def_rw("sampler",
                &OnePlusOneES::sampler,
                "The sampler of z (sampling.Sampler), built from the modules.")
        .def_rw("rejection_sampling",
                &OnePlusOneES::rejection_sampling,
                "Whether points outside the bounds are resampled (bound_correction RESAMPLE).")
        .def_rw("corrector",
                &OnePlusOneES::corrector,
                "The bound correction (bounds.BoundCorrection), by modules.bound_correction.");

    nb::class_<MuCommaLambdaES>(
        m,
        "MuCommaLambdaES",
        "A (mu, lambda)-ES with self-adapted coordinate-wise step sizes.\n\n"
        "lambda = 5 d offspring and mu = lambda // 4 parents. Offspring i gets the step sizes "
        "S_i = sigma * exp(tau_i N(0, I)) * exp(tau N(0, 1)) (elementwise) and the point "
        "m + S_i * z. The new m and sigma are the means of the mu best points and of their step "
        "sizes.")
        .def(nb::init<size_t, Vector, Float, size_t, Float, parameters::Modules>(),
             nb::arg("d"),
             nb::arg("x0"),
             nb::arg("sigma0") = 1.0,
             nb::arg("budget") = 10'000,
             nb::arg("target") = 1e-8,
             nb::arg("modules") = parameters::Modules(),
             "A (mu, lambda)-ES with its mean at x0.\n\n"
             "Parameters\n"
             "----------\n"
             "d : int\n"
             "    The dimension.\n"
             "x0 : numpy.ndarray\n"
             "    (d,) the initial mean (not evaluated).\n"
             "sigma0 : float, default 1.0\n"
             "    The initial step size of every coordinate.\n"
             "budget : int, default 10000\n"
             "    The maximum number of evaluations (checked before each generation).\n"
             "target : float, default 1e-8\n"
             "    Stop when f_min is at most target.\n"
             "modules : Modules, default Modules()\n"
             "    The sampler and the bound correction (RESAMPLE resamples up to 5 d times).")
        .def(
            "__call__",
            [](MuCommaLambdaES &es, FunctionType objective) { es(objective); },
            "Run generations until e reaches budget or f_min is at most target.\n\n"
            "The argument is the objective function, called with a (d,) array.")
        .def(
            "step",
            [](MuCommaLambdaES &es, FunctionType objective) { es.step(objective); },
            "One generation: lambda evaluations, selection and recombination.\n\n"
            "The argument is the objective function, called with a (d,) array.")
        .def("sample",
             &MuCommaLambdaES::sample,
             "A point m + si * z for the (d,) step sizes si, bound corrected; not evaluated.")
        .def_rw("d", &MuCommaLambdaES::d, "The dimension.")
        .def_rw("lamb", &MuCommaLambdaES::lambda, "The number of offspring, 5 d.")
        .def_rw("mu", &MuCommaLambdaES::mu, "The number of parents, lambda // 4.")
        .def_rw("sigma", &MuCommaLambdaES::sigma, "(d,) array: the step sizes of the mean.")
        .def_rw("m", &MuCommaLambdaES::m, "(d,) array: the mean.")
        .def_rw("X",
                &MuCommaLambdaES::X,
                "(d, lambda) array: the offspring of the last generation, sorted by f.")
        .def_rw("S",
                &MuCommaLambdaES::S,
                "(d, lambda) array: the step sizes of the offspring, in the order of X.")
        .def_rw("f",
                &MuCommaLambdaES::f,
                "(lambda,) array: the objective values of the offspring, sorted.")
        .def_rw("tau",
                &MuCommaLambdaES::tau,
                "The learning rate of the common step-size factor, 1 / sqrt(d).")
        .def_rw("tau_i",
                &MuCommaLambdaES::tau_i,
                "The learning rate of the coordinate-wise step-size factors, 1 / d**0.25.")
        .def_rw("mu_inv", &MuCommaLambdaES::mu_inv, "1 / mu: the recombination weight.")
        .def_rw("f_min", &MuCommaLambdaES::f_min, "The best objective value found (+inf at first).")
        .def_rw("x_min",
                &MuCommaLambdaES::x_min,
                "(d,) array: the best point found (NaN before the first generation).")
        .def_rw("t", &MuCommaLambdaES::t, "The number of generations.")
        .def_rw("e", &MuCommaLambdaES::e, "The number of evaluations.")
        .def_rw("budget", &MuCommaLambdaES::budget, "The maximum number of evaluations.")
        .def_rw("target", &MuCommaLambdaES::target, "Stop when f_min is at most target.")
        .def_rw("sampler",
                &MuCommaLambdaES::sampler,
                "The sampler of z (sampling.Sampler), built from the modules.")
        .def_rw("sigma_sampler",
                &MuCommaLambdaES::sigma_sampler,
                "The standard normal sampler of the coordinate-wise step-size factors.")
        .def_rw("rejection_sampling",
                &MuCommaLambdaES::rejection_sampling,
                "Whether points outside the bounds are resampled (bound_correction RESAMPLE).")
        .def_rw("corrector",
                &MuCommaLambdaES::corrector,
                "The bound correction (bounds.BoundCorrection), by modules.bound_correction.");
}
} // namespace modcma_python
