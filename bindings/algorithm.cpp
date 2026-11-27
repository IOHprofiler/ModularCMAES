// Bindings of the modules of the algorithm: selection, center placement, repelling, matrix
// adaptation, bound correction and mutation (step-size adaptation).
#include "common.hpp"

namespace modcma_python
{
namespace
{
//! The members that every matrix adaptation shows in its repr
void adaptation_repr(std::stringstream &ss, const matrix_adaptation::Adaptation &a)
{
    ss << " m: " << a.m.transpose();
    ss << " m_old: " << a.m_old.transpose();
    ss << " dm: " << a.dm.transpose();
    ss << " ps: " << a.ps.transpose();
}

//! The constructor (dimension, x0) of the matrix adaptations
template <typename Class> void adaptation_init(Class &cls)
{
    cls.def(nb::init<size_t, Vector>(),
            nb::arg("dimension"),
            nb::arg("x0"),
            "Create the adaptation with mean x0 and the initial matrix (the identity).\n\n"
            "Parameters\n"
            "----------\n"
            "dimension : int\n"
            "    The dimension d.\n"
            "x0 : numpy.ndarray\n"
            "    The initial mean m, shape (d,).");
}

//! The constructor of a step-size adaptation (all inherit the one of mutation::Strategy)
template <typename Class> void strategy_init(Class &cls)
{
    using namespace mutation;
    cls.def(nb::init<std::shared_ptr<ThresholdConvergence>,
                     std::shared_ptr<SequentialSelection>,
                     std::shared_ptr<SigmaSampler>,
                     Float>(),
            nb::arg("threshold_convergence"),
            nb::arg("sequential_selection"),
            nb::arg("sigma_sampler"),
            nb::arg("sigma0"),
            "Create the step-size adaptation (the optimizer creates it from `Modules.ssa`).\n\n"
            "Parameters\n"
            "----------\n"
            "threshold_convergence : ThresholdConvergence\n"
            "    The threshold convergence (or `NoThresholdConvergence`).\n"
            "sequential_selection : SequentialSelection\n"
            "    The sequential selection (or `NoSequentialSelection`).\n"
            "sigma_sampler : SigmaSampler\n"
            "    The sampler of the step sizes (or `NoSigmaSampler`).\n"
            "sigma0 : float\n"
            "    The initial step size.");
}

constexpr const char *repr_doc = "A summary of the state (mean, paths and matrices).";

//! The parameters of the updates of the evolution paths (a macro: adjacent literals)
#define ADAPT_PARAMS_DOC                                                                          \
    "Parameters\n"                                                                               \
    "----------\n"                                                                               \
    "pop : Population\n"                                                                         \
    "    The selected population (sorted, best first).\n"                                        \
    "weights : Weights\n"                                                                        \
    "    The recombination weights and learning rates (`es.p.weights`).\n"                       \
    "stats : Stats\n"                                                                            \
    "    The statistics of the run (`es.p.stats`).\n"                                            \
    "settings : Settings\n"                                                                      \
    "    The settings (`es.p.settings`)."
} // namespace

void define_selection(nb::module_ &main)
{
    auto m = main.def_submodule(
        "selection",
        "Selection of the evaluated population: pairwise selection, elitism and sorting.\n\n"
        "The optimizer holds a `Strategy` as `es.p.selection`; it is built from "
        "`Modules.mirrored` and `Modules.elitist`.");
    using namespace selection;
    nb::class_<Elitsm>(
        m,
        "Elitism",
        "Elitist (mu + lambda) selection: the parents compete with the offspring.\n\n"
        "Selected by `Modules.elitist = True` (always on for a (1+1)-ES); "
        "`NoElitism` otherwise. From the second generation on, the mu parents (the mu best "
        "of the previous selection, the first columns of `es.p.old_pop`) are appended to the "
        "offspring before sorting, so the best points found so far survive. Their y and z "
        "are recomputed in the current distribution with the current step size, "
        "y = (x - m) / sigma and z = A^-1 y, as for a repaired point (the framework paper, "
        "Sec. 4.1). Can speed up convergence on unimodal functions, at the cost of "
        "diversity.")
        .def(nb::init<>(), "Create the elitist selection.")
        .def("__call__",
             &Elitsm::operator(),
             nb::arg("parameters"),
             "Append the mu parents to the current population (from the second generation "
             "on).\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`); its `pop` is changed in place.");

    nb::class_<NoElitsm, Elitsm>(
        m, "NoElitism", "No elitism, (mu, lambda) selection: only the offspring are selected "
                       "(the default).")
        .def(nb::init<>(), "Create the non-elitist selection.")
        .def("__call__",
             &NoElitsm::operator(),
             nb::arg("parameters"),
             "Do nothing.\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`), unchanged.");

    nb::class_<Pairwise>(m,
                         "Pairwise",
                         "Pairwise selection for mirrored sampling (Auger, Brockhoff and Hansen "
                         "2011): of each mirrored pair only the better point is ranked.\n\n"
                         "Selected by `Modules.mirrored = Mirror.PAIRWISE`; `NoPairwise` "
                         "otherwise. The better point of each pair (offspring 2i and 2i + 1) is "
                         "ranked with the unpaired points and the parents of elitism; the losers "
                         "follow them, sorted by f, with their real values "
                         "(`Population.n_ranked`). So the two mirrored steps do not cancel each "
                         "other in the recombination, and the losers get no negative weights in "
                         "the active update. Settings makes lambda even with PAIRWISE. Before "
                         "2.0 the losers got f = +inf (#75).")
        .def(nb::init<>(), "Create the pairwise selection.")
        .def("sort",
             &Pairwise::sort,
             nb::arg("population"),
             nb::arg("n_offspring"),
             "Sort the population: the ranked individuals by f, then the losers by f.\n\n"
             "Sets population.n_ranked.\n\n"
             "Parameters\n"
             "----------\n"
             "population : Population\n"
             "    Changed in place.\n"
             "n_offspring : int\n"
             "    The first n_offspring individuals are the offspring, in their pairs (2i, "
             "2i + 1); an odd last one is unpaired. The individuals after them (the parents of "
             "elitism) are ranked.")
        .def("__call__",
             &Pairwise::operator(),
             nb::arg("parameters"),
             "sort(parameters.pop, parameters.pop.n): every individual is an offspring.\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`); `pop` is changed in place.");

    nb::class_<NoPairwise, Pairwise>(
        m,
        "NoPairwise",
        "No pairwise selection (the default unless `Mirror.PAIRWISE`): `sort` sorts the "
        "population by f.")
        .def(nb::init<>(), "Create the selection without pairs.")
        .def("sort",
             &NoPairwise::sort,
             nb::arg("population"),
             nb::arg("n_offspring"),
             "Sort the population by f (`Population.sort`); every individual is ranked.\n\n"
             "Parameters\n"
             "----------\n"
             "population : Population\n"
             "    Changed in place.\n"
             "n_offspring : int\n"
             "    Not used.")
        .def("__call__",
             &NoPairwise::operator(),
             nb::arg("parameters"),
             "Sort parameters.pop by f. Before 2.0 this did nothing: the selection sorted "
             "afterwards.\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`); `pop` is changed in place.");

    nb::class_<Strategy>(m,
                         "Strategy",
                         "The selection of a generation: elitism, then sorting (with pairwise "
                         "selection).\n\n"
                         "`select` applies `elitism` (the parents join the offspring), sorts "
                         "the population with `pairwise` (by objective value; with "
                         "`Mirror.PAIRWISE` the losers of the mirrored pairs last), keeps the "
                         "first lambda and updates `es.p.stats` (the best solution and the mean "
                         "objective value).")
        .def(nb::init<parameters::Modules>(),
             nb::arg("modules"),
             "Create the selection that the modules ask for.\n\n"
             "Parameters\n"
             "----------\n"
             "modules : Modules\n"
             "    The modules; `mirrored` and `elitist` choose `pairwise` and `elitsm`.")
        .def("select",
             &Strategy::select,
             nb::arg("parameters"),
             "Select the evaluated population in place.\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`); `pop` is sorted (best first; with "
             "PAIRWISE the losers of the pairs last) and cut to lambda individuals, `stats` is "
             "updated.")
        .def_rw("pairwise",
                &Strategy::pairwise,
                "The pairwise selection: `Pairwise` with `Mirror.PAIRWISE`, `NoPairwise` "
                "otherwise.")
        .def_rw("elitism",
                &Strategy::elitsm,
                "The elitism: `Elitism` with `Modules.elitist`, `NoElitism` otherwise.")
        .def_prop_rw(
            "elitsm",
            [](const Strategy &s) {
                warn_deprecated("Strategy.elitsm is deprecated; use Strategy.elitism");
                return s.elitsm;
            },
            [](Strategy &s, const std::shared_ptr<Elitsm> &e) {
                warn_deprecated("Strategy.elitsm is deprecated; use Strategy.elitism");
                s.elitsm = e;
            },
            "Deprecated: the old (misspelled) name of `elitism`.");

    // The old, misspelled class names of 1.x: available through the module's __getattr__, with a
    // DeprecationWarning
    m.def(
        "__getattr__",
        [m](const std::string &name) -> nb::object {
            if (name == "Elitsm" || name == "NoElitsm")
            {
                const std::string now = name == "Elitsm" ? "Elitism" : "NoElitism";
                const std::string message =
                    "selection." + name + " is deprecated; use selection." + now;
                warn_deprecated(message.c_str());
                return m.attr(now.c_str());
            }
            const std::string module = nb::cast<std::string>(m.attr("__name__"));
            throw nb::attribute_error(
                ("module '" + module + "' has no attribute '" + name + "'").c_str());
        },
        nb::arg("name"),
        "The deprecated names of 1.x: `Elitsm` and `NoElitsm` (now `Elitism`, `NoElitism`).");
}

void define_center_placement(nb::module_ &main)
{
    auto m = main.def_submodule(
        "center",
        "Center placement: where the mean of the search distribution is placed at a restart.\n\n"
        "The optimizer holds one as `es.p.center_placement`, selected by "
        "`Modules.center_placement` (`options.CenterPlacement`). It is applied at every restart; "
        "the first run starts at `Settings.x0`, or at the center of the bounds when x0 is not "
        "given.\n\n"
        "UNIFORM, NOVELTY_WEIGHTED and MAXIMIN_TABOO sample in the box [lb, ub]. In a coordinate "
        "with an infinite bound they sample uniformly in [x0 - 2 sigma0, x0 + 2 sigma0], with x0 "
        "`Settings.x0` or `Settings.center`; the interval is shifted to start at a finite lb, or "
        "to end at a finite ub, when it would cross it. NOVELTY_WEIGHTED and MAXIMIN_TABOO then "
        "divide the distances in that coordinate by the width of the interval, 4 sigma0.");
    using namespace center;
    nb::class_<Placement>(
        m, "Placement", "Base class of the center placements: sets the mean `es.p.adaptation.m`.")
        .def("__call__",
             &Placement::operator(),
             nb::arg("parameters"),
             "Set the mean of the search distribution.\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`); `adaptation.m` is set.");

    nb::class_<X0, Placement>(m,
                              "X0",
                              "The mean is `Settings.x0`, or the center of the box [lb, ub] (the "
                              "origin without finite bounds) when x0 is not given "
                              "(`CenterPlacement.X0`, the default).")
        .def(nb::init<>(), "Create the placement.");
    nb::class_<Uniform, Placement>(m,
                                   "Uniform",
                                   "The mean is a uniform random point in the box [lb, ub] "
                                   "(`CenterPlacement.UNIFORM`); in a coordinate with an "
                                   "infinite bound, in x0 +- 2 sigma0 (see the module).")
        .def(nb::init<>(), "Create the placement.");
    nb::class_<Zero, Placement>(m, "Zero", "The mean is the origin (`CenterPlacement.ZERO`).")
        .def(nb::init<>(), "Create the placement.");
    nb::class_<Center, Placement>(m,
                                  "Center",
                                  "The mean is the center of the box, (lb + ub) / 2, or the origin "
                                  "without finite bounds (`CenterPlacement.CENTER`).")
        .def(nb::init<>(), "Create the placement.");
    nb::class_<MaximinTaboo, Placement>(
        m,
        "MaximinTaboo",
        "The mean is placed as far as possible from the taboo points of repelling "
        "(`CenterPlacement.MAXIMIN_TABOO`).\n\n"
        "Of 512 uniform random points in the box, the one whose nearest taboo point of "
        "`es.p.repelling.archive` is farthest away (in coordinates divided by the widths of the "
        "box); a uniform random point while the archive is empty. For use with repelling "
        "restarts. In a coordinate with an infinite bound, the candidates are drawn in x0 +- 2 "
        "sigma0 and the distances divided by 4 sigma0 (see the module).")
        .def(nb::init<>(), "Create the placement.");
    nb::class_<NoveltyWeighted, Placement>(
        m,
        "NoveltyWeighted",
        "The mean is a random point that favours regions away from the taboo points of "
        "repelling (`CenterPlacement.NOVELTY_WEIGHTED`).\n\n"
        "Of 512 uniform random points in the box, one is drawn with probability proportional "
        "to the square of the distance to its nearest taboo point (in coordinates divided by the "
        "widths of the box); a uniform random point while the archive is empty. For use with "
        "repelling restarts; needs finite bounds.")
        .def(nb::init<>(), "Create the placement.");
}

void define_repelling(nb::module_ &main)
{
    using namespace repelling;
    auto m = main.def_submodule(
        "repelling",
        "Repelling restarts: an archive of the optima of earlier runs (taboo points), and the "
        "rejection of samples close to them.\n\n"
        "The optimizer holds one as `es.p.repelling`, selected by `Modules.repelling_type` "
        "(`options.RepellingType`); `Modules.repelling_metric` (`options.RepellingMetric`) sets "
        "the distance. Helps restarts find new basins of attraction on multimodal functions. "
        "de Nobel et al. (2024).");

    nb::class_<TabooPoint>(
        m,
        "TabooPoint",
        "A taboo point: an optimum found by an earlier run, with a radius around it in which "
        "samples are rejected.\n\n"
        "The distance of a sample to it is measured in the metric of "
        "`Modules.repelling_metric` (EUCLIDEAN; CURRENT: the Mahalanobis distance in the current "
        "search distribution; ARCHIVED: `metric`) and divided by the current sigma. A sample is "
        "rejected when this distance is less than shrinkage^attempts * radius, with attempts "
        "the number of samples rejected so far in the population.")
        .def(nb::init<const Solution &, Float, const Matrix &>(),
             nb::arg("solution"),
             nb::arg("radius"),
             nb::arg("metric") = Matrix(),
             "Create a taboo point with an unbounded radius (as `CoverageRepelling` does).\n\n"
             "Parameters\n"
             "----------\n"
             "solution : Solution\n"
             "    The optimum (its x and objective value y).\n"
             "radius : float\n"
             "    The rejection radius, in units of sigma.\n"
             "metric : numpy.ndarray\n"
             "    For `RepellingMetric.ARCHIVED`: A^-1 of the run that found the point, scaled "
             "to unit determinant, shape (d, d). Empty (the default) for the other metrics.")
        .def(nb::init<const Solution &, Float, Float, Float, const Matrix &>(),
             nb::arg("solution"),
             nb::arg("radius"),
             nb::arg("min_radius"),
             nb::arg("max_radius"),
             nb::arg("metric") = Matrix(),
             "Create a taboo point whose radius stays in [min_radius, max_radius] (as "
             "`AdaptiveRepelling` does).\n\n"
             "Parameters\n"
             "----------\n"
             "solution : Solution\n"
             "    The optimum (its x and objective value y).\n"
             "radius : float\n"
             "    The rejection radius, in units of sigma.\n"
             "min_radius : float\n"
             "    The lower bound of the radius.\n"
             "max_radius : float\n"
             "    The upper bound of the radius.\n"
             "metric : numpy.ndarray\n"
             "    For `RepellingMetric.ARCHIVED`: A^-1 of the run that found the point, scaled "
             "to unit determinant, shape (d, d). Empty (the default) for the other metrics.")
        .def("distance",
             &TabooPoint::distance,
             nb::arg("x"),
             nb::arg("p"),
             "The distance from x to this point, divided by the current sigma.\n\n"
             "Parameters\n"
             "----------\n"
             "x : numpy.ndarray\n"
             "    The point, shape (d,).\n"
             "p : Parameters\n"
             "    The state of the optimizer; `settings.modules.repelling_metric` selects the "
             "metric.\n\n"
             "Returns\n"
             "-------\n"
             "float\n"
             "    The distance in units of sigma.")
        .def_rw("metric",
                &TabooPoint::metric,
                "For `RepellingMetric.ARCHIVED`: A^-1 of the run that found the point, scaled "
                "to unit determinant (only the shape of its basin), shape (d, d). Empty for the "
                "other metrics, and when A^-1 was degenerate (the Euclidean distance is used "
                "then).")
        .def("rejects",
             &TabooPoint::rejects,
             nb::arg("xi"),
             nb::arg("p"),
             nb::arg("attempts"),
             "Whether the sample xi lies within the rejection radius "
             "shrinkage^attempts * radius.\n\n"
             "Parameters\n"
             "----------\n"
             "xi : numpy.ndarray\n"
             "    The sample, shape (d,).\n"
             "p : Parameters\n"
             "    The state of the optimizer.\n"
             "attempts : int\n"
             "    The number of samples rejected so far in this population; every rejection "
             "shrinks the radius.\n\n"
             "Returns\n"
             "-------\n"
             "bool")
        .def("calculate_criticality",
             &TabooPoint::calculate_criticality,
             nb::arg("p"),
             "Compute `criticality` from the distance delta of the current mean to this "
             "point: Phi(delta + radius) - Phi(delta - radius), with Phi the standard normal "
             "CDF.\n\n"
             "Parameters\n"
             "----------\n"
             "p : Parameters\n"
             "    The state of the optimizer (the mean and sigma).")
        .def("clamp_radius",
             &TabooPoint::clamp_radius,
             "Clamp `radius` to [min_radius, max_radius].")
        .def_rw("solution", &TabooPoint::solution, "The optimum (a Solution: x, y, t, e).")
        .def_rw("radius", &TabooPoint::radius, "The rejection radius, in units of sigma.")
        .def_rw("min_radius", &TabooPoint::min_radius, "The lower bound of the radius (default 0).")
        .def_rw("max_radius",
                &TabooPoint::max_radius,
                "The upper bound of the radius (default inf).")
        .def_rw("shrinkage",
                &TabooPoint::shrinkage,
                "The factor by which the radius shrinks for every sample rejected in a "
                "population, 0.95^(1/d).")
        .def_rw("n_rep",
                &TabooPoint::n_rep,
                "The number of runs that ended in the basin of this point (1 when it is "
                "created).")
        .def_rw("criticality",
                &TabooPoint::criticality,
                "How close the current mean is to this point relative to its radius (see "
                "`calculate_criticality`); points with a criticality below 0.01 do not reject "
                "samples.")
        .def_rw("duplicate_evaluations",
                &TabooPoint::duplicate_evaluations,
                "`AdaptiveRepelling`: the evaluations of the runs that ended in this basin "
                "again.")
        .def_rw("rejected_count",
                &TabooPoint::rejected_count,
                "The samples this point rejected (reset after every population by "
                "`AdaptiveRepelling`).")
        .def_rw("checked_count",
                &TabooPoint::checked_count,
                "The samples tested against this point (reset after every population by "
                "`AdaptiveRepelling`).")
        .def_rw("last_rejection_rate",
                &TabooPoint::last_rejection_rate,
                "`AdaptiveRepelling`: rejected_count / checked_count of the last population; 0 "
                "when the point was checked fewer than `min_checks_before_local_shrink` times.")
        .def(
            "__repr__",
            [](const TabooPoint &tb) {
                return "<TabooPoint n_rep: " + std::to_string(tb.n_rep) +
                       " y: " + std::to_string(tb.solution.y) +
                       " radius: " + std::to_string(tb.radius) +
                       " criticality: " + std::to_string(tb.criticality) + ">";
            },
            "A summary: n_rep, y, radius and criticality.");

    nb::class_<Repelling>(
        m,
        "Repelling",
        "Base class of the repelling restarts: the archive of taboo points and the rejection "
        "of samples.\n\n"
        "Before each population the archive is sorted by criticality (`prepare_sampling`); "
        "each sample is tested against the points with criticality >= 0.01 (`is_rejected`) and "
        "sampled again when one rejects it. After a restart, the better of the last mean and "
        "the best solution of the run is compared with the archive by hill-valley tests (the "
        "optimizer evaluates their points, one at a time); the subclasses add it as a new "
        "taboo point or update the point "
        "whose basin it shares. The base class does not add points.")
        .def(nb::init<>(), "Create a repelling with an empty archive.")
        .def("is_rejected",
             &Repelling::is_rejected,
             nb::arg("xi"),
             nb::arg("p"),
             "Whether a critical taboo point rejects the sample xi; a rejection is counted in "
             "`attempts` and in the point.\n\n"
             "Parameters\n"
             "----------\n"
             "xi : numpy.ndarray\n"
             "    The sample, shape (d,).\n"
             "p : Parameters\n"
             "    The state of the optimizer.\n\n"
             "Returns\n"
             "-------\n"
             "bool")
        .def("prepare_sampling",
             &Repelling::prepare_sampling,
             nb::arg("p"),
             "Before a population: reset `attempts` and sort the archive by criticality.\n\n"
             "Parameters\n"
             "----------\n"
             "p : Parameters\n"
             "    The state of the optimizer.")
        .def("adapt_radii_after_sampling",
             &Repelling::adapt_radii_after_sampling,
             nb::arg("p"),
             "After a population: adapt the radii (`AdaptiveRepelling`; nothing for the "
             "others).\n\n"
             "Parameters\n"
             "----------\n"
             "p : Parameters\n"
             "    The state of the optimizer.")
        .def_rw("archive",
                &Repelling::archive,
                "The taboo points, a list of TabooPoint (sorted by criticality before each "
                "population). Reading returns a copy; assign a list to change it.")
        .def_rw("attempts",
                &Repelling::attempts,
                "The number of samples rejected in the current population.")
        .def_rw("coverage",
                &Repelling::coverage,
                "`CoverageRepelling`: the coverage factor in the formula of the radius "
                "(default 2).");

    nb::class_<CoverageRepelling, Repelling>(
        m,
        "CoverageRepelling",
        "Repelling with radii from a coverage of the search space (`RepellingType.COVERAGE`).\n\n"
        "A candidate that shares no basin with a taboo point becomes a new one; when it shares "
        "a basin, the n_rep of that point grows and the point keeps the better solution. After "
        "each update all radii are set to r = (V n_rep / (sigma0 coverage n))^(1/d) "
        "Gamma(d/2 + 1)^(1/d) / sqrt(pi d), with V the volume of the box and n the number of "
        "runs so far, so that points found more often repel more. "
        "de Nobel et al. (2024).")
        .def(nb::init<>(), "Create a coverage repelling with an empty archive.");

    nb::class_<AdaptiveRepelling, Repelling>(
        m,
        "AdaptiveRepelling",
        "Repelling with adaptive radii (`RepellingType.ADAPTIVE`).\n\n"
        "A new taboo point gets the radius initial_radius_factor / sqrt(d), bounded to "
        "[min_radius_factor, max_radius_factor] / sqrt(d). When a run ends in the basin of a "
        "point again (the first one in the archive that shares it), its radius grows by "
        "exp(grow_eta * min(w, max_waste_ratio)), with w the evaluations of that run divided by "
        "their running average (`average_restart_evaluations`: every run counted from the "
        "previous restart, the first from the start, also a run without a finite result; the "
        "average starts from the first run's length. Before 2.0 the first run counted as one "
        "evaluation and the average started at 1, #121). After each population, the radius of "
        "a critical point shrinks "
        "by shrink_factor when the share of rejected samples of the population exceeds "
        "max_rejection_rate and the point's own rejection rate exceeds "
        "local_max_rejection_rate (after at least min_checks_before_local_shrink checks).")
        .def(nb::init<>(), "Create an adaptive repelling with an empty archive.")
        .def("initial_radius",
             &AdaptiveRepelling::initial_radius,
             nb::arg("p"),
             "The radius of a new taboo point, initial_radius_factor / sqrt(d).\n\n"
             "Parameters\n"
             "----------\n"
             "p : Parameters\n"
             "    The state of the optimizer (the dimension).\n\n"
             "Returns\n"
             "-------\n"
             "float")
        .def("min_radius",
             &AdaptiveRepelling::min_radius,
             nb::arg("p"),
             "The lower bound of the radius of a new taboo point, min_radius_factor / "
             "sqrt(d).\n\n"
             "Parameters\n"
             "----------\n"
             "p : Parameters\n"
             "    The state of the optimizer (the dimension).\n\n"
             "Returns\n"
             "-------\n"
             "float")
        .def("max_radius",
             &AdaptiveRepelling::max_radius,
             nb::arg("p"),
             "The upper bound of the radius of a new taboo point, max_radius_factor / "
             "sqrt(d).\n\n"
             "Parameters\n"
             "----------\n"
             "p : Parameters\n"
             "    The state of the optimizer (the dimension).\n\n"
             "Returns\n"
             "-------\n"
             "float")
        .def_rw("initial_radius_factor",
                &AdaptiveRepelling::initial_radius_factor,
                "The radius of a new taboo point times sqrt(d) (default 0.5).")
        .def_rw("min_radius_factor",
                &AdaptiveRepelling::min_radius_factor,
                "The lower bound of a radius times sqrt(d) (default 0.05).")
        .def_rw("max_radius_factor",
                &AdaptiveRepelling::max_radius_factor,
                "The upper bound of a radius times sqrt(d) (default 2).")
        .def_rw("shrink_factor",
                &AdaptiveRepelling::shrink_factor,
                "The factor applied to a radius that shrinks (default 0.9).")
        .def_prop_rw(
            "target_rejection_rate",
            [](const AdaptiveRepelling &r) {
                warn_deprecated("AdaptiveRepelling.target_rejection_rate is deprecated: it has "
                                "no effect");
                return r.target_rejection_rate;
            },
            [](AdaptiveRepelling &r, const Float v) {
                warn_deprecated("AdaptiveRepelling.target_rejection_rate is deprecated: it has "
                                "no effect");
                r.target_rejection_rate = v;
            },
            "Deprecated: has no effect (the update shrinks radii above max_rejection_rate).")
        .def_rw("max_rejection_rate",
                &AdaptiveRepelling::max_rejection_rate,
                "The share of rejected samples of a population, rejected / (lambda + "
                "rejected), above which radii can shrink (default 0.5).")
        .def_rw("last_restart_evaluations",
                &AdaptiveRepelling::last_restart_evaluations,
                "The evaluation count at the last restart (0 before the first).")
        .def_rw("average_restart_evaluations",
                &AdaptiveRepelling::average_restart_evaluations,
                "The running average of the evaluations per run, n the evaluations since the "
                "last restart (since the start for the first run). 0 before the first restart, "
                "which sets it to n; then, at every restart, (1 - restart_eval_alpha) * average "
                "+ restart_eval_alpha * n. A positive value set before the first restart is "
                "kept as the start of the average.")
        .def_rw("restart_eval_alpha",
                &AdaptiveRepelling::restart_eval_alpha,
                "The smoothing factor of average_restart_evaluations (default 0.2).")
        .def_rw("grow_eta",
                &AdaptiveRepelling::grow_eta,
                "The growth rate of the radius of a point whose basin is found again "
                "(default 0.15).")
        .def_rw("max_waste_ratio",
                &AdaptiveRepelling::max_waste_ratio,
                "The cap on the ratio of a run's evaluations to their running average in the "
                "growth of a radius (default 3).")
        .def_prop_rw(
            "local_target_rejection_rate",
            [](const AdaptiveRepelling &r) {
                warn_deprecated("AdaptiveRepelling.local_target_rejection_rate is deprecated: it "
                                "has no effect");
                return r.local_target_rejection_rate;
            },
            [](AdaptiveRepelling &r, const Float v) {
                warn_deprecated("AdaptiveRepelling.local_target_rejection_rate is deprecated: it "
                                "has no effect");
                r.local_target_rejection_rate = v;
            },
            "Deprecated: has no effect (the update shrinks a radius above "
            "local_max_rejection_rate).")
        .def_rw("local_max_rejection_rate",
                &AdaptiveRepelling::local_max_rejection_rate,
                "The rejection rate of a point above which its radius can shrink (default 0.5).")
        .def_rw("min_checks_before_local_shrink",
                &AdaptiveRepelling::min_checks_before_local_shrink,
                "The number of samples a point must have tested in a population before its "
                "radius can shrink (default 5).");

    nb::class_<NoRepelling, Repelling>(
        m,
        "NoRepelling",
        "No repelling (`RepellingType.NONE`, the default): no sample is rejected and the "
        "archive stays empty.")
        .def(nb::init<>(), "Create the no-op repelling.");

    m.def("euclidean",
          &distance::euclidian,
          nb::arg("u"),
          nb::arg("v"),
          "The Euclidean distance ||u - v||.\n\n"
          "Parameters\n"
          "----------\n"
          "u, v : numpy.ndarray\n"
          "    The points, shape (d,).\n\n"
          "Returns\n"
          "-------\n"
          "float");
    m.def("manhattan",
          &distance::manhattan,
          nb::arg("u"),
          nb::arg("v"),
          "The Manhattan distance sum_i |u_i - v_i|.\n\n"
          "Parameters\n"
          "----------\n"
          "u, v : numpy.ndarray\n"
          "    The points, shape (d,).\n\n"
          "Returns\n"
          "-------\n"
          "float");
    m.def("mahalanobis",
          &distance::mahanolobis,
          nb::arg("u"),
          nb::arg("v"),
          nb::arg("C_inv"),
          "The Mahalanobis distance sqrt((u - v)^T C_inv (u - v)).\n\n"
          "Parameters\n"
          "----------\n"
          "u, v : numpy.ndarray\n"
          "    The points, shape (d,).\n"
          "C_inv : numpy.ndarray\n"
          "    The inverse of the covariance matrix, shape (d, d).\n\n"
          "Returns\n"
          "-------\n"
          "float");
    // The old, misspelled names of 1.x, with a DeprecationWarning
    m.def(
        "euclidian",
        [](const Vector &u, const Vector &v) {
            warn_deprecated("repelling.euclidian is deprecated; use repelling.euclidean");
            return distance::euclidian(u, v);
        },
        nb::arg("u"),
        nb::arg("v"),
        "Deprecated: the old (misspelled) name of `euclidean`.");
    m.def(
        "mahanolobis",
        [](const Vector &u, const Vector &v, const Matrix &C_inv) {
            warn_deprecated("repelling.mahanolobis is deprecated; use repelling.mahalanobis");
            return distance::mahanolobis(u, v, C_inv);
        },
        nb::arg("u"),
        nb::arg("v"),
        nb::arg("C_inv"),
        "Deprecated: the old (misspelled) name of `mahalanobis`.");

    m.def("archived_metric",
          &archived_metric,
          nb::arg("p"),
          "The metric that `RepellingMetric.ARCHIVED` stores with a new taboo point.\n\n"
          "A^-1 of the current run (for every matrix adaptation), scaled to unit determinant, "
          "shape (d, d). Empty for the other metrics, and when A^-1 is degenerate.\n\n"
          "Parameters\n"
          "----------\n"
          "p : Parameters\n"
          "    The state of the optimizer.\n\n"
          "Returns\n"
          "-------\n"
          "numpy.ndarray");
}

void define_matrix_adaptation(nb::module_ &main)
{
    using namespace matrix_adaptation;
    auto m = main.def_submodule(
        "matrix_adaptation",
        "Matrix adaptations: the mean and the shape of the search distribution.\n\n"
        "The optimizer holds one as `es.p.adaptation`, selected by `Modules.matrix_adaptation` "
        "(`options.MatrixAdaptationType`). A sample is x = m + sigma * y with y = A z, z drawn "
        "by the sampler, and A the sampling matrix of the adaptation (C = A A^T). They differ "
        "in what they learn and in their cost per generation: O(d^3) with a decomposition "
        "(COVARIANCE, COVARIANCE_NO_EIGV, CMSA), O(d^2 lambda) for MA-ES, Cholesky-CMA-ES and "
        "xNES, O(d lambda) for SEPARABLE and NONE (see docs/performance.md).");

    nb::class_<MetricCopy>(
        m,
        "Metric",
        "The metric of a search distribution: its shape without the step size, from "
        "`Adaptation.metric()`.\n\n"
        "A sample is x = m + sigma * B (d * z), z ~ N(0, I), so the samples have the covariance "
        "sigma^2 C with C = B diag(d)^2 B^T: the one-sigma ellipsoid around m has the semi-axes "
        "sigma * d_i along the columns B_i.\n\n"
        "It describes the distribution that is sampled. `CovarianceAdaptation` and "
        "`CovarianceNoEigvAdaptation` update C every generation but decompose it only every "
        "`es.p.weights.lazy_update_interval` generations, and sample with the last decomposition, "
        "so the metric can be that many generations behind their `C` (with the default settings "
        "they decompose every generation up to d = 82). `coordinate_wise_variances`, the diagonal "
        "of the latest C, can be ahead of it by as much.\n\n"
        "Strictly, the metric (the matrix of the Mahalanobis distance) is C^-1; this holds the "
        "decomposition, from which C and C^-1 follow. A copy: it does not change with the "
        "optimizer.")
        .def_prop_ro(
            "d",
            [](const MetricCopy &metric) { return metric.d; },
            "The axis lengths, the square roots of the eigenvalues of C (pycma's D), shape (d,), "
            "in the order of the columns of `B`. For the diagonal adaptations (SEPARABLE, NONE) in "
            "the order of the coordinates.")
        .def_prop_ro(
            "B",
            [](const MetricCopy &metric) { return metric.B; },
            "The eigenvectors of C as columns, shape (d, d), in the order of `d`; None when C is "
            "diagonal (SEPARABLE, NONE), where B is the identity.")
        .def_prop_ro(
            "is_diagonal",
            [](const MetricCopy &metric) { return !metric.B.has_value(); },
            "Whether C is diagonal (B is None).")
        .def(
            "eigenvalues",
            [](const MetricCopy &metric) {
                return matrix_adaptation::Metric::of(metric).eigenvalues();
            },
            "The eigenvalues of C, d**2, in the order of `d`.\n\n"
            "Returns\n"
            "-------\n"
            "numpy.ndarray\n"
            "    Shape (d,).")
        .def(
            "inv_C",
            [](const MetricCopy &metric) { return matrix_adaptation::Metric::of(metric).inv_C(); },
            "C^-1 = B diag(d)^-2 B^T, computed on call (O(d^3); a diagonal matrix when C is "
            "diagonal, where 1 / d**2 is the cheap form).\n\n"
            "On a convex-quadratic function with Hessian H, CMA-ES adapts C towards H^-1 up to a "
            "constant factor: Hansen (2016), The CMA Evolution Strategy: A Tutorial, Section 0.4, "
            "\"we assume that the optimal covariance matrix equals to the inverse Hessian matrix, "
            "up to a constant factor\" (an assumption with strong empirical support; \"a rigorous "
            "proof is missing\"). pycma uses the same relation "
            "(`inverse_hessian_scalar_correction` fits the factor). So C^-1 estimates the shape of the Hessian, not its scale; sigma "
            "does not give the scale, it follows the distance to the optimum. The relation needs a "
            "full matrix: SEPARABLE can only match a diagonal Hessian, and NONE keeps C = I.\n\n"
            "Returns\n"
            "-------\n"
            "numpy.ndarray\n"
            "    Shape (d, d).")
        .def(
            "inv_root_C",
            [](const MetricCopy &metric) {
                return matrix_adaptation::Metric::of(metric).inv_root_C();
            },
            "C^-1/2 = B diag(d)^-1 B^T, computed on call (O(d^3); a diagonal matrix when C is "
            "diagonal).\n\n"
            "Returns\n"
            "-------\n"
            "numpy.ndarray\n"
            "    Shape (d, d).")
        .def(
            "__repr__",
            [](const MetricCopy &metric) {
                return repr("Metric", [&](std::stringstream &ss) {
                    ss << " d: " << metric.d.transpose();
                    ss << " diagonal: " << !metric.B.has_value();
                });
            },
            repr_doc);

    auto adaptation = nb::class_<Adaptation>(
        m,
        "Adaptation",
        "Base class of the matrix adaptations: the mean, the evolution path of the step size "
        "and the sampling transformation y = A z.\n\n"
        "Each generation, after selection, `adapt_evolution_paths` updates dm, dz and the "
        "paths, the step-size adaptation (`es.p.mutation`) updates sigma, and `adapt_matrix` "
        "updates the matrix. `compute_y` and `invert_y` apply A and A^-1.")
        .def_rw("m",
                &Adaptation::m,
                "The mean (center) of the search distribution, shape (d,).")
        .def_rw("m_old", &Adaptation::m_old, "The mean of the previous generation, shape (d,).")
        .def_rw("dm",
                &Adaptation::dm,
                "The last step of the mean divided by the step size, (m - m_old) / sigma, "
                "shape (d,), with sigma the mean step size of the population.")
        .def_rw("ps",
                &Adaptation::ps,
                "The evolution path of the step size (conjugate evolution path), shape (d,), "
                "used by CSA: ps = (1 - cs) ps + sqrt(cs (2 - cs) mueff) dz (C^-1/2 dm in "
                "`CovarianceAdaptation`). Starts at zeros, also after a restart.")
        .def_rw("dz",
                &Adaptation::dz,
                "The weighted mean of the z of the mu selected individuals, sum_i w_i z_i "
                "(of every individual, with the utilities of UTILITY), shape (d,); empty before "
                "the first generation.")
        .def_rw("dd", &Adaptation::dd, "The dimension d, as a float.")
        .def_rw("coordinate_wise_variances",
                &Adaptation::coordinate_wise_variances,
                "The variance of each coordinate under the sampling transformation, the "
                "diagonal of C = A A^T (without sigma^2), shape (d,). Used for the lower bound on "
                "the step size of integer variables; a NaN or inf in it fails the matrix "
                "update.")
        .def(
            "metric",
            [](Adaptation &a) { return a.metric().copy(); },
            "The metric of the sampled distribution: its axis lengths d and eigenvectors B, "
            "without sigma (see `Metric`).\n\n"
            "`CovarianceAdaptation`, `SeparableAdaptation` and `NoAdaptation` keep it: it costs "
            "nothing. The others decompose C = F F^T of their sampling factor F (M, or A) when "
            "asked, O(d^3), and reuse the result while `factor_version` is unchanged; nothing is "
            "computed unless something asks. A value written from Python to their matrices is seen "
            "after the next change of the factor (an update or a restart). NaN when the "
            "factor is not finite.\n\n"
            "Returns\n"
            "-------\n"
            "Metric\n"
            "    A copy.")
        .def("keeps_metric",
             &Adaptation::keeps_metric,
             "Whether `metric()` costs nothing: True for `CovarianceAdaptation` (not "
             "COVARIANCE_NO_EIGV), `SeparableAdaptation` and `NoAdaptation`, which keep it up to "
             "date; False for the others, which decompose their matrix on request, O(d^3).\n\n"
             "The restart criteria ConditionC and NoEffectAxis read the metric every generation; "
             "they are created by default only when this is True.")
        .def_ro("factor_version",
                &Adaptation::factor_version,
                "A counter increased wherever the sampling factor changes (an update of the "
                "matrix, a decomposition, a restart): `metric()` recomputes only when it changed.")
        .def("adapt_evolution_paths",
             &Adaptation::adapt_evolution_paths,
             nb::arg("pop"),
             nb::arg("weights"),
             nb::arg("stats"),
             nb::arg("settings"),
             nb::arg("mu"),
             "Update dm, dz and the evolution paths after selection.\n\n"
             "Computes dm = (m - m_old) / sigma, dz = sum_i w_i z_i over the mu "
             "best (every point, with the utilities of `RecombinationWeights.UTILITY`), then "
             "the update of the subclass (ps, and pc where there is one). Called by "
             "`es.adapt()` before the step-size and matrix updates.\n\n"
             ADAPT_PARAMS_DOC "\n"
             "mu : int\n"
             "    The number of selected individuals.")
        .def("adapt_evolution_paths_inner",
             &Adaptation::adapt_evolution_paths_inner,
             nb::arg("pop"),
             nb::arg("weights"),
             nb::arg("stats"),
             nb::arg("settings"),
             "The update of the evolution paths of the subclass (ps, and pc "
             "where there is one), without recomputing dm and dz; the second "
             "part of `adapt_evolution_paths`.\n\n"
             ADAPT_PARAMS_DOC)
        .def("adapt_matrix",
             &Adaptation::adapt_matrix,
             nb::arg("weights"),
             nb::arg("population"),
             nb::arg("mu"),
             nb::arg("settings"),
             nb::arg("stats"),
             "Update the matrix of the search distribution from the selected population.\n\n"
             "In a (1+1)-ES (lambda = 1) the matrix is only updated after an improvement. A "
             "failed update (a failed decomposition, or a NaN or inf in the matrix) returns "
             "False; the optimizer stores it in `es.p.successful_adaptation`, which the restart "
             "criterion `UnableToAdapt` turns into a restart.\n\n"
             "Parameters\n"
             "----------\n"
             "weights : Weights\n"
             "    The recombination weights and learning rates (`es.p.weights`).\n"
             "population : Population\n"
             "    The selected population (sorted, best first).\n"
             "mu : int\n"
             "    The number of selected individuals.\n"
             "settings : Settings\n"
             "    The settings (`modules.active`: negative weights for the worst individuals).\n"
             "stats : Stats\n"
             "    The statistics; `last_update` and `n_updates` are updated.\n\n"
             "Returns\n"
             "-------\n"
             "bool\n"
             "    Whether the update succeeded.")
        .def("restart",
             &Adaptation::restart,
             nb::arg("settings"),
             "Reset the distribution for a restart: m = `settings.x0` (`settings.center`, the "
             "center of the bounds, when it is not given), the paths and the matrices to their "
             "initial values. The center placement sets m afterwards.\n\n"
             "Parameters\n"
             "----------\n"
             "settings : Settings\n"
             "    The settings.")
        .def("distance",
             &Adaptation::distance,
             nb::arg("u"),
             nb::arg("v"),
             "The Mahalanobis distance ||A^-1 (u - v)|| in the current search distribution "
             "(without sigma).\n\n"
             "Parameters\n"
             "----------\n"
             "u, v : numpy.ndarray\n"
             "    The points, shape (d,).\n\n"
             "Returns\n"
             "-------\n"
             "float")
        .def("distance_from_center",
             &Adaptation::distance_from_center,
             nb::arg("x"),
             "The distance of x from the mean, `distance(m, x)`.\n\n"
             "Parameters\n"
             "----------\n"
             "x : numpy.ndarray\n"
             "    The point, shape (d,).\n\n"
             "Returns\n"
             "-------\n"
             "float")
        .def("compute_y",
             &Adaptation::compute_y,
             nb::arg("zi"),
             "The step y = A z of a sample z of the sampler.\n\n"
             "Parameters\n"
             "----------\n"
             "zi : numpy.ndarray\n"
             "    The sample z, shape (d,).\n\n"
             "Returns\n"
             "-------\n"
             "numpy.ndarray\n"
             "    y, shape (d,).")
        .def("invert_x",
             &Adaptation::invert_x,
             nb::arg("xi"),
             nb::arg("sigma"),
             "The step y = (x - m) / sigma of a point x (elementwise).\n\n"
             "Parameters\n"
             "----------\n"
             "xi : numpy.ndarray\n"
             "    The point x, shape (d,).\n"
             "sigma : numpy.ndarray\n"
             "    The step sizes per coordinate, shape (d,) (a column of `pop.S`).\n\n"
             "Returns\n"
             "-------\n"
             "numpy.ndarray\n"
             "    y, shape (d,).")
        .def("invert_y",
             &Adaptation::invert_y,
             nb::arg("yi"),
             "The sample z = A^-1 y of a step y, the inverse of `compute_y`.\n\n"
             "Parameters\n"
             "----------\n"
             "yi : numpy.ndarray\n"
             "    The step y, shape (d,).\n\n"
             "Returns\n"
             "-------\n"
             "numpy.ndarray\n"
             "    z, shape (d,).")
        .def(
            "__repr__",
            [](Adaptation &a) {
                return repr("Adaptation", [&](std::stringstream &ss) {
                    adaptation_repr(ss, a);
                    ss << " dd: " << a.dd;
                });
            },
            repr_doc);

    auto none = nb::class_<None, Adaptation>(
        m,
        "NoAdaptation",
        "No matrix adaptation: only the mean and the step size are adapted, A = I "
        "(`MatrixAdaptationType.NONE`).\n\n"
        "The cheapest option, O(d lambda) per generation; for well-conditioned (sphere-like) "
        "functions, or as a baseline. Settings switches `Modules.active` off with it.");
    adaptation_init(none);
    none.def(
        "__repr__",
        [](None &a) {
            return repr("NoAdaptation", [&](std::stringstream &ss) {
                adaptation_repr(ss, a);
                ss << " dd: " << a.dd;
            });
        },
        repr_doc);

    auto covariance = nb::class_<CovarianceAdaptation, Adaptation>(
        m,
        "CovarianceAdaptation",
        "The covariance matrix adaptation of CMA-ES (`MatrixAdaptationType.COVARIANCE`, the "
        "default).\n\n"
        "C receives a rank-one update from the evolution path pc and a rank-mu update from the "
        "selected steps (with `Modules.active`, negative weights for the worst steps), and is "
        "decomposed as C = B diag(d)^2 B^T; samples are y = B diag(d) z. C is updated every "
        "generation; the eigendecomposition only every `es.p.weights.lazy_update_interval` = "
        "1 / (10 d (c1 + cmu)) generations (every generation unless d is large). The "
        "eigendecomposition costs O(d^3) (LAPACK from d = `constants.lapack_min_dim`), the "
        "rank updates O(d^2 lambda). Settings replaces it by COVARIANCE_NO_EIGV when the "
        "step-size adaptation is not CSA (unless `always_compute_eigv`). Hansen & Ostermeier "
        "(2001); Hansen (2016), The CMA Evolution Strategy: A Tutorial.");
    adaptation_init(covariance);
    covariance
        .def_rw("pc",
                &CovarianceAdaptation::pc,
                "The evolution path of the rank-one update of C, shape (d,) (zeros at the "
                "start).")
        .def_rw("d",
                &CovarianceAdaptation::d,
                "The square roots of the eigenvalues of C (the standard deviations along the "
                "principal axes), shape (d,), in the order of the columns of B (ones at the "
                "start). With B, the `metric()`. In `CovarianceNoEigvAdaptation` filled only by "
                "`metric()`.")
        .def_rw("B",
                &CovarianceAdaptation::B,
                "The eigenvectors of C as columns, shape (d, d) (the identity at the start). In "
                "`CovarianceNoEigvAdaptation` filled only by `metric()`.")
        .def_rw("C",
                &CovarianceAdaptation::C,
                "The covariance matrix (without sigma^2), symmetric, shape (d, d) (the identity "
                "at the start).")
        .def_rw("A",
                &CovarianceAdaptation::A,
                "The sampling matrix, y = A z, shape (d, d): B diag(d) (the lower triangular "
                "Cholesky factor of C in `CovarianceNoEigvAdaptation`).")
        .def_rw("hs",
                &CovarianceAdaptation::hs,
                "The indicator h_sigma: False when ||ps|| is large (the step size is growing "
                "fast), which stalls the update of pc; True otherwise.")
        .def("adapt_covariance_matrix",
             &CovarianceAdaptation::adapt_covariance_matrix,
             nb::arg("weights"),
             nb::arg("modules"),
             nb::arg("population"),
             nb::arg("mu"),
             "Update C with the rank-one (pc) and rank-mu (selected steps) updates.\n\n"
             "With `modules.active` the whole population is used, the negative weights of the "
             "worst steps rescaled by d / ||C^-1/2 y_i||^2 (Hansen 2016, eq. 46); otherwise "
             "the mu best. Does not decompose C.\n\n"
             "Parameters\n"
             "----------\n"
             "weights : Weights\n"
             "    The recombination weights and learning rates (`es.p.weights`).\n"
             "modules : Modules\n"
             "    The modules.\n"
             "population : Population\n"
             "    The selected population (sorted, best first).\n"
             "mu : int\n"
             "    The number of selected individuals.")
        .def("perform_eigendecomposition",
             &CovarianceAdaptation::perform_eigendecomposition,
             nb::arg("stats"),
             "Decompose C = B diag(d)^2 B^T and set A = B diag(d).\n\n"
             "Uses Eigen or LAPACK (dsyevd, dsyevr) as `constants.eigen_solver` says. In "
             "`CovarianceNoEigvAdaptation`: the Cholesky factorization C = A A^T instead.\n\n"
             "Parameters\n"
             "----------\n"
             "stats : Settings\n"
             "    The settings (only `verbose` is used).\n\n"
             "Returns\n"
             "-------\n"
             "bool\n"
             "    False when the decomposition fails or gives a negative or non-finite "
             "eigenvalue.")
        .def("adapt_ps",
             &CovarianceAdaptation::adapt_ps,
             nb::arg("weights"),
             "Update ps = (1 - cs) ps + sqrt(cs (2 - cs) mueff) C^-1/2 dm (with dz instead of "
             "C^-1/2 dm in `CovarianceNoEigvAdaptation`).\n\n"
             "Parameters\n"
             "----------\n"
             "weights : Weights\n"
             "    The recombination weights and learning rates (`es.p.weights`).")
        .def(
            "__repr__",
            [](CovarianceAdaptation &a) {
                return repr("CovarianceAdaptation", [&](std::stringstream &ss) {
                    adaptation_repr(ss, a);
                    ss << " pc: " << a.pc.transpose() << " d: " << a.d.transpose();
                    ss << " B: " << a.B << " C: " << a.C;
                    ss << " dd: " << a.dd;
                    ss << " hs: " << a.hs;
                });
            },
            repr_doc);

    auto separable = nb::class_<SeparableAdaptation, Adaptation>(
        m,
        "SeparableAdaptation",
        "Separable CMA-ES: a diagonal covariance matrix (`MatrixAdaptationType.SEPARABLE`).\n\n"
        "Only the d variances are learned, with the rank-one and rank-mu updates of CMA-ES "
        "applied per coordinate (the default cmu multiplied by (d + 2) / 3); samples are "
        "y = d * z. "
        "O(d lambda) per generation: for separable functions, and for high dimension, where a "
        "full matrix is slow to learn. Ros & Hansen (2008).");
    adaptation_init(separable);
    separable
        .def_rw("c",
                &SeparableAdaptation::c,
                "The diagonal of C, the variance of each coordinate (without sigma^2), shape "
                "(d,), at least 1e-12 (ones at the start).")
        .def_rw("pc",
                &SeparableAdaptation::pc,
                "The evolution path of the rank-one update, shape (d,) (zeros at the start).")
        .def_rw("d",
                &SeparableAdaptation::d,
                "The standard deviations sqrt(c), shape (d,): y = d * z (ones at the start).")
        .def(
            "__repr__",
            [](SeparableAdaptation &a) {
                return repr("SeparableAdaptation", [&](std::stringstream &ss) {
                    adaptation_repr(ss, a);
                    ss << " pc: " << a.pc.transpose() << " d: " << a.d.transpose();
                    ss << " c: " << a.c.transpose();
                    ss << " hs: " << a.hs;
                });
            },
            repr_doc);

    auto matrix = nb::class_<MatrixAdaptation, Adaptation>(
        m,
        "MatrixAdaptation",
        "MA-ES, the matrix adaptation evolution strategy (`MatrixAdaptationType.MATRIX`).\n\n"
        "Learns the sampling matrix M directly, y = M z, without a covariance matrix, an "
        "evolution path pc or a decomposition: M = (1 - c1/2 - cmu/2) M + (c1/2) (M ps) ps^T + "
        "(cmu/2) sum_i w_i y_i z_i^T. O(d^2 lambda) per generation (BLAS from d = "
        "`constants.blas_min_dim`), so it suits high dimension. Beyer & Sendhoff (2017).");
    adaptation_init(matrix);
    matrix
        .def_rw("M",
                &MatrixAdaptation::M,
                "The sampling matrix, y = M z, shape (d, d) (the identity at the start).")
        .def_rw("M_inv",
                &MatrixAdaptation::M_inv,
                "The inverse of M, shape (d, d). Updated with M under elitism (lambda > 1); "
                "otherwise recomputed only when `invert_y` needs it (an LU inverse from d = "
                "`constants.blas_min_dim`, else the pseudo-inverse), so it can lag behind M.")
        .def(
            "__repr__",
            [](MatrixAdaptation &a) {
                return repr("MatrixAdaptation", [&](std::stringstream &ss) {
                    adaptation_repr(ss, a);
                    ss << " M: " << a.M << " dd: " << a.dd;
                });
            },
            repr_doc);

    auto cholesky = nb::class_<CholeskyAdaptation, Adaptation>(
        m,
        "CholeskyAdaptation",
        "Cholesky-CMA-ES: the covariance matrix kept as its triangular Cholesky factor "
        "(`MatrixAdaptationType.CHOLESKY`).\n\n"
        "The rank-one (pc) and rank-mu updates of C = A A^T are applied to A as rank-one "
        "Cholesky updates, one per selected individual, so there is no decomposition: "
        "O(d^2 lambda) per generation and O(d^2) memory. Krause et al. (2016).");
    adaptation_init(cholesky);
    cholesky
        .def_rw("A",
                &CholeskyAdaptation::A,
                "The lower triangular Cholesky factor of C, y = A z, shape (d, d) (the identity "
                "at the start).")
        .def_rw("pc",
                &CholeskyAdaptation::pc,
                "The evolution path of the rank-one update, shape (d,) (zeros at the start).");

    auto self = nb::class_<SelfAdaptation, Adaptation>(
        m,
        "SelfAdaptation",
        "CMSA-ES, covariance matrix self-adaptation (`MatrixAdaptationType.CMSA`).\n\n"
        "C = (1 - 1/tc) C + (1/tc) sum_i w_i y_i y_i^T with tc = 1 + d (d + 1) / (2 mueff), "
        "without an evolution path, then a Cholesky factorization C = A A^T every generation "
        "(O(d^3); LAPACK from d = `constants.lapack_min_dim`). With CMSA the default mu is "
        "lambda / 4. The CMSA-ES of the paper adapts the step size by self-adaptation "
        "(`StepSizeAdaptation.SA`). Beyer & Sendhoff (2008).");
    adaptation_init(self);
    self.def_rw("A",
                &SelfAdaptation::A,
                "The lower triangular Cholesky factor of C, y = A z, shape (d, d) (the identity "
                "at the start).")
        .def_rw("C",
                &SelfAdaptation::C,
                "The covariance matrix (without sigma^2), shape (d, d) (the identity at the "
                "start).");

    auto no_eigv = nb::class_<CovarianceNoEigvAdaptation, CovarianceAdaptation>(
        m,
        "CovarianceNoEigvAdaptation",
        "CMA-ES with a Cholesky factorization instead of an eigendecomposition "
        "(`MatrixAdaptationType.COVARIANCE_NO_EIGV`).\n\n"
        "C is updated as in `CovarianceAdaptation` and factorized as C = A A^T with A lower "
        "triangular (Eigen, or LAPACK dpotrf from d = `constants.lapack_min_dim`, see "
        "`constants.cholesky_solver`), which is several times cheaper than an "
        "eigendecomposition. B and d are filled only by `metric()`, from A A^T, the C of the last "
        "factorization. ps is updated with dz instead "
        "of C^-1/2 dm, which only CSA needs: Settings selects this adaptation in place of "
        "COVARIANCE for the other step-size adaptations.");
    adaptation_init(no_eigv);

    nb::class_<NaturalGradientAdaptation, Adaptation>(
        m,
        "NaturalGradientAdaptation",
        "xNES, exponential natural evolution strategies "
        "(`MatrixAdaptationType.NATURAL_GRADIENT`).\n\n"
        "The sampling matrix is updated multiplicatively, A = A expm(cc G), with the natural "
        "gradient G = sum_i w_i (z_i z_i^T - I) made trace-free (its trace part, `sigma_g`, "
        "goes to the step size with `StepSizeAdaptation.XNES`) and cc = "
        "`es.p.weights.cc`. For lambda < d and d >= 16 the matrix exponential is computed "
        "exactly from a thin QR of the samples, O(d^2 lambda); otherwise as a dense d x d "
        "exponential. Glasmachers et al. (2010).")
        .def(nb::init<size_t, Vector>(),
             nb::arg("dimension"),
             nb::arg("x0"),
             "Create the adaptation with mean x0 and A = I (the step size sets the spread).\n\n"
             "Parameters\n"
             "----------\n"
             "dimension : int\n"
             "    The dimension d.\n"
             "x0 : numpy.ndarray\n"
             "    The initial mean m, shape (d,).")
        .def_rw("A",
                &NaturalGradientAdaptation::A,
                "The sampling matrix, y = A z, shape (d, d) (I at the start).")
        .def_rw("A_inv",
                &NaturalGradientAdaptation::A_inv,
                "The inverse of A, shape (d, d). Recomputed only when `invert_y` needs it (an "
                "LU inverse from d = `constants.blas_min_dim`, else the pseudo-inverse), so it "
                "can lag behind A (the identity at the start).")
        .def_prop_ro("G",
                     &NaturalGradientAdaptation::G,
                     "The natural gradient G = Z diag(u) Z^T - alpha I as a dense (d, d) "
                     "matrix, computed on access from `gradient_samples` (Z), "
                     "`gradient_weights` (u) and `gradient_shift` (alpha); the update does not "
                     "form it.")
        .def_rw("gradient_samples",
                &NaturalGradientAdaptation::gradient_samples,
                "The samples Z of the last gradient, the z of the population as columns, "
                "shape (d, k) with k <= lambda.")
        .def_rw("gradient_weights",
                &NaturalGradientAdaptation::gradient_weights,
                "The weights u of the samples, shape (k,) (with `Modules.active`, negative "
                "weights for the worst).")
        .def_rw("gradient_shift",
                &NaturalGradientAdaptation::gradient_shift,
                "The shift alpha of G: the sum of the weights plus `sigma_g`, which makes G "
                "trace-free (the sum of the weights alone in a (1+1)-ES).")
        .def_rw("sigma_g",
                &NaturalGradientAdaptation::sigma_g,
                "trace(G) / d of the gradient before the shift: its isotropic (step-size) part, "
                "removed from G and used by `StepSizeAdaptation.XNES`, "
                "sigma *= exp(cs / 2 * sigma_g).")
        .def("compute_gradients",
             &NaturalGradientAdaptation::compute_gradients,
             nb::arg("pop"),
             nb::arg("weights"),
             nb::arg("settings"),
             "Compute the natural gradient of the population: "
             "`gradient_samples`, `gradient_weights`, `gradient_shift` and "
             "`sigma_g`. Called by `adapt_evolution_paths`.\n\n"
             "Parameters\n"
             "----------\n"
             "pop : Population\n"
             "    The selected population (sorted, best first).\n"
             "weights : Weights\n"
             "    The recombination weights and learning rates (`es.p.weights`).\n"
             "settings : Settings\n"
             "    The settings (`es.p.settings`).");

    // The old, misspelled name of 1.x, with a DeprecationWarning
    adaptation.def(
        "adapt_evolution_paths_innner",
        [](Adaptation &a,
           const Population &pop,
           const parameters::Weights &weights,
           const parameters::Stats &stats,
           const parameters::Settings &settings,
           const size_t mu,
           const size_t lamb) {
            warn_deprecated("Adaptation.adapt_evolution_paths_innner is deprecated; use "
                            "Adaptation.adapt_evolution_paths_inner");
            // mu and lamb are accepted as in 1.x, and not used
            static_cast<void>(mu);
            static_cast<void>(lamb);
            a.adapt_evolution_paths_inner(pop, weights, stats, settings);
        },
        nb::arg("pop"),
        nb::arg("weights"),
        nb::arg("stats"),
        nb::arg("settings"),
        nb::arg("mu"),
        nb::arg("lamb"),
        "Deprecated: the old (misspelled) name of `adapt_evolution_paths_inner`; mu and lamb "
        "are accepted, as in 1.x, and not used.");
}

void define_bounds(nb::module_ &main)
{
    auto m = main.def_submodule(
        "bounds",
        "Bound correction: what happens to samples outside the box [lb, ub].\n\n"
        "The optimizer holds one as `es.p.bounds`, selected by `Modules.bound_correction` "
        "(`options.CorrectionMethod`). Each coordinate and each side is corrected "
        "separately: an infinite bound is never crossed, and a coordinate with one infinite "
        "bound is set to its finite bound when it crosses it (the methods other than SATURATE "
        "need the width ub - lb). The methods are described in de Nobel et al. (2021).");
    using namespace bounds;

    nb::class_<BoundCorrection>(
        m,
        "BoundCorrection",
        "Base class of the bound corrections.\n\n"
        "After a sample x = m + sigma y is formed, `correct` counts it when it lies outside "
        "[lb, ub] and, unless the method is NONE, moves the coordinates outside into the box "
        "(`correct_x`) and recomputes y and z of the individual from the corrected x, so the "
        "updates learn from the point that is evaluated.")
        .def_ro("n_out_of_bounds",
                &BoundCorrection::n_out_of_bounds,
                "The number of samples of the current population that were outside the bounds "
                "(reset at the start of a population; with RESAMPLE every rejected sample "
                "counts).")
        .def("correct",
             &BoundCorrection::correct,
             nb::arg("index"),
             nb::arg("parameters"),
             "Correct individual `index` of the population in place.\n\n"
             "Counts it in `n_out_of_bounds` when it is outside the bounds and applies "
             "`correct_x`; X, X_transformed, Y and Z of the individual are updated.\n\n"
             "Parameters\n"
             "----------\n"
             "index : int\n"
             "    The column of the individual in `parameters.pop`.\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`).")
        .def("correct_x",
             &BoundCorrection::correct_x,
             nb::arg("xi"),
             nb::arg("oob"),
             nb::arg("sigma"),
             nb::arg("settings"),
             "The corrected point: the coordinates of xi where oob is True moved into "
             "[lb, ub], the others unchanged.\n\n"
             "Coordinates with two finite bounds are corrected by the method; a coordinate with "
             "one infinite bound is set to the finite bound it crossed, as SATURATE does.\n\n"
             "Parameters\n"
             "----------\n"
             "xi : numpy.ndarray\n"
             "    The point, shape (d,).\n"
             "oob : numpy.ndarray\n"
             "    The mask (bool) of the coordinates outside the bounds, shape (d,) (see "
             "`is_out_of_bounds`).\n"
             "sigma : float\n"
             "    The step size (used by COTN).\n"
             "settings : Settings\n"
             "    The settings with the bounds lb and ub.\n\n"
             "Returns\n"
             "-------\n"
             "numpy.ndarray\n"
             "    The corrected point, shape (d,).")
        .def("delta_out_of_bounds",
             &BoundCorrection::delta_out_of_bounds,
             nb::arg("xi"),
             nb::arg("oob"),
             nb::arg("settings"),
             "The coordinates of xi where oob is True normalized to the box, "
             "(x - lb) / (ub - lb) (in [0, 1] inside it); the others unchanged.\n\n"
             "Parameters\n"
             "----------\n"
             "xi : numpy.ndarray\n"
             "    The point, shape (d,).\n"
             "oob : numpy.ndarray\n"
             "    The mask (bool) of the coordinates to normalize, shape (d,).\n"
             "settings : Settings\n"
             "    The settings with the bounds lb and ub.\n\n"
             "Returns\n"
             "-------\n"
             "numpy.ndarray")
        .def("is_out_of_bounds",
             &BoundCorrection::is_out_of_bounds,
             nb::arg("xi"),
             nb::arg("settings"),
             "The mask of the coordinates of xi outside [lb, ub].\n\n"
             "Parameters\n"
             "----------\n"
             "xi : numpy.ndarray\n"
             "    The point, shape (d,).\n"
             "settings : Settings\n"
             "    The settings with the bounds lb and ub.\n\n"
             "Returns\n"
             "-------\n"
             "numpy.ndarray\n"
             "    A bool array, shape (d,).");

    nb::class_<Resample, BoundCorrection>(
        m,
        "Resample",
        "Sample again (`CorrectionMethod.RESAMPLE`): an individual outside the bounds is "
        "drawn again, up to 5 d times, and kept as it is after that.\n\n"
        "Keeps the distribution undistorted inside the box; costs samples, not evaluations.")
        .def(nb::init<>(), "Create the correction.");
    nb::class_<NoCorrection, BoundCorrection>(
        m,
        "NoCorrection",
        "No bound correction (`CorrectionMethod.NONE`, the default): points outside the "
        "bounds are evaluated as they are (and counted in `n_out_of_bounds`).")
        .def(nb::init<>(), "Create the correction.");
    nb::class_<COTN, BoundCorrection>(
        m,
        "COTN",
        "Complete one-tailed normal correction (`CorrectionMethod.COTN`; Caraffini et al. "
        "2019): a coordinate outside the bounds is moved inside, at a random distance from the "
        "violated bound.\n\n"
        "In coordinates normalized to [0, 1], the new value is |b - sigma |n||, with b the "
        "violated bound (0 or 1) and n normal with mean 0 and standard deviation 1/3.")
        .def(nb::init<size_t>(),
             nb::arg("dim"),
             "Create the correction.\n\n"
             "Parameters\n"
             "----------\n"
             "dim : int\n"
             "    The dimension d.")
        .def_ro("sampler",
                &COTN::sampler,
                "The Gaussian sampler of the offsets n (standard deviation 1/3).");
    nb::class_<Mirror, BoundCorrection>(
        m,
        "Mirror",
        "Reflect a coordinate outside the bounds back into the box at the violated bound "
        "(`CorrectionMethod.MIRROR`); a point more than one box width outside is reflected "
        "repeatedly.")
        .def(nb::init<>(), "Create the correction.");
    nb::class_<UniformResample, BoundCorrection>(
        m,
        "UniformResample",
        "Replace each coordinate outside the bounds by a uniform random value in [lb, ub] "
        "(`CorrectionMethod.UNIFORM_RESAMPLE`).")
        .def(nb::init<size_t>(),
             nb::arg("dim"),
             "Create the correction.\n\n"
             "Parameters\n"
             "----------\n"
             "dim : int\n"
             "    The dimension d.");
    nb::class_<Saturate, BoundCorrection>(
        m,
        "Saturate",
        "Set a coordinate outside the bounds to the violated bound, i.e. clip x to [lb, ub] "
        "(`CorrectionMethod.SATURATE`).")
        .def(nb::init<>(), "Create the correction.");
    nb::class_<Toroidal, BoundCorrection>(
        m,
        "Toroidal",
        "Wrap a coordinate outside the bounds around to the other side of the box "
        "(`CorrectionMethod.TOROIDAL`): periodic boundaries.")
        .def(nb::init<>(), "Create the correction.");
}

void define_mutation(nb::module_ &main)
{
    auto m = main.def_submodule(
        "mutation",
        "Mutation: the sampling of a population and the step-size adaptation.\n\n"
        "The optimizer holds a `Strategy` subclass as `es.p.mutation`, selected by "
        "`Modules.ssa` (`options.StepSizeAdaptation`) and created anew at every restart. It "
        "holds the threshold convergence (`Modules.threshold_convergence`), the sequential "
        "selection (`Modules.sequential_selection`) and the sampler of the step sizes "
        "(`Modules.sample_sigma`). The step-size adaptations are described in de Nobel et al. "
        "(2021).");
    using namespace mutation;

    nb::class_<ThresholdConvergence>(
        m,
        "ThresholdConvergence",
        "Threshold convergence: samples shorter than a decreasing threshold are lengthened "
        "(`Modules.threshold_convergence`).\n\n"
        "The threshold is a length of z: t = init_threshold * E||z|| * ((budget - evaluations) / "
        "budget)^decay_factor, computed once per generation from the evaluations at its start "
        "(0 once the budget is used up). A sample with ||z|| < t is lengthened to "
        "||z|| = 2 t - ||z|| (z and y are scaled). Keeps the search global early on and turns "
        "into local search as the budget is used. After Piad-Morffis et al. (2015), who put the "
        "threshold on x - m as a fraction of the box diameter; on z it is relative to the search "
        "distribution and does not depend on the box.")
        .def(nb::init<>(),
             "Create threshold convergence with init_threshold = 1 and decay_factor = 0.995.")
        .def_rw("init_threshold",
                &ThresholdConvergence::init_threshold,
                "The threshold at the start of a run as a multiple of E||z|| (default 1).")
        .def_rw("decay_factor",
                &ThresholdConvergence::decay_factor,
                "The exponent of the decrease of the threshold with the remaining budget "
                "(default 0.995).")
        .def("threshold",
             &ThresholdConvergence::threshold,
             nb::arg("expected_length"),
             nb::arg("budget"),
             nb::arg("evaluations"),
             "The threshold after `evaluations` of `budget`.\n\n"
             "Parameters\n"
             "----------\n"
             "expected_length : float\n"
             "    E||z|| of the sampler (`es.p.sampler.expected_length()`).\n"
             "budget : int\n"
             "    The evaluation budget.\n"
             "evaluations : int\n"
             "    The evaluations used so far (more than the budget gives 0).\n\n"
             "Returns\n"
             "-------\n"
             "float\n"
             "    init_threshold * expected_length * ((budget - evaluations) / budget)^"
             "decay_factor.")
        .def("apply",
             &ThresholdConvergence::apply,
             nb::arg("z"),
             nb::arg("y"),
             nb::arg("t"),
             "Lengthen one sample in place: when ||z|| < t, z and y are scaled so that "
             "||z|| = 2 t - ||z||.\n\n"
             "Parameters\n"
             "----------\n"
             "z, y : numpy.ndarray\n"
             "    The sample and its y = A z, shape (d,), float64; changed in place.\n"
             "t : float\n"
             "    The threshold.");

    nb::class_<NoThresholdConvergence, ThresholdConvergence>(
        m,
        "NoThresholdConvergence",
        "No threshold convergence (the default): samples are not scaled.")
        .def(nb::init<>(), "Create the no-op threshold convergence.");

    nb::class_<SequentialSelection>(
        m,
        "SequentialSelection",
        "Sequential selection: stop evaluating a population once it improves on the previous "
        "generation (`Modules.sequential_selection`).\n\n"
        "The population is cut after individual i (0-based) when its objective value is better "
        "than the best of the previous generation (in the first generation of a run, never) "
        "and i >= mu * seq_cutoff_factor; with pairwise selection the "
        "factor is at least 2 and the cut falls after a complete pair. The updates use the "
        "evaluated individuals. Saves evaluations on functions where improvements are easy to "
        "find. Brockhoff et al. (2010).")
        .def(nb::init<parameters::Mirror, Float>(),
             nb::arg("mirror"),
             nb::arg("seq_cutoff_factor") = 1.0,
             "Create the sequential selection.\n\n"
             "Parameters\n"
             "----------\n"
             "mirror : options.Mirror\n"
             "    The mirrored sampling option; with `Mirror.PAIRWISE` the cutoff factor is at "
             "least 2.\n"
             "seq_cutoff_factor : float\n"
             "    The minimum number of evaluated individuals as a multiple of mu (default 1).")
        .def(
            "__init__",
            [](SequentialSelection *self, parameters::Mirror mirror, Float seq_cuttoff_factor) {
                warn_deprecated("the keyword seq_cuttoff_factor is deprecated; use "
                                "seq_cutoff_factor");
                new (self) SequentialSelection(mirror, seq_cuttoff_factor);
            },
            nb::arg("mirror"),
            nb::kw_only(),
            nb::arg("seq_cuttoff_factor"),
            "Deprecated: the old (misspelled) keyword of 1.x.")
        .def("break_conditions",
             &SequentialSelection::break_conditions,
             nb::arg("i"),
             nb::arg("f"),
             nb::arg("fopt"),
             nb::arg("mirror"),
             nb::arg("mu"),
             "Whether the population stops after individual i.\n\n"
             "Parameters\n"
             "----------\n"
             "i : int\n"
             "    The index of the individual just evaluated (0-based).\n"
             "f : float\n"
             "    Its objective value.\n"
             "fopt : float\n"
             "    The reference value: the best of the previous generation.\n"
             "mirror : options.Mirror\n"
             "    The mirrored sampling option.\n"
             "mu : int\n"
             "    The number of selected individuals.\n\n"
             "Returns\n"
             "-------\n"
             "bool\n"
             "    True when f < fopt, i >= mu * seq_cutoff_factor and, with pairwise "
             "selection, i is odd.");

    nb::class_<NoSequentialSelection, SequentialSelection>(
        m,
        "NoSequentialSelection",
        "No sequential selection (the default): every individual of a population is "
        "evaluated.")
        .def(nb::init<parameters::Mirror, Float>(),
             nb::arg("mirror"),
             nb::arg("seq_cutoff_factor") = 1.0,
             "Create the no-op sequential selection.\n\n"
             "Parameters\n"
             "----------\n"
             "mirror : options.Mirror\n"
             "    The mirrored sampling option (not used).\n"
             "seq_cutoff_factor : float\n"
             "    Not used (default 1).")
        .def(
            "__init__",
            [](NoSequentialSelection *self, parameters::Mirror mirror, Float seq_cuttoff_factor) {
                warn_deprecated("the keyword seq_cuttoff_factor is deprecated; use "
                                "seq_cutoff_factor");
                new (self) NoSequentialSelection(mirror, seq_cuttoff_factor);
            },
            nb::arg("mirror"),
            nb::kw_only(),
            nb::arg("seq_cuttoff_factor"),
            "Deprecated: the old (misspelled) keyword of 1.x.");

    nb::class_<SigmaSampler>(
        m,
        "SigmaSampler",
        "Sampled step sizes: each individual gets its own step size per coordinate, "
        "sigma * exp(beta n) with n standard normal (`Modules.sample_sigma`; always used by "
        "the step-size adaptations LPXNES and SA).\n\n"
        "beta = `es.p.weights.beta` = 1 / sqrt(2 mueff) (ln 2 / (sqrt(d) ln d) with LPXNES). "
        "The step sizes are the columns of `es.p.pop.S`.")
        .def(nb::init<Float>(),
             nb::arg("dimension"),
             "Create the sampler.\n\n"
             "Parameters\n"
             "----------\n"
             "dimension : float\n"
             "    The dimension d.")
        .def("apply_integer_bounds",
             nb::overload_cast<parameters::Parameters &>(&SigmaSampler::apply_integer_bounds),
             nb::arg("parameters"),
             "Raise the step sizes of the integer variables to their lower bound "
             "min(0.2, mueff / d) / sqrt(C_ii) (C_ii from "
             "`adaptation.coordinate_wise_variances`).\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer; `pop.S` is changed in place.")
        .def("sample",
             &SigmaSampler::sample,
             nb::arg("sigma"),
             nb::arg("parameters"),
             "Set the step sizes `pop.S` of the population from sigma, then apply the lower "
             "bounds of the integer variables.\n\n"
             "Parameters\n"
             "----------\n"
             "sigma : float\n"
             "    The global step size.\n"
             "parameters : Parameters\n"
             "    The state of the optimizer; `pop.S` is written.");

    nb::class_<NoSigmaSampler, SigmaSampler>(
        m,
        "NoSigmaSampler",
        "No sampled step sizes (the default): every individual uses sigma (apart from the "
        "lower bounds of integer variables).")
        .def(nb::init<Float>(),
             nb::arg("dimension"),
             "Create the sampler.\n\n"
             "Parameters\n"
             "----------\n"
             "dimension : float\n"
             "    The dimension d.");

    nb::class_<Strategy>(
        m,
        "Strategy",
        "Base class of the step-size adaptations; also samples the population.\n\n"
        "The optimizer samples a population through it, in batches, passes it the objective "
        "values, and `adapt` updates sigma after selection. "
        "The base class does not change sigma. The optimizer creates the subclass of "
        "`Modules.ssa`; each subclass has the same constructor.")
        .def(nb::init<std::shared_ptr<ThresholdConvergence>,
                      std::shared_ptr<SequentialSelection>,
                      std::shared_ptr<SigmaSampler>,
                      Float>(),
             nb::arg("threshold_convergence"),
             nb::arg("sequential_selection"),
             nb::arg("sigma_sampler"),
             nb::arg("sigma0"),
             "Create a strategy that does not adapt sigma.\n\n"
             "Parameters\n"
             "----------\n"
             "threshold_convergence : ThresholdConvergence\n"
             "    The threshold convergence (or `NoThresholdConvergence`).\n"
             "sequential_selection : SequentialSelection\n"
             "    The sequential selection (or `NoSequentialSelection`).\n"
             "sigma_sampler : SigmaSampler\n"
             "    The sampler of the step sizes (or `NoSigmaSampler`).\n"
             "sigma0 : float\n"
             "    The initial step size.")
        .def("adapt",
             &Strategy::adapt,
             nb::arg("weights"),
             nb::arg("dynamic"),
             nb::arg("population"),
             nb::arg("old_population"),
             nb::arg("stats"),
             "Update the step size sigma after selection.\n\n"
             "Parameters\n"
             "----------\n"
             "weights : Weights\n"
             "    The recombination weights and learning rates (`es.p.weights`).\n"
             "dynamic : matrix_adaptation.Adaptation\n"
             "    The matrix adaptation (`es.p.adaptation`).\n"
             "population : Population\n"
             "    The selected population of this generation.\n"
             "old_population : Population\n"
             "    The population of the previous generation.\n"
             "stats : Stats\n"
             "    The statistics of the run.")
        .def_rw("threshold_convergence",
                &Strategy::tc,
                "The threshold convergence (`ThresholdConvergence` or "
                "`NoThresholdConvergence`).")
        .def_rw("sequential_selection",
                &Strategy::sq,
                "The sequential selection (`SequentialSelection` or `NoSequentialSelection`).")
        .def_rw("sigma_sampler",
                &Strategy::ss,
                "The sampler of the step sizes (`SigmaSampler` or `NoSigmaSampler`).")
        .def_rw("sigma", &Strategy::sigma, "The global step size sigma.")
        .def_ro("generation_threshold",
                &Strategy::generation_threshold,
                "The threshold of threshold convergence in the current generation, computed at "
                "its start (`ThresholdConvergence.threshold`; also set without the module).")
        .def_rw("s",
                &Strategy::s,
                "The smoothed success measure of TPA, MSR and PSR, whose exponential scales "
                "sigma (default 0).");

    auto ssa_csa = nb::class_<CSA, Strategy>(
        m,
        "CSA",
        "Cumulative step-size adaptation (`StepSizeAdaptation.CSA`, the default).\n\n"
        "sigma *= exp(l), l = (cs / damps) (||ps|| / E||z|| - 1) clamped to [-1, 1] (as in "
        "pycma): sigma grows when the evolution path ps is longer than under random selection "
        "and shrinks when it is shorter. Hansen & Ostermeier (2001).");
    strategy_init(ssa_csa);
    auto ssa_tpa = nb::class_<TPA, Strategy>(
        m,
        "TPA",
        "Two-point step-size adaptation (`StepSizeAdaptation.TPA`).\n\n"
        "With every population (in its first batch) the two points m + sigma dm and "
        "m - sigma dm (along the last step of the mean) are evaluated; s = (1 - cs) s + cs r, "
        "with r = -a_tpa when m - sigma dm is better, a_tpa + b_tpa when m + sigma dm is better and 0 for a tie (as "
        "after a restart, when dm = 0), and sigma *= exp(s). Costs two evaluations per "
        "generation. Hansen (2008); a tie as in pycma.")
        .def_rw("a_tpa", &TPA::a_tpa, "The magnitude of r (default 0.5).")
        .def_rw("b_tpa",
                &TPA::b_tpa,
                "The offset added to r when the forward point is better (default 0).")
        .def_rw("rank_tpa", &TPA::rank_tpa, "The last r: -a_tpa, a_tpa + b_tpa or 0 (default 0).");
    strategy_init(ssa_tpa);
    auto ssa_msr = nb::class_<MSR, Strategy>(
        m,
        "MSR",
        "Median success rule (`StepSizeAdaptation.MSR`).\n\n"
        "With k the number of individuals better than the j-th best of the previous "
        "population, j = quantile * lambda (0-based, interpolated between its neighbours, as "
        "pycma): s = (1 - cs) s + cs (2 / lambda) (k - lambda / 2) and "
        "sigma *= exp(s / damps), with cs = 0.3 and damps = 2 - 2 / d (1 for d = 1) by "
        "default: sigma grows when the median offspring beats the j-th best of the previous "
        "generation. Ait ElHara, Auger & Hansen (2013).")
        .def_rw("quantile",
                &MSR::quantile,
                "The position j / lambda of the previous generation's value that the offspring "
                "are compared with (default 0.3).");
    strategy_init(ssa_msr);
    auto ssa_psr = nb::class_<PSR, Strategy>(
        m,
        "PSR",
        "Population success rule (`StepSizeAdaptation.PSR`).\n\n"
        "The current and the previous population are ranked together; with z the difference "
        "of their summed ranks (previous minus current) divided by n^2, minus success_ratio: "
        "s = (1 - cs) s + cs z and sigma *= exp(s / damps), with cs = 0.3 and damps = 1 by "
        "default (Loshchilov 2014, Alg. 6).")
        .def_rw("success_ratio", &PSR::success_ratio, "The target success ratio (default 0.25).");
    strategy_init(ssa_psr);
    auto ssa_xnes = nb::class_<XNES, Strategy>(
        m,
        "XNES",
        "The step-size update of xNES (`StepSizeAdaptation.XNES`).\n\n"
        "With NATURAL_GRADIENT: sigma *= exp(cs / 2 * sigma_g), sigma_g the trace part of the "
        "natural gradient (`NaturalGradientAdaptation.sigma_g`). With the other matrix "
        "adaptations: sigma *= exp(cs / sqrt(d) * sum_i w_i (||z_i||^2 - d)) over the "
        "positive weights. Glasmachers et al. (2010).");
    strategy_init(ssa_xnes);
    auto ssa_mxnes = nb::class_<MXNES, Strategy>(
        m,
        "MXNES",
        "A variant of the xNES step-size update on the mean step "
        "(`StepSizeAdaptation.MXNES`).\n\n"
        "sigma *= exp((cs / d) (mueff ||dz||^2 - d)), with dz the weighted mean of the "
        "selected z: sigma grows when the mean step is longer than under random selection. "
        "m-xNES of Krause, Glasmachers & Igel (2017), "
        "\"Qualitative and quantitative assessment of step size adaptation "
        "rules\".");
    strategy_init(ssa_mxnes);
    auto ssa_lpxnes = nb::class_<LPXNES, Strategy>(
        m,
        "LPXNES",
        "A variant of the xNES step-size update on sampled step sizes "
        "(`StepSizeAdaptation.LPXNES`).\n\n"
        "Every individual samples its own step sizes (a `SigmaSampler` is always used, with "
        "beta = ln 2 / (sqrt(d) ln d)); sigma *= exp(cs sum_i w_i mean_j ln(S_ij / sigma)) "
        "over the positive weights, which moves sigma towards the step sizes of the better "
        "individuals. lp-xNES of Krause, Glasmachers & Igel (2017), "
        "\"Qualitative and quantitative assessment of step size adaptation "
        "rules\".");
    strategy_init(ssa_lpxnes);
    auto ssa_sr = nb::class_<SR, Strategy>(
        m,
        "SR",
        "The success rule of the (1+1)-CMA-ES (`StepSizeAdaptation.SR`; always used when "
        "lambda = 1).\n\n"
        "sigma *= exp((p_s - p_t) / (damps (1 - p_t))), with p_s the smoothed rate of "
        "successes (`es.p.stats.success_ratio`): a generation is a success when its best "
        "individual is better than the best of the run so far (with elitism, the parent). The "
        "target p_t = 2/11 and damps = 1 + d / 2 by default. Igel et al. (2006).\n\n"
        "The rule is designed for elitist selection, where the parent is the best point so "
        "far. Without elitism, the best of the run takes the parent's role: comparing with the "
        "previous generation's best instead would count a success about every other "
        "generation even at a good sigma, and sigma would diverge. For non-elitist selection "
        "see also `MSR`.");
    strategy_init(ssa_sr);
    auto ssa_sa = nb::class_<SA, Strategy>(
        m,
        "SA",
        "Self-adaptation of the step size (`StepSizeAdaptation.SA`).\n\n"
        "Every individual samples its own step sizes (a `SigmaSampler` is always used); "
        "sigma is multiplied by the weighted geometric mean of the step sizes of the mu "
        "selected individuals divided by the geometric mean over the whole population (Beyer & "
        "Sendhoff 2008, with the correction of Ahrari et al. 2017). With SA the default mu is "
        "lambda / 4.");
    strategy_init(ssa_sa);
}
} // namespace modcma_python
