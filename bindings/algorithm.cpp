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
        "`NoElitism` otherwise. From the second generation of a run on, the parents (the mu "
        "best of the previous selection, the first columns of `es.p.old_pop`; not the losers "
        "of pairwise selection, nor points with f = +inf) are appended to the offspring before "
        "sorting, so the best points found so far survive. Their y and z are recomputed in the "
        "current distribution with the current step size, y = (x - m) / sigma and z = A^-1 y, "
        "as for a repaired point (the framework paper, Sec. 4.1); integer variables keep their "
        "lower bound on the step size. Not in the (1+1)-ES (`Settings.one_plus_one`): its "
        "parent keeps the y and z it was sampled with. The selection keeps lambda of the "
        "mu + lambda points; the dropped ones stay after the population for the active update "
        "(`Population.n_pool`, #161). Can speed up convergence on unimodal functions, at the "
        "cost of diversity.")
        .def(nb::init<>(), "Create the elitist selection.")
        .def("__call__",
             &Elitsm::operator(),
             nb::arg("parameters"),
             "Append the parents (at most mu) to the current population, with y and z "
             "recomputed (from the second generation of a run on).\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`); its `pop` is changed in place, and the "
             "step sizes, y and z of the parents in `old_pop`.");

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
                         "the active update. The Settings constructor makes lambda0 even with "
                         "PAIRWISE, and turns PAIRWISE into NONE for the (1+1)-ES, lambda0 = 1 "
                         "(#136). Before 2.0 the losers got f = +inf (#75).")
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
                         "first lambda (with elitism the dropped points of the pool stay after "
                         "them, `Population.n_pool`, #161) and updates `es.p.stats`: the best "
                         "solutions (with a bound correction, only a point in the box becomes "
                         "`global_best`, #168), the mean objective value and `success_ratio` (a "
                         "success: the best of the generation beats the best of the run).")
        .def(nb::init<parameters::Modules>(),
             nb::arg("modules"),
             "Create the selection that the modules ask for.\n\n"
             "Parameters\n"
             "----------\n"
             "modules : Modules\n"
             "    The modules; `mirrored` and `elitist` choose `pairwise` and `elitism`.")
        .def("select",
             &Strategy::select,
             nb::arg("parameters"),
             "Select the evaluated population in place.\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : Parameters\n"
             "    The state of the optimizer (`es.p`); `pop` is sorted (best first; with "
             "PAIRWISE the losers of the pairs last) and cut to at most lambda individuals "
             "(`pop.n_pool`: the size of the pool before the cut), `stats` is updated.")
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

    // Pickling and copying (#91)
    def_pickle<Elitsm, Elitsm>();
    def_pickle<NoElitsm, Elitsm>();
    def_pickle<Pairwise, Pairwise>();
    def_pickle<NoPairwise, Pairwise>();
    def_pickle<Strategy>();
}

void define_center_placement(nb::module_ &main)
{
    auto m = main.def_submodule(
        "center",
        "Center placement: where the mean of the search distribution is placed at a restart.\n\n"
        "The optimizer holds one as `es.p.center_placement`, selected by "
        "`Modules.center_placement` (`options.CenterPlacement`). It is applied at every restart; "
        "the first run starts at `Settings.x0`, or at `Settings.center` (the center of the box; "
        "see it for infinite bounds) when x0 is not given.\n\n"
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
                              "The mean is `Settings.x0`, `Settings.center` (the "
                              "center of the box; see it for infinite bounds) when x0 is not given "
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
                                  "The mean is `Settings.center`: the midpoint (lb + ub) / 2 of "
                                  "a finite coordinate, with the rule for infinite bounds there "
                                  "(`CenterPlacement.CENTER`).")
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
        "repelling restarts. In a coordinate with an infinite bound, the candidates are drawn in "
        "x0 +- 2 sigma0 and the distances divided by 4 sigma0 (see the module).")
        .def(nb::init<>(), "Create the placement.");

    // Pickling and copying (#91)
    def_pickle<X0, Placement>();
    def_pickle<Uniform, Placement>();
    def_pickle<Zero, Placement>();
    def_pickle<Center, Placement>();
    def_pickle<MaximinTaboo, Placement>();
    def_pickle<NoveltyWeighted, Placement>();
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
        "`Modules.repelling_metric` (CURRENT, the default: the Mahalanobis distance in the "
        "current search distribution; EUCLIDEAN; ARCHIVED: `metric`) and divided by the current "
        "sigma. A sample is "
        "rejected when this distance is less than shrinkage^attempts * radius, with attempts "
        "the number of samples rejected so far in the population.")
        .def(nb::init<const Solution &, Float, const Matrix &>(),
             nb::arg("solution"),
             nb::arg("radius"),
             nb::arg("metric") = Matrix(),
             "Create a taboo point whose radius has no bounds (min_radius 0, max_radius inf; as "
             "`CoverageRepelling` does).\n\n"
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
        "sampled again when one rejects it (a mirrored pair as one). At a restart the "
        "candidate, the better of the run's evaluated center of mass and its best solution, is "
        "compared with the archive by hill-valley tests (Maree et al. 2018): 10 points on the "
        "segment from a taboo point to the candidate, which the optimizer evaluates one at a "
        "time; a point worse than both ends separates the two basins. The tests stop early when "
        "the next point would leave too little budget for the first generation of the new run. "
        "The subclasses add the candidate as a new taboo point or update the point whose basin "
        "it shares. The base class runs no tests and adds no points.")
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
        // A copy (#176): def_rw gave references into the vector, which a new taboo point
        // (reallocation) or the sort by criticality invalidated; a list held across step() read
        // freed memory
        .def_prop_rw(
            "archive",
            [](const Repelling &r) { return r.archive; },
            [](Repelling &r, const std::vector<TabooPoint> &archive) { r.archive = archive; },
            "The taboo points, a list of TabooPoint (sorted by criticality before each "
            "population). Reading returns a copy, which stays valid and does not follow the "
            "optimizer; writing to its elements does not change the archive. Assign a list to "
            "change it.")
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
        "each update all radii are set to r = (S n_rep / (coverage sigma0^d n))^(1/d) "
        "Gamma(d/2 + 1)^(1/d) / sqrt(pi), with S the volume of the box (10^d with an infinite "
        "bound), sigma0 the step size of the settings and n the number of runs so far, so that "
        "points found more often repel more (de Nobel et al. 2024, Eqs. (10), (11)). The "
        "distance to a point is divided by the current sigma (Eq. (8)), so r is in units of "
        "sigma, and sigma0^d makes it the volume in sigma0 units: at sigma = sigma0, with the "
        "Euclidean distance, the balls of the archive have a total volume of S / coverage when "
        "their n_rep sum to n.\n\n"
        "sigma0 is the problem's (`Settings.sigma0`), as Eq. (10) has it, also in BIPOP's small "
        "runs, which start at sigma0 10^(-2U). r grows with the box width over sigma0: a large "
        "box with a small sigma0 gives a radius of many sigma, which rejects every sample near "
        "a taboo point (`restart.TooMuchRepelling` then ends the run). With d, r tends to "
        "sqrt(d / (2 pi e)) (box width / sigma0): about 0.24 sqrt(d) times the ratio. Computed "
        "in logarithms, so it stays finite at large d.\n\n"
        "Before 2.0 the volume was divided by sigma0 instead of sigma0^d (as the paper's "
        "Eq. (10) writes it; the radius then changed when the box and sigma0 were scaled "
        "together), and r had an extra factor 1 / sqrt(d) (#158).")
        .def(nb::init<>(), "Create a coverage repelling with an empty archive.");

    nb::class_<AdaptiveRepelling, Repelling>(
        m,
        "AdaptiveRepelling",
        "Repelling with adaptive radii (`RepellingType.ADAPTIVE`).\n\n"
        "Experimental: modcma's own work in progress, without a published source. In the "
        "experiment of #158 (BBOB f21 and f22, d = 5 to 40, RESTART and IPOP, 240 runs) it "
        "solved 99 runs, against 107 without repelling and 144 with COVERAGE.\n\n"
        "A new taboo point gets the radius initial_radius_factor / sqrt(d) and the bounds "
        "[min_radius_factor, max_radius_factor] / sqrt(d). When a run ends in the basin of a "
        "point again (the first one in the archive that shares it), its radius grows by "
        "exp(grow_eta * min(w, max_waste_ratio)), with w the evaluations of that run divided by "
        "their running average, which includes that run (`average_restart_evaluations`: every "
        "run counted from the previous restart, the first from the start, also a run without a "
        "finite result; the average starts from the first run's length. Before 2.0 the first "
        "run counted as one evaluation and the average started at 1, #121). The point also "
        "keeps the better solution. After each population, the radius of a critical point "
        "shrinks by shrink_factor when the share of rejected samples of the population exceeds "
        "max_rejection_rate and the point's own rejection rate exceeds "
        "local_max_rejection_rate (after at least min_checks_before_local_shrink checks). "
        "A radius that grows or shrinks is clamped to [min_radius, max_radius] of its point.")
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

    // Pickling and copying (#91)
    def_pickle<TabooPoint>();
    def_pickle<Repelling, Repelling>();
    def_pickle<CoverageRepelling, Repelling>();
    def_pickle<AdaptiveRepelling, Repelling>();
    def_pickle<NoRepelling, Repelling>();
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
        "The samples are distributed as x = m + sigma * B (d * z), z ~ N(0, I), whatever factor "
        "of C the adaptation samples with, so they have the covariance sigma^2 C with "
        "C = B diag(d)^2 B^T: the one-sigma ellipsoid around m has the semi-axes sigma * d_i "
        "along the columns B_i.\n\n"
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
            "in the order of the columns of `B`: ascending, the order of the eigensolver. For the "
            "diagonal adaptations (SEPARABLE, NONE) in the order of the coordinates.")
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
            "(`inverse_hessian_scalar_correction` fits the factor). So C^-1 estimates the shape of "
            "the Hessian, not its scale; sigma does not give the scale, it follows the distance to "
            "the optimum. The relation needs a "
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
        "updates the matrix. `compute_y` and `invert_y` apply A and A^-1.\n\n"
        "The matrices start at the identity. With `Settings.C0` the optimizer starts every run "
        "from a factor F of C0 instead, F F^T = C0 (#89): the eigendecomposition for COVARIANCE, "
        "the lower Cholesky factor for COVARIANCE_NO_EIGV, CHOLESKY and CMSA, the symmetric "
        "square root C0^1/2 for MATRIX and NATURAL_GRADIENT, and the square roots of the "
        "diagonal of C0 for SEPARABLE; NONE rejects C0.")
        .def_rw("m",
                &Adaptation::m,
                "The mean (center) of the search distribution, shape (d,).")
        .def_rw("m_old", &Adaptation::m_old, "The mean of the previous generation, shape (d,).")
        .def_rw("dm",
                &Adaptation::dm,
                "The last step of the mean in units of the step size, (m - m_old) / s, shape "
                "(d,), with s per coordinate the step size recombined from the mu selected "
                "individuals, sum_i w_i S_i (Hansen 2016, Eq. (24)): sigma when they all have "
                "the step size sigma; sigma raised by the lower bound of an integer variable "
                "(then dm = sum_i w_i y_i, as Marty et al. 2024); the recombined sampled step "
                "size with sample_sigma, SA and LPXNES.")
        .def_rw("ps",
                &Adaptation::ps,
                "The evolution path of the step size (conjugate evolution path), shape (d,): "
                "ps = (1 - cs_path) ps + sqrt(cs_path (2 - cs_path) mueff_mean) dz, with the rates "
                "of `es.p.weights` (C^-1/2 dm instead of dz in `CovarianceAdaptation`, not in its "
                "subclass `CovarianceNoEigvAdaptation`). Read by CSA, by h_sigma (`hs`) and by the "
                "rank-one term of MATRIX. Starts at zeros, also after a restart.")
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
                "the step size of integer variables, and by the restart criteria TolX and "
                "NoEffectCoord (and MaxSigma, MaxDSigma and MinDSigma where `keeps_metric()` is "
                "False, #80); a NaN or inf in it fails the matrix update.")
        .def(
            "metric",
            [](Adaptation &a) { return a.metric().copy(); },
            "The metric of the sampled distribution: its axis lengths d and eigenvectors B, "
            "without sigma (see `Metric`).\n\n"
            "`CovarianceAdaptation` (not its subclass `CovarianceNoEigvAdaptation`), "
            "`SeparableAdaptation` and `NoAdaptation` keep it: it costs nothing. The others "
            "decompose C = F F^T of their sampling factor F (M, or A; CMSA its C) when asked, "
            "O(d^3), and reuse the result while `factor_version` is unchanged; nothing is computed "
            "unless something asks. A value written from Python to their matrices is seen after "
            "the next change of the factor (an update or a restart). NaN when the factor is not "
            "finite or the eigensolver fails.\n\n"
            "Returns\n"
            "-------\n"
            "Metric\n"
            "    A copy.")
        .def("keeps_metric",
             &Adaptation::keeps_metric,
             "Whether `metric()` costs nothing: True for `CovarianceAdaptation` (not "
             "COVARIANCE_NO_EIGV), `SeparableAdaptation` and `NoAdaptation`, which keep it up to "
             "date; False for the others, which decompose their matrix on request, O(d^3).\n\n"
             "By default the restart criteria read the metric only where this is True (#80). "
             "ConditionC and NoEffectAxis are created only then. MaxSigma, MaxDSigma and "
             "MinDSigma take the longest and the shortest axis from the metric then, and "
             "otherwise from the diagonal of C: the square roots of the largest and the smallest "
             "entry of `coordinate_wise_variances`.")
        .def_ro("factor_version",
                &Adaptation::factor_version,
                "A counter increased wherever the sampling factor changes (an update of the "
                "matrix, a decomposition, a restart, the start from `Settings.C0`): `metric()` "
                "recomputes only when it changed. A write to the matrices from Python does not "
                "increase it.")
        .def("adapt_evolution_paths",
             &Adaptation::adapt_evolution_paths,
             nb::arg("pop"),
             nb::arg("weights"),
             nb::arg("stats"),
             nb::arg("settings"),
             nb::arg("mu"),
             "Update dm, dz and the evolution paths after selection.\n\n"
             "Computes dm = (m - m_old) / sum_i w_i S_i (per coordinate, over the mu best) and "
             "dz = sum_i w_i z_i over the mu "
             "best (every point, with the utilities of `RecombinationWeights.UTILITY`), then "
             "the update of the subclass, `adapt_evolution_paths_inner`. Called by "
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
             "The update of the evolution paths of the subclass (ps, and h_sigma (`hs`) and pc "
             "where there are), without recomputing dm and dz; the second part of "
             "`adapt_evolution_paths`. `NaturalGradientAdaptation` computes its natural gradient "
             "here (`compute_gradients`).\n\n"
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
             "Reset the distribution for a restart: m = `settings.x0` (`settings.center` when it "
             "is not given), the paths to zero and the matrices to the identity. The optimizer "
             "then starts the matrix from `Settings.C0` when it is given (every run starts from "
             "it, #89), and the center placement sets m.\n\n"
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
             "`MatrixAdaptation` and `NaturalGradientAdaptation` first compute their inverse "
             "from the matrix when it is out of date (O(d^3), see `MatrixAdaptation.M_inv`).\n\n"
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
        "functions, or as a baseline. Settings switches `Modules.active` off with it, and "
        "rejects `Settings.C0`.");
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
        "selected steps (with `Modules.active`, negative weights for the worst steps: with "
        "elitism the worst points of the (mu + lambda) pool, #161), and is "
        "decomposed as C = B diag(d)^2 B^T; samples are y = B diag(d) z. C is updated every "
        "generation; the eigendecomposition only every `es.p.weights.lazy_update_interval` = "
        "1 / (10 d (c1 + cmu)) generations (every generation unless d is large). The "
        "eigendecomposition costs O(d^3) (LAPACK from d = `constants.lapack_min_dim`), the "
        "rank updates O(d^2 lambda). Settings replaces it by COVARIANCE_NO_EIGV when the "
        "step-size adaptation is not CSA, also for lambda0 = 1, which runs SR (unless "
        "`always_compute_eigv`). Hansen & Ostermeier "
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
                "fast), which stalls the update of pc; True otherwise. With "
                "`StepSizeAdaptation.XNES` and `Modules.active` (not with UTILITY weights), False "
                "also drops the negative weights of the generation's update (#175).")
        .def("adapt_covariance_matrix",
             &CovarianceAdaptation::adapt_covariance_matrix,
             nb::arg("weights"),
             nb::arg("modules"),
             nb::arg("population"),
             nb::arg("mu"),
             "Update C with the rank-one (pc) and rank-mu (selected steps) updates.\n\n"
             "With `modules.active` the whole population is used, the negative weights of the "
             "worst steps rescaled by d / ||C^-1/2 y_i||^2 (Hansen 2016, eq. 46); otherwise "
             "the mu best. With elitism (`population.n_pool`) the negative weights go to the "
             "points that the selection dropped, the worst of the (mu + lambda) pool (#161). "
             "Does not decompose C.\n\n"
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
        "applied per coordinate, with pycma's separable learning rates (c1_sep, cmu_sep: see "
        "`Weights.c1`, `Weights.cmu`); samples are y = d * z. With `Modules.active` the "
        "negative weights are multiplied by d / sum_j (y_ij / d_j)^2, the Mahalanobis length in "
        "the diagonal C, as in pycma. O(d lambda) per generation: for separable functions, and "
        "for high dimension, where a full matrix is slow to learn. Ros & Hansen (2008).");
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
        .def_rw("hs",
                &SeparableAdaptation::hs,
                "The indicator h_sigma: False when ||ps|| is long (Hansen 2016, after Eq. (45)), "
                "which stalls the update of pc; True otherwise. With `StepSizeAdaptation.XNES` "
                "and `Modules.active` (not with UTILITY weights), False also drops the negative "
                "weights of the generation's update (#175).")
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
        "evolution path pc or a decomposition: M = (1 - c1/2 - cmu/2 sum(w)) M + (c1/2) (M ps) "
        "ps^T + (cmu/2) sum_i w_i y_i z_i^T, with sum(w) = 1 for the positive weights. O(d^2 "
        "lambda) per generation (BLAS from d = `constants.blas_min_dim`), so it suits high "
        "dimension. Beyer & Sendhoff (2017).\n\n"
        "MA-ES has no active update. With `Modules.active` this is the active update of the "
        "CMA-ES (Hansen 2016, Eq. (47)) written for M, as MA-ES derives its own update: sum(w) "
        "includes the negative weights, and the negative weights are multiplied by d / "
        "||z_i||^2. With elitism the negative weights go to the worst points of the (mu + "
        "lambda) pool, as with every matrix adaptation (#161).\n\n"
        "With a step-size adaptation other than CSA, h_sigma (`hs`) skips the rank-one term "
        "in a generation in which ||ps|| is long, as COVARIANCE stalls pc (Hansen 2016, after "
        "Eq. (45); #160). MA-ES has no h_sigma: with CSA, which updates sigma from the same "
        "path, the update is MA-ES's.");
    adaptation_init(matrix);
    matrix
        .def_rw("M",
                &MatrixAdaptation::M,
                "The sampling matrix, y = M z, shape (d, d) (the identity at the start).")
        .def_rw("M_inv",
                &MatrixAdaptation::M_inv,
                "The inverse of M, shape (d, d).\n\n"
                "With elitism or TPA_INPLACE (lambda > 1) it is updated with M every "
                "generation: M <- M F with "
                "F = a I + U D U^T (U = [ps, the selected z], D = diag(c1/2, cmu/2 w)), so M_inv "
                "<- F^-1 M_inv by the Woodbury identity, O(k d^2); up to d = "
                "`inverse_recompute_max_dim` (30) it is computed exactly from M after every "
                "`inverse_recompute_interval` (400) generations of a run, which bounds the "
                "rounding error that the update accumulates with the active update at small d "
                "(the maintainer's choice, #74). When the "
                "update cannot be applied (`inverse_fallbacks`), and without elitism, it is "
                "computed only when `invert_y` needs it (injected and repaired points; an LU "
                "inverse from d = `constants.blas_min_dim`, else the pseudo-inverse), so it can "
                "lag behind M. Before 2.0 the update with elitism was not the inverse of M F, and "
                "drifted from the inverse of M.")
        .def_rw("hs",
                &MatrixAdaptation::hs,
                "The indicator h_sigma: False when ||ps|| is long (Hansen 2016, after Eq. (45)), "
                "which skips the rank-one term of the generation; always True with CSA and in "
                "the (1+1)-ES (#160). With `StepSizeAdaptation.XNES` and `Modules.active` (not "
                "with UTILITY weights), False also drops the negative weights of the generation's "
                "update (#175).")
        .def_ro("inverse_fallbacks",
                &MatrixAdaptation::inverse_fallbacks,
                nb::name("inverse_fallbacks"),
                "The generations with elitism in which the Woodbury update of `M_inv` could not "
                "be applied, since the adaptation was created (read-only): the decay a = 1 - c1/2 "
                "- cmu/2 sum(w) was not positive, or the k x k system was singular or badly "
                "conditioned (the ratio of its smallest to its largest pivot below 1e-8). M_inv "
                "is then computed when `invert_y` needs it, and the update continues from it.")
        .def_ro_static("inverse_recompute_interval",
                       &MatrixAdaptation::inverse_recompute_interval,
                       "With elitism and d <= `inverse_recompute_max_dim`, M_inv is computed "
                       "exactly from M after every this many generations of a run, and updated by "
                       "the Woodbury identity in between (400, the maintainer's choice on #74). "
                       "Without it, the largest entry of M M_inv - I over 5000 generations reached "
                       "2.4 to 26 with the active update at d = 10 and 1e-8 at d = 20; with it, at "
                       "most 1.4e-9 at d = 10.")
        .def_ro_static("inverse_recompute_max_dim",
                       &MatrixAdaptation::inverse_recompute_max_dim,
                       "The largest dimension with the exact recomputation of M_inv every "
                       "`inverse_recompute_interval` generations (30, the maintainer's choice on "
                       "#74): the smallest d at which the drift without it stayed below 1e-10 "
                       "over 5000 generations in every run measured (at most 4.3e-11 at d = 30, "
                       "below 1e-12 from d = 40 on, with the active update).")
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
        "O(d^2 lambda) per generation and O(d^2) memory. Krause et al. (2016).\n\n"
        "With `Modules.active` the negative weights, multiplied by d / ||A^-1 y_i||^2 as in "
        "COVARIANCE, are rank-one downdates, and A is scaled by sqrt(1 - c1 - cmu sum(w)) with "
        "the negative weights in the sum. A downdate whose result would not be positive "
        "definite is skipped (`skipped_downdates`). With elitism the negative weights go to "
        "the worst points of the (mu + lambda) pool (#161).\n\n"
        "With a step-size adaptation other than CSA, h_sigma (`hs`) stalls the update of pc "
        "and A is scaled by sqrt(1 + c1 delta(h_sigma) - c1 - cmu sum(w)), as in COVARIANCE "
        "(Hansen 2016, after Eq. (45); #160).");
    adaptation_init(cholesky);
    cholesky
        .def_rw("A",
                &CholeskyAdaptation::A,
                "The lower triangular Cholesky factor of C, y = A z, shape (d, d) (the identity "
                "at the start).")
        .def_rw("pc",
                &CholeskyAdaptation::pc,
                "The evolution path of the rank-one update, shape (d,) (zeros at the start).")
        .def_rw("hs",
                &CholeskyAdaptation::hs,
                "The indicator h_sigma: False when ||ps|| is long (Hansen 2016, after Eq. (45)), "
                "which stalls the update of pc; always True with CSA (#160). With "
                "`StepSizeAdaptation.XNES` and `Modules.active` (not with UTILITY weights), False "
                "also drops the negative weights of the generation's update (#175).")
        .def_ro("skipped_downdates",
                &CholeskyAdaptation::skipped_downdates,
                nb::name("skipped_downdates"),
                "The rank-one downdates of the active update that were skipped, since the "
                "adaptation was created (read-only; restarts do not reset it).\n\n"
                "A downdate is skipped when its result would not be positive definite: the "
                "factor from before the downdates of the generation is restored and they are "
                "applied again without it. The rescaled negative weights prevent this in exact "
                "arithmetic; UTILITY's weights do not.");

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
        "`es.p.weights.cc`, xNES's eta_B / 2 by default. For lambda < d and d >= 16 the matrix "
        "exponential is computed "
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
                "weights for the worst; with elitism, the worst of the (mu + lambda) pool, "
                "#161).")
        .def_rw("gradient_shift",
                &NaturalGradientAdaptation::gradient_shift,
                "The shift alpha of G: the sum of the weights plus `sigma_g`, which makes G "
                "trace-free (the sum of the weights alone in a (1+1)-ES).")
        .def_rw("sigma_g",
                &NaturalGradientAdaptation::sigma_g,
                "trace(G) / d of the gradient before the shift: its isotropic (step-size) part, "
                "removed from G and used by `StepSizeAdaptation.XNES`, "
                "sigma *= exp(cs / 2 * sigma_g). With elitism XNES computes it from the "
                "offspring alone instead (#172).")
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

    // Pickling and copying (#91)
    def_pickle<MetricCopy>();
    def_pickle<None, Adaptation>();
    def_pickle<CovarianceAdaptation, Adaptation>();
    def_pickle<SeparableAdaptation, Adaptation>();
    def_pickle<MatrixAdaptation, Adaptation>();
    def_pickle<CholeskyAdaptation, Adaptation>();
    def_pickle<SelfAdaptation, Adaptation>();
    def_pickle<CovarianceNoEigvAdaptation, Adaptation>();
    def_pickle<NaturalGradientAdaptation, Adaptation>();
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
        .def("correct_point",
             &BoundCorrection::correct_point,
             nb::arg("xi"),
             nb::arg("sigma"),
             nb::arg("settings"),
             "A point that is not sampled from the search distribution moved into "
             "[lb, ub].\n\n"
             "TPA moves its two points toward the mean first, both by the same factor, until "
             "both are in the box; it uses this only when the mean itself is outside.\n\n"
             "As `correct_x` for the coordinates outside the bounds, without counting it in "
             "`n_out_of_bounds`. RESAMPLE cannot sample such a point again: it sets the "
             "coordinates outside on the bound they crossed, as SATURATE does. With NONE the "
             "point is returned "
             "as it is.\n\n"
             "Parameters\n"
             "----------\n"
             "xi : numpy.ndarray\n"
             "    The point, shape (d,).\n"
             "sigma : float\n"
             "    The step size (used by COTN).\n"
             "settings : Settings\n"
             "    The settings with the bounds lb and ub and the method "
             "(`modules.bound_correction`).\n\n"
             "Returns\n"
             "-------\n"
             "numpy.ndarray\n"
             "    The point inside the box (unless the method is NONE), shape (d,).")
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

    // Pickling and copying (#91)
    def_pickle<NoCorrection, BoundCorrection>();
    def_pickle<Resample, BoundCorrection>();
    def_pickle<COTN, BoundCorrection>();
    def_pickle<Mirror, BoundCorrection>();
    def_pickle<UniformResample, BoundCorrection>();
    def_pickle<Saturate, BoundCorrection>();
    def_pickle<Toroidal, BoundCorrection>();
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
        "(2021); `TPAInPlace` (TPA_INPLACE, new in 2.0) in its own docstring.");
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
             "    The population can stop after individual i >= floor(mu * seq_cutoff_factor) "
             "(0-based), so at least floor(mu * seq_cutoff_factor) + 1 individuals are "
             "evaluated (default 1).")
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
        "Sampled step sizes: each individual gets its own step size, sigma_i = "
        "sigma * exp(beta n_i) with n_i standard normal, the same in every coordinate, so the "
        "point m + sigma_i y_i lies along y_i (`Modules.sample_sigma`; always used by the "
        "step-size adaptations LPXNES and SA). One step size per individual, as in CMSA-ES "
        "(Beyer & Sendhoff 2008) and lp-xNES (Krause, Glasmachers & Igel 2017, Sec. 4.4).\n\n"
        "beta = `es.p.weights.beta`: 1 / sqrt(2 d) with SA (CMSA's tau; Spettel, Beyer & "
        "Hellwig 2018, Table I), ln 2 / (sqrt(d) ln(max(d, 2))) with LPXNES (Krause et al. 2017, "
        "Table 3), 1 / sqrt(2 mueff) otherwise. The step sizes are the columns of "
        "`es.p.pop.S`; the lower bound of an integer variable can raise single coordinates.")
        .def(nb::init<Float>(),
             nb::arg("dimension"),
             "Create the sampler. It draws from the global generator (`set_seed`); the "
             "sampler of an optimizer draws from the optimizer's own.\n\n"
             "Parameters\n"
             "----------\n"
             "dimension : float\n"
             "    The dimension d (not used: the step size is one number per individual).")
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
             "Set the step sizes `pop.S` of the population from sigma (one per individual, "
             "constant in its column), then apply the lower bounds of the integer "
             "variables.\n\n"
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
             "    The dimension d (not used).");

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
                "The smoothed success measure of TPA, TPA_INPLACE, MSR and PSR (default 0): "
                "TPA multiplies sigma by exp(s), the others by exp(s / damps).");

    auto ssa_csa = nb::class_<CSA, Strategy>(
        m,
        "CSA",
        "Cumulative step-size adaptation (`StepSizeAdaptation.CSA`, the default).\n\n"
        "sigma *= exp(l), l = (cs / damps) (||ps|| / E||z|| - 1) clamped to [-1, 1] (as in "
        "pycma): sigma grows when the evolution path ps is longer than under random selection "
        "and shrinks when it is shorter. E||z|| is the expected length of a sample of the "
        "sampler (`Weights.expected_length_z`). By default cs = (mueff + 2) / (d + mueff + 5) "
        "and damps = 1 + 2 max(0, sqrt((mueff - 1) / (d + 1)) - 1) + cs (Hansen 2016, Eqs. "
        "(44), (55)); modcma 1.1 and 1.2 used pycma's denominator d + mueff + 3 for cs (#138). "
        "Hansen & Ostermeier (2001).");
    strategy_init(ssa_csa);
    auto ssa_tpa = nb::class_<TPA, Strategy>(
        m,
        "TPA",
        "Two-point step-size adaptation (`StepSizeAdaptation.TPA`).\n\n"
        "With every population (in its first batch) the two points m + 0.5 sigma_old dm and "
        "m - 0.5 sigma_old dm are evaluated: half the last step of the mean, sigma_old dm, "
        "with the step size that sampled it (Hansen, CMA-ES with Two-Point Step-Size "
        "Adaptation, 2008, Eqs. (8), (9), alpha = 0.5). s = (1 - cs) s + cs r, with r = "
        "-a_tpa when the point behind is better, a_tpa + b_tpa when the point ahead is better "
        "and 0 for a tie (as after a restart, when dm = 0), and sigma *= exp(s), with cs = 0.3 "
        "by default (Hansen 2008, Eqs. (10) to (13); `damps` is not used). Costs two "
        "evaluations per generation. A tie gives r = 0, as the sign of the rank difference in "
        "pycma; Hansen (2008, Eq. (10)) counts it as a success, which made sigma grow by 16 % "
        "after every restart (#78). Before 2.0 the points were "
        "m +- sigma dm with the new sigma, about one step of the mean (#167). With a bound "
        "correction other than "
        "NONE both points move toward m by the same factor until both are in the box, so the "
        "test steps stay symmetric about m (Hansen 2008, Sec. 2.2; pycma's boundary handler "
        "moves each point on its own); `BoundCorrection.correct_point` corrects them only "
        "when m itself is outside the box.\n\n"
        "With elitism (#179): after a generation whose (mu + lambda) selection kept every "
        "parent (none of the mu selected points is an offspring), r = -a_tpa and the two points "
        "are not evaluated: the mean did not move, so both would be m up to rounding, and "
        "their comparison by rounding made sigma diverge. The test of the mean shift of "
        "generation t is made in generation t + 1, and so is this failure. The budget still "
        "reserves the two points in that generation. No TPA source covers elitism: Hansen "
        "(2008) and Hansen, Atamna & Auger (2014, Algorithm 3) define TPA for comma "
        "selection, and pycma's CMA_elitist adds only the best solution so far, so the mean "
        "always moves. The rule follows the success rules of elitist ESs (Rechenberg's 1/5th "
        "rule; the (1+1)-CMA-ES, Igel, Suttorp & Hansen 2006), in which a generation without "
        "a successful offspring is a failure.")
        .def_rw("a_tpa", &TPA::a_tpa, "The magnitude of r (default 0.5).")
        .def_rw("b_tpa",
                &TPA::b_tpa,
                "The offset added to r when the forward point is better (default 0). Hansen "
                "(2008, Eq. (10)) adds his beta to the other case, r = -a_tpa + beta; with the "
                "default 0 the two agree.")
        .def_rw("rank_tpa",
                &TPA::rank_tpa,
                "The last r: -a_tpa, a_tpa + b_tpa or 0 (default 0); -a_tpa also after a "
                "generation whose elitist selection kept every parent.")
        .def_rw("sigma_old",
                &TPA::sigma_old,
                "The step size that sampled the last generation, before the update changed "
                "it: the test steps are 0.5 sigma_old dm. At the start of a run, the step size "
                "the run starts with (sigma0, or the step size of the restart).");
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
        "generation. Ait ElHara, Auger & Hansen (2013).\n\n"
        "lambda is the number of ranked individuals of each generation: fewer after "
        "sequential selection stopped it, and with PAIRWISE only the winners of the mirrored "
        "pairs (Auger et al. 2011, Alg. 1).")
        .def_rw("quantile",
                &MSR::quantile,
                "The position j / lambda of the previous generation's value that the offspring "
                "are compared with (default 0.3).");
    strategy_init(ssa_msr);
    auto ssa_psr = nb::class_<PSR, Strategy>(
        m,
        "PSR",
        "Population success rule (`StepSizeAdaptation.PSR`).\n\n"
        "The n best individuals of the current and of the previous population are ranked "
        "together, n the smaller number of ranked individuals of the two (fewer than lambda "
        "after a stop of sequential selection; with PAIRWISE the winners of the mirrored "
        "pairs). With z the difference of their summed ranks (previous minus current) divided "
        "by n^2, minus success_ratio: "
        "s = (1 - cs) s + cs z and sigma *= exp(s / damps), with cs = 0.3 and damps = 1 by "
        "default (Loshchilov 2014, Alg. 6). Equal values share their mean rank (mid-ranks; "
        "the paper does not say), so a tie counts for neither population; with elitism the "
        "parents are in both.")
        .def_rw("success_ratio", &PSR::success_ratio, "The target success ratio (default 0.25).");
    strategy_init(ssa_psr);
    auto ssa_xnes = nb::class_<XNES, Strategy>(
        m,
        "XNES",
        "The step-size update of xNES (`StepSizeAdaptation.XNES`).\n\n"
        "With NATURAL_GRADIENT: sigma *= exp(cs / 2 * sigma_g), sigma_g the trace part of the "
        "natural gradient (`NaturalGradientAdaptation.sigma_g`), cs = xNES's eta_sigma = (9 + "
        "3 ln(d)) / (5 d sqrt(d)) (Glasmachers et al. 2010). With the other matrix "
        "adaptations, the xNES SA of Krause, Glasmachers & Igel (2017): sigma *= exp(cs / "
        "sqrt(d) * sum_i w_i (||z_i||^2 - d)) over the positive weights (`Weights.positive`, "
        "summing to 1), cs = mueff / (2 ln(max(2, d)) sqrt(d)).\n\n"
        "With elitism both forms read the offspring alone, ranked among themselves (the "
        "losers of pairwise selection last), with the same weights: xNES's gradient and xNES "
        "SA are estimates from the samples of the current distribution (Glasmachers et al. "
        "2010, Algorithm 1; Krause et al. 2017, Sec. 4.2). The parents' z, recomputed against "
        "the mean they recombined, are short, and as part of the mu best they kept sigma 3 to "
        "14 times too small (#172).");
    strategy_init(ssa_xnes);
    auto ssa_mxnes = nb::class_<MXNES, Strategy>(
        m,
        "MXNES",
        "A variant of the xNES step-size update on the mean step "
        "(`StepSizeAdaptation.MXNES`).\n\n"
        "sigma *= exp((cs / d) (mueff ||dz||^2 - d)), with dz the weighted mean of the "
        "selected z (`Adaptation.dz`) and cs = 1 by default (Krause et al. 2017, Sec. 4.3, "
        "Table 3): sigma grows when the mean step is longer than under random selection. "
        "With UTILITY weights dz recombines all lambda z with their utilities, and mueff is "
        "theirs (`Weights.mueff_mean`). "
        "m-xNES of Krause, Glasmachers & Igel (2017), "
        "\"Qualitative and quantitative assessment of step size adaptation "
        "rules\".");
    strategy_init(ssa_mxnes);
    auto ssa_lpxnes = nb::class_<LPXNES, Strategy>(
        m,
        "LPXNES",
        "A variant of the xNES step-size update on sampled step sizes "
        "(`StepSizeAdaptation.LPXNES`).\n\n"
        "Every individual samples its own step size sigma_i (a `SigmaSampler` is always "
        "used, with beta = ln 2 / (sqrt(d) ln(max(d, 2)))); sigma *= exp(cs sum_i w_i "
        "ln(sigma_i / sigma)) over the positive weights (`Weights.positive`, summing to 1; "
        "ln sigma_i: the mean of the logarithms of the column of `pop.S`), which moves "
        "sigma towards the step sizes of the better individuals; cs = min(1, 9 mueff / (10 "
        "sqrt(d))). The rate is that of Krause et al.; the cap is modcma's: above 1 the "
        "update overshoots the weighted geometric mean of the step sizes (1.29 at d = 2, "
        "1.05 at d = 5 with the default lambda). lp-xNES of Krause, Glasmachers & Igel "
        "(2017), Sec. 4.4, "
        "\"Qualitative and quantitative assessment of step size adaptation "
        "rules\".");
    strategy_init(ssa_lpxnes);
    auto ssa_sr = nb::class_<SR, Strategy>(
        m,
        "SR",
        "The success rule of the (1+1)-CMA-ES (`StepSizeAdaptation.SR`; always used when "
        "lambda = 1).\n\n"
        "sigma *= exp((p_s - p_t) / (damps (1 - p_t))), with p_s the rate of successes, "
        "smoothed with weight 1/12 (`es.p.stats.success_ratio`): a generation is a success "
        "when its best individual is better than the best of the run so far (with elitism, the "
        "parent). The target p_t = 2/11 and damps = 1 + d / 2 by default. Igel et al. "
        "(2006).\n\n"
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
        "Every individual samples its own step size sigma_i = sigma exp(n_i / sqrt(2 d)) (a "
        "`SigmaSampler` is always used; CMSA's tau = 1 / sqrt(2 d)); sigma is multiplied by "
        "the weighted geometric mean of the step sizes of the mu selected individuals "
        "(`Weights.positive`) divided by the geometric mean of the step sizes of all "
        "offspring of the generation, taken before elitism adds the parents (the CMSA-ES "
        "operator of Beyer & Sendhoff 2008, weighted, with the correction for elitism of "
        "Ahrari et al. 2017, as the framework paper describes). The step size of an individual "
        "is the geometric mean of its column of `pop.S`; a parent of elitism has the "
        "generation's sigma. With SA the default mu is lambda / 4.");
    strategy_init(ssa_sa);
    auto ssa_tpa_inplace = nb::class_<TPAInPlace, Strategy>(
        m,
        "TPAInPlace",
        "Two-point step-size adaptation within the population "
        "(`StepSizeAdaptation.TPA_INPLACE`, #173).\n\n"
        "pycma 4.6.0's CMAAdaptSigmaTPA (sigma_adaptation.py, evolution_strategy.py:2251-2254, "
        "2575-2576), after Hansen, Atamna & Auger (2014), How to Assess Step-Size Adaptation "
        "Mechanisms in Randomised Search, Eq. (6) and Algorithm 3. From the second generation "
        "of a run on (in a population of at least two), the first two offspring are a mirrored "
        "pair along the last mean shift: "
        "z_1 = ||n|| A^-1 dm / ||A^-1 dm|| with n an ordinary sample of the sampler, z_2 = -z_1 "
        "(both at the mean when it did not move); they are part of the lambda offspring, so "
        "TPA_INPLACE needs no extra evaluations. Once the offspring are evaluated, z = "
        "(#{f < f_2} - #{f < f_1}) / (n - 1) over the n offspring (the parents of elitism not "
        "included), and s = (1 - cs) s + cs sign(z) |z|^0.5, sigma *= exp(s / damps), with cs "
        "= 0.3 and damps = 0.7 + 2 ln(d) + 2 ln(max(1, lambda - d)) by default.\n\n"
        "Where pycma departs from the paper, and modcma follows pycma: |z|^0.5 instead of z; "
        "the damping (the paper: sqrt(d), tuned on the sphere with the default lambda); and "
        "the update in the first generation of a run, from the ranks of two ordinary samples "
        "(the paper updates from the second generation on). A^-1 dm costs O(d^2) with most "
        "matrix adaptations (a triangular solve, or B^T and D); MATRIX keeps M^-1 up to date "
        "for it, as with elitism; "
        "NATURAL_GRADIENT inverts A every generation (O(d^3)). With sampled step sizes the "
        "second point takes the step size of the first; bound correction and repelling treat "
        "the two as a pair, as a mirrored pair.\n\n"
        "With elitism (#179): after a generation whose (mu + lambda) selection kept every "
        "parent (none of the mu selected points is an offspring), the mean did not move, and "
        "the pair would lie along its rounding noise: the generation samples ordinary "
        "offspring instead, and z = -1. The pair for the mean shift of generation t is placed "
        "in generation t + 1, and so is this failure. No source covers elitism: Hansen, "
        "Atamna & Auger (2014, Algorithm 3) define TPA for the (mu/mu, lambda)-ES, and pycma's "
        "CMA_elitist adds only the best solution so far, so its mean always moves. The rule "
        "follows the success rules of elitist ESs (Rechenberg's 1/5th rule; the (1+1)-CMA-ES, "
        "Igel, Suttorp & Hansen 2006).")
        .def_rw("rank_difference",
                &TPAInPlace::rank_difference,
                "The last normalised rank difference z of the two test points, in [-1, 1] "
                "(positive when the point ahead, along the mean shift, is the better one; 0 "
                "before the first update).");
    strategy_init(ssa_tpa_inplace);

    // Pickling and copying (#91)
    def_pickle<ThresholdConvergence, ThresholdConvergence>();
    def_pickle<NoThresholdConvergence, ThresholdConvergence>();
    def_pickle<SequentialSelection, SequentialSelection>();
    def_pickle<NoSequentialSelection, SequentialSelection>();
    def_pickle<SigmaSampler, SigmaSampler>();
    def_pickle<NoSigmaSampler, SigmaSampler>();
    def_pickle<Strategy, Strategy>();
    def_pickle<CSA, Strategy>();
    def_pickle<TPA, Strategy>();
    def_pickle<MSR, Strategy>();
    def_pickle<PSR, Strategy>();
    def_pickle<XNES, Strategy>();
    def_pickle<MXNES, Strategy>();
    def_pickle<LPXNES, Strategy>();
    def_pickle<SR, Strategy>();
    def_pickle<SA, Strategy>();
    def_pickle<TPAInPlace, Strategy>();
}
} // namespace modcma_python
