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

std::optional<Indices> to_indices(const std::optional<std::vector<int>> &idx)
{
    if (!idx)
        return std::nullopt;
    return Eigen::Map<const Indices>(idx->data(), static_cast<Eigen::Index>(idx->size()));
}
} // namespace

void define_population(nb::module_ &main)
{
    nb::class_<Population>(main, "Population")
        .def(nb::init<size_t, size_t>(), nb::arg("dimension"), nb::arg("n"))
        .def(nb::init<Matrix, Matrix, Matrix, Vector, Matrix>(),
             nb::arg("X"),
             nb::arg("Z"),
             nb::arg("Y"),
             nb::arg("f"),
             nb::arg("S"))
        .def("sort", &Population::sort)
        .def("resize_cols", &Population::resize_cols, nb::arg("size"))
        .def("keep_only", &Population::keep_only, nb::arg("idx"))
        .def_prop_ro("n_finite", &Population::n_finite)
        .def("__add__", &Population::operator+=, nb::arg("other"))
        // The arrays are views of the active columns 0..n-1 (fewer than the buffers hold after
        // sequential selection stopped a generation early); assigning replaces the buffer.
        .def_prop_rw("X", active_columns(&Population::X), assign_buffer(&Population::X), view)
        .def_prop_rw("X_transformed",
                     active_columns(&Population::X_transformed),
                     assign_buffer(&Population::X_transformed),
                     view)
        .def_prop_rw("Z", active_columns(&Population::Z), assign_buffer(&Population::Z), view)
        .def_prop_rw("Y", active_columns(&Population::Y), assign_buffer(&Population::Y), view)
        .def_prop_rw("S", active_columns(&Population::S), assign_buffer(&Population::S), view)
        .def_prop_rw("f", active_entries(&Population::f), assign_buffer(&Population::f), view)
        .def_prop_rw("t", active_entries(&Population::t), assign_buffer(&Population::t), view)
        .def_rw("d", &Population::d)
        .def_rw("n", &Population::n)
        .def_prop_ro("capacity", &Population::capacity);
}

void define_parameters(nb::module_ &main)
{
    auto m = main.def_submodule("parameters");
    using namespace parameters;

    nb::class_<Modules>(m, "Modules")
        .def(nb::init<>())
        .def_prop_rw("elitist", bool_getter(&Modules::elitist), bool_setter(&Modules::elitist))
        .def_prop_rw("active", bool_getter(&Modules::active), bool_setter(&Modules::active))
        .def_prop_rw("orthogonal", bool_getter(&Modules::orthogonal), bool_setter(&Modules::orthogonal))
        .def_prop_rw("sequential_selection", bool_getter(&Modules::sequential_selection), bool_setter(&Modules::sequential_selection))
        .def_prop_rw("threshold_convergence", bool_getter(&Modules::threshold_convergence), bool_setter(&Modules::threshold_convergence))
        .def_prop_rw("sample_sigma", bool_getter(&Modules::sample_sigma), bool_setter(&Modules::sample_sigma))
        .def_rw("weights", &Modules::weights)
        .def_rw("sampler", &Modules::sampler)
        .def_rw("mirrored", &Modules::mirrored)
        .def_rw("ssa", &Modules::ssa)
        .def_rw("bound_correction", &Modules::bound_correction)
        .def_rw("restart_strategy", &Modules::restart_strategy)
        .def_rw("repelling_type", &Modules::repelling_type)
        .def_rw("repelling_metric", &Modules::repelling_metric)
        .def_rw("matrix_adaptation", &Modules::matrix_adaptation)
        .def_rw("center_placement", &Modules::center_placement)
        .def_rw("sample_transformation", &Modules::sample_transformation)
        .def("__repr__", [](Modules &mod) { return to_string(mod); });

    nb::class_<Solution>(m, "Solution")
        .def(nb::init<>())
        .def_rw("x", &Solution::x)
        .def_rw("y", &Solution::y)
        .def_rw("t", &Solution::t)
        .def_rw("e", &Solution::e)
        .def("__repr__", &Solution::repr);

    nb::class_<Stats>(m, "Stats")
        .def(nb::init<>())
        .def_rw("t", &Stats::t)
        .def_rw("evaluations", &Stats::evaluations)
        .def_rw("current_avg", &Stats::current_avg)
        .def_rw("solutions", &Stats::solutions)
        .def_rw("centers", &Stats::centers)
        .def_rw("current_best", &Stats::current_best)
        .def_rw("global_best", &Stats::global_best)
        .def_rw("has_improved", &Stats::has_improved)
        .def_rw("success_ratio", &Stats::success_ratio)
        .def_rw("last_update", &Stats::last_update)
        .def_rw("n_updates", &Stats::n_updates)
        .def("__repr__", [](Stats &stats) {
            return repr("Stats", [&](std::stringstream &ss) {
                ss << " t: " << stats.t << " e: " << stats.evaluations;
                ss << " best: " << stats.global_best << " improved: " << stats.has_improved;
            });
        });

    nb::class_<Weights>(m, "Weights")
        .def(nb::init<Settings, Float>(), nb::arg("Settings"), nb::arg("expected_length_z"))
        .def_rw("mueff", &Weights::mueff)
        .def_rw("mueff_neg", &Weights::mueff_neg)
        .def_rw("c1", &Weights::c1)
        .def_rw("cmu", &Weights::cmu)
        .def_rw("cc", &Weights::cc)
        .def_rw("cs", &Weights::cs)
        .def_rw("damps", &Weights::damps)
        .def_rw("sqrt_cc_mueff", &Weights::sqrt_cc_mueff)
        .def_rw("sqrt_cs_mueff", &Weights::sqrt_cs_mueff)
        .def_rw("lazy_update_interval", &Weights::lazy_update_interval)
        .def_rw("expected_length_z", &Weights::expected_length_z)
        .def_rw("expected_length_ps", &Weights::expected_length_ps)
        .def_rw("beta", &Weights::beta)
        .def_rw("weights", &Weights::weights)
        .def_rw("positive", &Weights::positive)
        .def_rw("negative", &Weights::negative)
        .def("for_population",
             &Weights::for_population,
             nb::arg("n"),
             "The weights for n evaluated individuals (n < lambda after a sequential selection "
             "stop): the negative weights are scaled to keep their total")
        .def("__repr__", [](Weights &w) {
            return repr("Weights", [&](std::stringstream &ss) {
                ss << " mueff: " << w.mueff << " mueff_neg: " << w.mueff_neg;
                ss << " c1: " << w.c1 << " cmu: " << w.cmu << " cc: " << w.cc;
                ss << " weights: " << w.weights.transpose();
                ss << " positive: " << w.positive.transpose();
                ss << " negative: " << w.negative.transpose();
            });
        });

    // dim and modules are positional; everything else is keyword-only (mu0 before lambda0, as
    // everywhere in modcma). A positional call with more arguments raises a TypeError instead of
    // silently assigning values to the wrong parameters.
    nb::class_<Settings>(m, "Settings")
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
               bool fast_sampling) {
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
            },
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
            nb::arg("fast_sampling") = false)
        .def_ro("dim", &Settings::dim)
        .def_rw("seed",
                &Settings::seed,
                "The seed of the optimizer's own random generator; None: the optimizer starts "
                "from the global generator (utils.set_seed) when it is created")
        .def_rw("fast_sampling",
                &Settings::fast_sampling,
                "Opt-in: sample with xoshiro256++ and a ziggurat instead of mt19937 and the inverse "
                "normal CDF (the default UNIFORM + GAUSSIAN sampler only). Faster, but a different "
                "random stream. Set it before the optimizer is created.")
        .def_ro("modules", &Settings::modules)
        .def_rw("target", &Settings::target)
        .def_rw("max_generations", &Settings::max_generations)
        .def_rw("budget", &Settings::budget)
        .def_rw("sigma0", &Settings::sigma0)
        .def_rw("mu0", &Settings::mu0)
        .def_rw("lambda0", &Settings::lambda0)
        .def_rw("x0", &Settings::x0)
        .def_rw("lb", &Settings::lb)
        .def_rw("ub", &Settings::ub)
        .def_rw("db", &Settings::db)
        .def_rw("center", &Settings::center)
        .def_rw("diameter", &Settings::diameter)
        .def_ro("volume", &Settings::volume)
        .def_prop_rw("has_bounds",
                     bool_getter(&Settings::has_bounds),
                     bool_setter(&Settings::has_bounds))
        .def_ro("integer_variables", &Settings::integer_variables)
        .def_rw("cs", &Settings::cs)
        .def_rw("cc", &Settings::cc)
        .def_rw("cmu", &Settings::cmu)
        .def_rw("c1", &Settings::c1)
        .def_rw("damps", &Settings::damps)
        .def_rw("acov", &Settings::acov)
        .def_prop_rw("verbose", bool_getter(&Settings::verbose), bool_setter(&Settings::verbose))
        .def_ro("one_plus_one", &Settings::one_plus_one)
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
            });
        });

    nb::class_<mapping::CoordinateMapping>(m, "CoordinateMapping")
        .def(
            "__init__",
            [](mapping::CoordinateMapping *self, std::vector<int> idx, size_t d, Float mueff) {
                new (self) mapping::CoordinateMapping(*to_indices(idx), d, mueff);
            },
            nb::arg("integer_variables"),
            nb::arg("d"),
            nb::arg("mueff"))
        .def("transform", &mapping::CoordinateMapping::transform, nb::arg("x"));

    nb::class_<Parameters>(main, "Parameters")
        .def(nb::init<size_t>(), nb::arg("dimension"))
        .def(nb::init<Settings>(), nb::arg("settings"))
        .def("adapt", &Parameters::adapt)
        .def(
            "start",
            [](Parameters &p, FunctionType objective) { p.start(objective); },
            nb::arg("objective"))
        .def(
            "perform_restart",
            [](Parameters &p, FunctionType objective, std::optional<Float> sigma) {
                p.perform_restart(objective, sigma);
            },
            nb::arg("objective"),
            nb::arg("sigma") = nb::none())
        .def("resize_population",
             &Parameters::resize_population,
             nb::arg("new_mu"),
             nb::arg("new_lambda"))
        .def_rw("settings", &Parameters::settings)
        .def_rw("mu", &Parameters::mu)
        .def_rw("lamb", &Parameters::lambda)
        // shared_ptr<Adaptation> comes back as its most-derived bound type
        .def_rw("adaptation", &Parameters::adaptation)
        .def_rw("criteria", &Parameters::criteria)
        .def_rw("stats", &Parameters::stats)
        .def_rw("weights", &Parameters::weights)
        .def_rw("pop", &Parameters::pop)
        .def_rw("old_pop", &Parameters::old_pop)
        .def_rw("sampler", &Parameters::sampler)
        .def_rw("mutation", &Parameters::mutation)
        .def_rw("selection", &Parameters::selection)
        .def_rw("restart_strategy", &Parameters::restart_strategy)
        .def_rw("repelling", &Parameters::repelling)
        .def_rw("bounds", &Parameters::bounds)
        .def_rw("center_placement", &Parameters::center_placement)
        .def_rw("coordinate_mapping", &Parameters::coordinate_mapping);
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
    if (nb::cast<size_t>(a.attr("ndim")) == 1)
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
    while (step(c, objective, true))
        ;
}
} // namespace

void define_cmaes(nb::module_ &m)
{
    nb::class_<ModularCMAES>(m, "ModularCMAES")
        .def(nb::init<std::shared_ptr<parameters::Parameters>>(), nb::arg("parameters"))
        .def(nb::init<size_t>(), nb::arg("dimension"))
        .def(nb::init<parameters::Settings>(), nb::arg("settings"))
        .def("recombine", &ModularCMAES::recombine)
        .def(
            "mutate",
            [](ModularCMAES &c, FunctionType objective) { c.mutate(objective); },
            nb::arg("objective"))
        .def("select", &ModularCMAES::select)
        .def("adapt", &ModularCMAES::adapt)
        .def("step",
             &step,
             nb::sig("def step(self, objective: collections.abc.Callable[[numpy.typing.NDArray["
                     "numpy.float64]], typing.Any], *, vectorized: bool = False) -> bool"),
             nb::arg("objective"),
             nb::kw_only(),
             nb::arg("vectorized") = false,
             "One generation. Returns False when a break condition holds. A vectorized objective "
             "is called with every batch of points as a (k, d) array and returns k values.")
        .def("__call__",
             &run,
             nb::sig("def __call__(self, objective: collections.abc.Callable[[numpy.typing."
                     "NDArray[numpy.float64]], typing.Any], *, vectorized: bool = False) -> None"),
             nb::arg("objective"),
             nb::kw_only(),
             nb::arg("vectorized") = false)
        .def("run",
             &run,
             nb::sig("def run(self, objective: collections.abc.Callable[[numpy.typing.NDArray["
                     "numpy.float64]], typing.Any], *, vectorized: bool = False) -> None"),
             nb::arg("objective"),
             nb::kw_only(),
             nb::arg("vectorized") = false,
             "Run until a break condition holds. A vectorized objective is called with every "
             "batch of points as a (k, d) array and returns k values.")
        .def("ask",
             &ask,
             "The points to evaluate next, a new (k, d) array (row i is point i). Starts a "
             "generation when none is in progress; until tell(), it returns the same points.")
        .def("tell",
             &tell,
             nb::sig("def tell(self, f: numpy.typing.ArrayLike, X: numpy.typing.ArrayLike | None "
                     "= None) -> bool"),
             nb::arg("f"),
             nb::arg("X") = nb::none(),
             "The objective values of the points of ask(), in order. X (k, d): the points that "
             "were evaluated, when they differ from the asked ones (repaired or injected). "
             "Returns True when this completed a generation.")
        .def_prop_ro("has_pending",
                     &ModularCMAES::has_pending,
                     "Whether ask() has points that wait for tell()")
        .def("break_conditions", &ModularCMAES::break_conditions)
        .def_ro("p", &ModularCMAES::p);
}

void define_es(nb::module_ &main)
{
    auto m = main.def_submodule("es");
    using namespace es;
    nb::class_<OnePlusOneES>(m, "OnePlusOneES")
        .def(nb::init<size_t, Vector, Float, Float, size_t, Float, parameters::Modules>(),
             nb::arg("d"),
             nb::arg("x0"),
             nb::arg("f0"),
             nb::arg("sigma0") = 1.0,
             nb::arg("budget") = 10'000,
             nb::arg("target") = 1e-8,
             nb::arg("modules") = parameters::Modules())
        .def("__call__", [](OnePlusOneES &es, FunctionType objective) { es(objective); })
        .def("step", [](OnePlusOneES &es, FunctionType objective) { es.step(objective); })
        .def("sample", &OnePlusOneES::sample)
        .def_rw("d", &OnePlusOneES::d)
        .def_rw("sigma", &OnePlusOneES::sigma)
        .def_rw("decay", &OnePlusOneES::decay)
        .def_rw("x", &OnePlusOneES::x)
        .def_rw("f", &OnePlusOneES::f)
        .def_rw("t", &OnePlusOneES::t)
        .def_rw("budget", &OnePlusOneES::budget)
        .def_rw("target", &OnePlusOneES::target)
        .def_rw("sampler", &OnePlusOneES::sampler)
        .def_rw("rejection_sampling", &OnePlusOneES::rejection_sampling)
        .def_rw("corrector", &OnePlusOneES::corrector);

    nb::class_<MuCommaLambdaES>(m, "MuCommaLambdaES")
        .def(nb::init<size_t, Vector, Float, size_t, Float, parameters::Modules>(),
             nb::arg("d"),
             nb::arg("x0"),
             nb::arg("sigma0") = 1.0,
             nb::arg("budget") = 10'000,
             nb::arg("target") = 1e-8,
             nb::arg("modules") = parameters::Modules())
        .def("__call__", [](MuCommaLambdaES &es, FunctionType objective) { es(objective); })
        .def("step", [](MuCommaLambdaES &es, FunctionType objective) { es.step(objective); })
        .def("sample", &MuCommaLambdaES::sample)
        .def_rw("d", &MuCommaLambdaES::d)
        .def_rw("lamb", &MuCommaLambdaES::lambda)
        .def_rw("mu", &MuCommaLambdaES::mu)
        .def_rw("sigma", &MuCommaLambdaES::sigma)
        .def_rw("m", &MuCommaLambdaES::m)
        .def_rw("X", &MuCommaLambdaES::X)
        .def_rw("S", &MuCommaLambdaES::S)
        .def_rw("f", &MuCommaLambdaES::f)
        .def_rw("tau", &MuCommaLambdaES::tau)
        .def_rw("tau_i", &MuCommaLambdaES::tau_i)
        .def_rw("mu_inv", &MuCommaLambdaES::mu_inv)
        .def_rw("f_min", &MuCommaLambdaES::f_min)
        .def_rw("x_min", &MuCommaLambdaES::x_min)
        .def_rw("t", &MuCommaLambdaES::t)
        .def_rw("e", &MuCommaLambdaES::e)
        .def_rw("budget", &MuCommaLambdaES::budget)
        .def_rw("target", &MuCommaLambdaES::target)
        .def_rw("sampler", &MuCommaLambdaES::sampler)
        .def_rw("sigma_sampler", &MuCommaLambdaES::sigma_sampler)
        .def_rw("rejection_sampling", &MuCommaLambdaES::rejection_sampling)
        .def_rw("corrector", &MuCommaLambdaES::corrector);
}
} // namespace modcma_python
