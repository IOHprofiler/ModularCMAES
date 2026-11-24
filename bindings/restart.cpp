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
    using namespace restart;

    nb::class_<Criterion, PyCriterion>(m, "Criterion")
        .def(nb::init<std::string>(), nb::arg("name"))
        .def("on_reset", &Criterion::on_reset, nb::arg("parameters"))
        .def("update", &Criterion::update, nb::arg("parameters"))
        .def("reset", &Criterion::reset, nb::arg("parameters"))
        .def_rw("met", &Criterion::met)
        .def_rw("name", &Criterion::name)
        .def_rw("last_restart", &Criterion::last_restart)
        .def("__repr__", [](Criterion &self) {
            return "<" + self.name + " met: " + std::to_string(self.met) + ">";
        });

    nb::class_<ExceededMaxIter, Criterion>(m, "ExceededMaxIter")
        .def(nb::init<>())
        .def_rw("max_iter", &ExceededMaxIter::max_iter);

    nb::class_<NoImprovement, Criterion>(m, "NoImprovement")
        .def(nb::init<>())
        .def_rw("n_bin", &NoImprovement::n_bin)
        .def_rw("best_fitnesses", &NoImprovement::best_fitnesses);

    nb::class_<MaxSigma, Criterion>(m, "MaxSigma")
        .def(nb::init<>())
        .def_rw_static("tolerance", &MaxSigma::tolerance);

    nb::class_<MinSigma, Criterion>(m, "MinSigma")
        .def(nb::init<>())
        .def_rw_static("tolerance", &MinSigma::tolerance);

    nb::class_<UnableToAdapt, Criterion>(m, "UnableToAdapt").def(nb::init<>());

    nb::class_<FlatFitness, Criterion>(m, "FlatFitness")
        .def(nb::init<>())
        .def_rw("max_flat_fitness", &FlatFitness::max_flat_fitness)
        .def_rw("flat_fitness_index", &FlatFitness::flat_fitness_index)
        .def_rw("flat_fitnesses", &FlatFitness::flat_fitnesses);

    nb::class_<TolX, Criterion>(m, "TolX")
        .def(nb::init<>())
        .def_rw("tolx_vector", &TolX::tolx_vector)
        .def_rw_static("tolerance", &TolX::tolerance);

    nb::class_<MaxDSigma, Criterion>(m, "MaxDSigma")
        .def(nb::init<>())
        .def_rw_static("tolerance", &MaxDSigma::tolerance);

    nb::class_<MinDSigma, Criterion>(m, "MinDSigma")
        .def(nb::init<>())
        .def_rw_static("tolerance", &MinDSigma::tolerance);

    nb::class_<ConditionC, Criterion>(m, "ConditionC")
        .def(nb::init<>())
        .def_rw_static("tolerance", &ConditionC::tolerance);

    nb::class_<NoEffectAxis, Criterion>(m, "NoEffectAxis")
        .def(nb::init<>())
        .def_rw_static("tolerance", &NoEffectAxis::tolerance);

    nb::class_<NoEffectCoord, Criterion>(m, "NoEffectCoord")
        .def(nb::init<>())
        .def_rw_static("tolerance", &NoEffectCoord::tolerance);

    nb::class_<Stagnation, Criterion>(m, "Stagnation")
        .def(nb::init<>())
        .def_rw("n_stagnation", &Stagnation::n_stagnation)
        .def_rw("median_fitnesses", &Stagnation::median_fitnesses)
        .def_rw("best_fitnesses", &Stagnation::best_fitnesses)
        .def_rw_static("tolerance", &Stagnation::tolerance);

    nb::class_<TooMuchRepelling, Criterion>(m, "TooMuchRepelling")
        .def(nb::init<>())
        .def_rw_static("tolerance", &TooMuchRepelling::tolerance);

    nb::bind_vector<vCriteria>(m, "CriteriaVector").def("__repr__", [](const vCriteria &v) {
        std::string s = "CriteriaVector[";
        for (size_t i = 0; i < v.size(); ++i)
        {
            if (i)
                s += ", ";
            s += v[i] ? nb::cast<std::string>(nb::repr(nb::cast(v[i]))) : "None";
        }
        return s + "]";
    });

    nb::class_<Criteria>(m, "Criteria")
        .def_prop_rw(
            "items",
            [](Criteria &self) -> vCriteria & { return self.items; },
            [](Criteria &self, vCriteria v) { self.items = std::move(v); },
            nb::rv_policy::reference_internal)
        .def("reset", &Criteria::reset, nb::arg("parameters"))
        .def("update", &Criteria::update, nb::arg("parameters"))
        .def("reason", &Criteria::reason)
        .def("any", &Criteria::any);
}

void define_restart_strategy(nb::module_ &main)
{
    auto m = submodule(main, "restart");
    using namespace restart;

    nb::class_<Strategy>(m, "Strategy").def("update", &Strategy::update, nb::arg("parameters"));

    nb::class_<IPOP, Strategy>(m, "IPOP").def_rw("ipop_factor", &IPOP::ipop_factor);

    nb::class_<BIPOP, Strategy>(m, "BIPOP")
        .def("large", &BIPOP::large)
        .def_rw("mu_factor", &BIPOP::mu_factor)
        .def_rw("lambda_init", &BIPOP::lambda_init)
        .def_rw("budget", &BIPOP::budget)
        .def_rw("lambda_large", &BIPOP::lambda_large)
        .def_rw("lambda_small", &BIPOP::lambda_small)
        .def_rw("budget_small", &BIPOP::budget_small)
        .def_rw("budget_large", &BIPOP::budget_large)
        .def_ro("used_budget", &BIPOP::used_budget);
}
} // namespace modcma_python
