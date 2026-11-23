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

//! The constructor (dimension, x0, expected_length_z) of most matrix adaptations
template <typename T, typename Class> void adaptation_init(Class &cls)
{
    cls.def(nb::init<size_t, Vector, Float>(),
            nb::arg("dimension"),
            nb::arg("x0"),
            nb::arg("expected_length_z"));
}
} // namespace

void define_selection(nb::module_ &main)
{
    auto m = main.def_submodule("selection");
    using namespace selection;
    nb::class_<Elitsm>(m, "Elitsm")
        .def(nb::init<>())
        .def("__call__", &Elitsm::operator(), nb::arg("parameters"));

    nb::class_<NoElitsm, Elitsm>(m, "NoElitsm")
        .def(nb::init<>())
        .def("__call__", &NoElitsm::operator(), nb::arg("parameters"));

    nb::class_<Pairwise>(m, "Pairwise")
        .def(nb::init<>())
        .def("__call__", &Pairwise::operator(), nb::arg("parameters"));

    nb::class_<NoPairwise, Pairwise>(m, "NoPairwise")
        .def(nb::init<>())
        .def("__call__", &NoPairwise::operator(), nb::arg("parameters"));

    nb::class_<Strategy>(m, "Strategy")
        .def(nb::init<parameters::Modules>(), nb::arg("modules"))
        .def("select", &Strategy::select, nb::arg("parameters"))
        .def_rw("pairwise", &Strategy::pairwise)
        .def_rw("elitsm", &Strategy::elitsm);
}

void define_center_placement(nb::module_ &main)
{
    auto m = main.def_submodule("center");
    using namespace center;
    nb::class_<Placement>(m, "Placement")
        .def("__call__", &Placement::operator(), nb::arg("parameters"));

    nb::class_<X0, Placement>(m, "X0").def(nb::init<>());
    nb::class_<Uniform, Placement>(m, "Uniform").def(nb::init<>());
    nb::class_<Zero, Placement>(m, "Zero").def(nb::init<>());
    nb::class_<Center, Placement>(m, "Center").def(nb::init<>());
    nb::class_<MaximinTaboo, Placement>(m, "MaximinTaboo").def(nb::init<>());
    nb::class_<NoveltyWeighted, Placement>(m, "NoveltyWeighted").def(nb::init<>());
}

void define_repelling(nb::module_ &main)
{
    using namespace repelling;
    auto m = main.def_submodule("repelling");

    nb::class_<TabooPoint>(m, "TabooPoint")
        .def(nb::init<const Solution &, Float, const Matrix &>(),
             nb::arg("solution"),
             nb::arg("radius"),
             nb::arg("metric") = Matrix())
        .def(nb::init<const Solution &, Float, Float, Float, const Matrix &>(),
             nb::arg("solution"),
             nb::arg("radius"),
             nb::arg("min_radius"),
             nb::arg("max_radius"),
             nb::arg("metric") = Matrix())
        .def("distance", &TabooPoint::distance, nb::arg("x"), nb::arg("p"))
        .def_rw("metric", &TabooPoint::metric)
        .def("rejects", &TabooPoint::rejects, nb::arg("xi"), nb::arg("p"), nb::arg("attempts"))
        .def("shares_basin",
             &TabooPoint::shares_basin,
             nb::arg("objective"),
             nb::arg("xi"),
             nb::arg("p"))
        .def("calculate_criticality", &TabooPoint::calculate_criticality, nb::arg("p"))
        .def("clamp_radius", &TabooPoint::clamp_radius)
        .def_rw("solution", &TabooPoint::solution)
        .def_rw("radius", &TabooPoint::radius)
        .def_rw("min_radius", &TabooPoint::min_radius)
        .def_rw("max_radius", &TabooPoint::max_radius)
        .def_rw("shrinkage", &TabooPoint::shrinkage)
        .def_rw("n_rep", &TabooPoint::n_rep)
        .def_rw("criticality", &TabooPoint::criticality)
        .def_rw("duplicate_evaluations", &TabooPoint::duplicate_evaluations)
        .def_rw("rejected_count", &TabooPoint::rejected_count)
        .def_rw("checked_count", &TabooPoint::checked_count)
        .def_rw("last_rejection_rate", &TabooPoint::last_rejection_rate)
        .def("__repr__", [](const TabooPoint &tb) {
            return "<TabooPoint n_rep: " + std::to_string(tb.n_rep) +
                   " y: " + std::to_string(tb.solution.y) +
                   " radius: " + std::to_string(tb.radius) +
                   " criticality: " + std::to_string(tb.criticality) + ">";
        });

    nb::class_<Repelling>(m, "Repelling")
        .def(nb::init<>())
        .def("is_rejected", &Repelling::is_rejected, nb::arg("xi"), nb::arg("p"))
        .def("update_archive", &Repelling::update_archive, nb::arg("objective"), nb::arg("p"))
        .def("prepare_sampling", &Repelling::prepare_sampling, nb::arg("p"))
        .def("adapt_radii_after_sampling", &Repelling::adapt_radii_after_sampling, nb::arg("p"))
        .def_rw("archive", &Repelling::archive)
        .def_rw("attempts", &Repelling::attempts)
        .def_rw("coverage", &Repelling::coverage);

    nb::class_<CoverageRepelling, Repelling>(m, "CoverageRepelling").def(nb::init<>());

    nb::class_<AdaptiveRepelling, Repelling>(m, "AdaptiveRepelling")
        .def(nb::init<>())
        .def("initial_radius", &AdaptiveRepelling::initial_radius, nb::arg("p"))
        .def("min_radius", &AdaptiveRepelling::min_radius, nb::arg("p"))
        .def("max_radius", &AdaptiveRepelling::max_radius, nb::arg("p"))
        .def_rw("initial_radius_factor", &AdaptiveRepelling::initial_radius_factor)
        .def_rw("min_radius_factor", &AdaptiveRepelling::min_radius_factor)
        .def_rw("max_radius_factor", &AdaptiveRepelling::max_radius_factor)
        .def_rw("shrink_factor", &AdaptiveRepelling::shrink_factor)
        .def_rw("target_rejection_rate", &AdaptiveRepelling::target_rejection_rate)
        .def_rw("max_rejection_rate", &AdaptiveRepelling::max_rejection_rate)
        .def_rw("last_restart_evaluations", &AdaptiveRepelling::last_restart_evaluations)
        .def_rw("average_restart_evaluations", &AdaptiveRepelling::average_restart_evaluations)
        .def_rw("restart_eval_alpha", &AdaptiveRepelling::restart_eval_alpha)
        .def_rw("grow_eta", &AdaptiveRepelling::grow_eta)
        .def_rw("max_waste_ratio", &AdaptiveRepelling::max_waste_ratio)
        .def_rw("local_target_rejection_rate", &AdaptiveRepelling::local_target_rejection_rate)
        .def_rw("local_max_rejection_rate", &AdaptiveRepelling::local_max_rejection_rate)
        .def_rw("min_checks_before_local_shrink",
                &AdaptiveRepelling::min_checks_before_local_shrink);

    nb::class_<NoRepelling, Repelling>(m, "NoRepelling").def(nb::init<>());

    m.def("euclidian", &distance::euclidian, nb::arg("u"), nb::arg("v"));
    m.def("manhattan", &distance::manhattan, nb::arg("u"), nb::arg("v"));
    m.def("mahanolobis", &distance::mahanolobis, nb::arg("u"), nb::arg("v"), nb::arg("C_inv"));
    m.def("archived_metric",
          &archived_metric,
          nb::arg("p"),
          "The metric RepellingMetric.ARCHIVED stores with a new taboo point (empty for the "
          "other metrics)");
    m.def("hill_valley_test",
          &distance::hill_valley_test,
          nb::arg("u"),
          nb::arg("v"),
          nb::arg("f"),
          nb::arg("n_evals"));
}

void define_matrix_adaptation(nb::module_ &main)
{
    using namespace matrix_adaptation;
    auto m = main.def_submodule("matrix_adaptation");
    nb::class_<Adaptation>(m, "Adaptation")
        .def_rw("m", &Adaptation::m)
        .def_rw("m_old", &Adaptation::m_old)
        .def_rw("dm", &Adaptation::dm)
        .def_rw("ps", &Adaptation::ps)
        .def_rw("dz", &Adaptation::dz)
        .def_rw("dd", &Adaptation::dd)
        .def_rw("coordinate_wise_variances", &Adaptation::coordinate_wise_variances)
        .def_rw("expected_length_z", &Adaptation::expected_length_z)
        .def("adapt_evolution_paths",
             &Adaptation::adapt_evolution_paths,
             nb::arg("pop"),
             nb::arg("weights"),
             nb::arg("stats"),
             nb::arg("settings"),
             nb::arg("mu"),
             nb::arg("lamb"))
        .def("adapt_evolution_paths_innner",
             &Adaptation::adapt_evolution_paths_inner,
             nb::arg("pop"),
             nb::arg("weights"),
             nb::arg("stats"),
             nb::arg("settings"),
             nb::arg("mu"),
             nb::arg("lamb"))
        .def("adapt_matrix",
             &Adaptation::adapt_matrix,
             nb::arg("weights"),
             nb::arg("modules"),
             nb::arg("population"),
             nb::arg("mu"),
             nb::arg("settings"),
             nb::arg("stats"))
        .def("restart", &Adaptation::restart, nb::arg("settings"), nb::arg("sigma"))
        .def("distance", &Adaptation::distance, nb::arg("u"), nb::arg("v"))
        .def("distance_from_center", &Adaptation::distance_from_center, nb::arg("x"))
        .def("compute_y", &Adaptation::compute_y, nb::arg("zi"))
        .def("invert_x", &Adaptation::invert_x, nb::arg("xi"), nb::arg("sigma"))
        .def("invert_y", &Adaptation::invert_y, nb::arg("yi"))
        .def("__repr__", [](Adaptation &a) {
            return repr("Adaptation", [&](std::stringstream &ss) {
                adaptation_repr(ss, a);
                ss << " dd: " << a.dd << " expected_length_z: " << a.expected_length_z;
            });
        });

    auto none = nb::class_<None, Adaptation>(m, "NoAdaptation");
    adaptation_init<None>(none);
    none.def("__repr__", [](None &a) {
        return repr("NoAdaptation", [&](std::stringstream &ss) {
            adaptation_repr(ss, a);
            ss << " dd: " << a.dd << " expected_length_z: " << a.expected_length_z;
        });
    });

    auto covariance = nb::class_<CovarianceAdaptation, Adaptation>(m, "CovarianceAdaptation");
    adaptation_init<CovarianceAdaptation>(covariance);
    covariance.def_rw("pc", &CovarianceAdaptation::pc)
        .def_rw("d", &CovarianceAdaptation::d)
        .def_rw("B", &CovarianceAdaptation::B)
        .def_rw("C", &CovarianceAdaptation::C)
        .def_rw("A", &CovarianceAdaptation::A)
        .def_prop_ro("inv_root_C",
                     &CovarianceAdaptation::inv_root_C,
                     "C^-1/2 = B D^-1 B^T, computed from B and d (the updates do not form it)")
        .def_rw("hs", &CovarianceAdaptation::hs)
        .def("adapt_covariance_matrix",
             &CovarianceAdaptation::adapt_covariance_matrix,
             nb::arg("weights"),
             nb::arg("modules"),
             nb::arg("population"),
             nb::arg("mu"))
        .def("perform_eigendecomposition",
             &CovarianceAdaptation::perform_eigendecomposition,
             nb::arg("stats"))
        .def("adapt_ps", &CovarianceAdaptation::adapt_ps, nb::arg("weights"))
        .def("__repr__", [](CovarianceAdaptation &a) {
            return repr("CovarianceAdaptation", [&](std::stringstream &ss) {
                adaptation_repr(ss, a);
                ss << " pc: " << a.pc.transpose() << " d: " << a.d.transpose();
                ss << " B: " << a.B << " C: " << a.C << " inv_root_C: " << a.inv_root_C();
                ss << " dd: " << a.dd << " expected_length_z: " << a.expected_length_z;
                ss << " hs: " << a.hs;
            });
        });

    auto separable = nb::class_<SeparableAdaptation, Adaptation>(m, "SeparableAdaptation");
    adaptation_init<SeparableAdaptation>(separable);
    separable.def_rw("c", &SeparableAdaptation::c)
        .def_rw("pc", &SeparableAdaptation::pc)
        .def_rw("d", &SeparableAdaptation::d)
        .def("__repr__", [](SeparableAdaptation &a) {
            return repr("SeparableAdaptation", [&](std::stringstream &ss) {
                adaptation_repr(ss, a);
                ss << " pc: " << a.pc.transpose() << " d: " << a.d.transpose();
                ss << " c: " << a.c.transpose() << " expected_length_z: " << a.expected_length_z;
                ss << " hs: " << a.hs;
            });
        });

    auto matrix = nb::class_<MatrixAdaptation, Adaptation>(m, "MatrixAdaptation");
    adaptation_init<MatrixAdaptation>(matrix);
    matrix.def_rw("M", &MatrixAdaptation::M)
        .def_rw("M_inv", &MatrixAdaptation::M_inv)
        .def("__repr__", [](MatrixAdaptation &a) {
            return repr("MatrixAdaptation", [&](std::stringstream &ss) {
                adaptation_repr(ss, a);
                ss << " M: " << a.M << " dd: " << a.dd;
                ss << " expected_length_z: " << a.expected_length_z;
            });
        });

    auto cholesky = nb::class_<CholeskyAdaptation, Adaptation>(m, "CholeskyAdaptation");
    adaptation_init<CholeskyAdaptation>(cholesky);
    cholesky.def_rw("A", &CholeskyAdaptation::A).def_rw("pc", &CholeskyAdaptation::pc);

    auto self = nb::class_<SelfAdaptation, Adaptation>(m, "SelfAdaptation");
    adaptation_init<SelfAdaptation>(self);
    self.def_rw("A", &SelfAdaptation::A).def_rw("C", &SelfAdaptation::C);

    nb::class_<CovarianceNoEigvAdaptation, CovarianceAdaptation>(m, "CovarianceNoEigvAdaptation");

    nb::class_<NaturalGradientAdaptation, Adaptation>(m, "NaturalGradientAdaptation")
        .def(nb::init<size_t, Vector, Float, Float>(),
             nb::arg("dimension"),
             nb::arg("x0"),
             nb::arg("expected_length_z"),
             nb::arg("sigma"))
        .def_rw("A", &NaturalGradientAdaptation::A)
        .def_rw("A_inv", &NaturalGradientAdaptation::A_inv)
        .def_rw("G", &NaturalGradientAdaptation::G)
        .def_rw("sigma_g", &NaturalGradientAdaptation::sigma_g)
        .def("compute_gradients",
             &NaturalGradientAdaptation::compute_gradients,
             nb::arg("pop"),
             nb::arg("weights"),
             nb::arg("stats"),
             nb::arg("settings"),
             nb::arg("mu"),
             nb::arg("lamb"));
}

void define_bounds(nb::module_ &main)
{
    auto m = main.def_submodule("bounds");
    using namespace bounds;

    nb::class_<BoundCorrection>(m, "BoundCorrection")
        .def_ro("n_out_of_bounds", &BoundCorrection::n_out_of_bounds)
        .def("correct", &BoundCorrection::correct, nb::arg("index"), nb::arg("parameters"))
        .def("correct_x",
             &BoundCorrection::correct_x,
             nb::arg("xi"),
             nb::arg("oob"),
             nb::arg("sigma"),
             nb::arg("settings"))
        .def("delta_out_of_bounds",
             &BoundCorrection::delta_out_of_bounds,
             nb::arg("xi"),
             nb::arg("oob"),
             nb::arg("settings"))
        .def("is_out_of_bounds",
             &BoundCorrection::is_out_of_bounds,
             nb::arg("xi"),
             nb::arg("settings"));

    nb::class_<Resample, BoundCorrection>(m, "Resample").def(nb::init<>());
    nb::class_<NoCorrection, BoundCorrection>(m, "NoCorrection").def(nb::init<>());
    nb::class_<COTN, BoundCorrection>(m, "COTN")
        .def(nb::init<size_t>(), nb::arg("dim"))
        .def_ro("sampler", &COTN::sampler);
    nb::class_<Mirror, BoundCorrection>(m, "Mirror").def(nb::init<>());
    nb::class_<UniformResample, BoundCorrection>(m, "UniformResample")
        .def(nb::init<size_t>(), nb::arg("dim"));
    nb::class_<Saturate, BoundCorrection>(m, "Saturate").def(nb::init<>());
    nb::class_<Toroidal, BoundCorrection>(m, "Toroidal").def(nb::init<>());
}

void define_mutation(nb::module_ &main)
{
    auto m = main.def_submodule("mutation");
    using namespace mutation;

    nb::class_<ThresholdConvergence>(m, "ThresholdConvergence")
        .def(nb::init<>())
        .def_rw("init_threshold", &ThresholdConvergence::init_threshold)
        .def_rw("decay_factor", &ThresholdConvergence::decay_factor)
        .def("scale",
             &ThresholdConvergence::scale,
             nb::arg("population"),
             nb::arg("diameter"),
             nb::arg("budget"),
             nb::arg("evaluations"));

    nb::class_<NoThresholdConvergence, ThresholdConvergence>(m, "NoThresholdConvergence")
        .def(nb::init<>());

    nb::class_<SequentialSelection>(m, "SequentialSelection")
        .def(nb::init<parameters::Mirror, Float>(),
             nb::arg("mirror"),
             nb::arg("seq_cuttoff_factor") = 1.0)
        .def("break_conditions",
             &SequentialSelection::break_conditions,
             nb::arg("i"),
             nb::arg("f"),
             nb::arg("fopt"),
             nb::arg("mirror"),
             nb::arg("mu"));

    nb::class_<NoSequentialSelection, SequentialSelection>(m, "NoSequentialSelection")
        .def(nb::init<parameters::Mirror, Float>(),
             nb::arg("mirror"),
             nb::arg("seq_cuttoff_factor") = 1.0);

    nb::class_<SigmaSampler>(m, "SigmaSampler")
        .def(nb::init<Float>(), nb::arg("dimension"))
        .def("apply_integer_bounds", &SigmaSampler::apply_integer_bounds, nb::arg("parameters"))
        .def("sample", &SigmaSampler::sample, nb::arg("sigma"), nb::arg("parameters"));

    nb::class_<NoSigmaSampler, SigmaSampler>(m, "NoSigmaSampler")
        .def(nb::init<Float>(), nb::arg("dimension"));

    nb::class_<Strategy>(m, "Strategy")
        .def(nb::init<std::shared_ptr<ThresholdConvergence>,
                      std::shared_ptr<SequentialSelection>,
                      std::shared_ptr<SigmaSampler>,
                      Float>(),
             nb::arg("threshold_convergence"),
             nb::arg("sequential_selection"),
             nb::arg("sigma_sampler"),
             nb::arg("sigma0"))
        .def("adapt",
             &Strategy::adapt,
             nb::arg("weights"),
             nb::arg("dynamic"),
             nb::arg("population"),
             nb::arg("old_population"),
             nb::arg("stats"),
             nb::arg("lamb"))
        .def("mutate",
             &Strategy::mutate,
             nb::arg("objective"),
             nb::arg("n_offspring"),
             nb::arg("parameters"))
        .def_rw("threshold_convergence", &Strategy::tc)
        .def_rw("sequential_selection", &Strategy::sq)
        .def_rw("sigma_sampler", &Strategy::ss)
        .def_rw("sigma", &Strategy::sigma)
        .def_rw("s", &Strategy::s);

    nb::class_<CSA, Strategy>(m, "CSA");
    nb::class_<TPA, Strategy>(m, "TPA")
        .def_rw("a_tpa", &TPA::a_tpa)
        .def_rw("b_tpa", &TPA::b_tpa)
        .def_rw("rank_tpa", &TPA::rank_tpa);
    nb::class_<MSR, Strategy>(m, "MSR");
    nb::class_<PSR, Strategy>(m, "PSR").def_rw("success_ratio", &PSR::success_ratio);
    nb::class_<XNES, Strategy>(m, "XNES");
    nb::class_<MXNES, Strategy>(m, "MXNES");
    nb::class_<LPXNES, Strategy>(m, "LPXNES");
    nb::class_<SR, Strategy>(m, "SR");
    nb::class_<SA, Strategy>(m, "SA");
}
} // namespace modcma_python
