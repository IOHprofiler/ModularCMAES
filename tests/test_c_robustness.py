"""Invalid input, budgets, stale views and crashes found by the third and fourth reviews
(#113, #114, #115, #117, #118, #119). None of these fixes changes the golden trajectories."""

import unittest

import numpy as np

from modcma import c_maes
from modcma.c_maes import ModularCMAES, Settings, fmin, restart, sampling, settings_from_dict


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


class RestartEvery(restart.Criterion):
    """Met every `period` generations"""

    def __init__(self, period):
        super().__init__("RestartEvery")
        self.period = period

    def update(self, parameters):
        self.met = parameters.stats.t % self.period == self.period - 1


class TestSettingsValidation(unittest.TestCase):
    """#113: invalid values raise ValueError, naming the argument, instead of crashing later"""

    def assert_rejected(self, word, **kwargs):
        dim = kwargs.pop("dim", 3)
        with self.assertRaisesRegex(ValueError, word):
            Settings(dim, **kwargs)

    def test_dimension_and_population(self):
        self.assert_rejected("dim", dim=0)
        self.assert_rejected("lambda0", lambda0=0)
        self.assert_rejected("mu0", mu0=0)
        for lambda0 in (1, 2, 3, 4):  # 1: the (1+1)-ES
            Settings(3, lambda0=lambda0)

    def test_mu0_above_lambda0_is_rejected(self):
        # The maintainer's decision on #136: rejected, not replaced by lambda0 // 2
        sa = c_maes.Modules()
        sa.ssa = c_maes.options.StepSizeAdaptation.SA
        for modules in (None, sa):
            with self.assertRaisesRegex(ValueError, "mu0 = 20 exceeds lambda0 = 10"):
                Settings(3, modules, mu0=20, lambda0=10)
            self.assertEqual(Settings(3, modules, mu0=10, lambda0=10).mu0, 10)
        # The (1+1)-ES still sets mu0 = 1: a module adjustment, not this rule
        settings = Settings(3, mu0=5, lambda0=1)
        self.assertEqual((settings.mu0, settings.one_plus_one), (1, True))

    def test_target(self):
        # The maintainer's decision on #136: NaN would never stop a run; None is no target
        self.assert_rejected("target must not be NaN", target=float("nan"))
        settings = Settings(3, target=None)
        self.assertIsNone(settings.target)
        settings.target = 1e-8
        with self.assertRaisesRegex(ValueError, "target must not be NaN"):
            settings.target = float("nan")
        self.assertEqual(settings.target, 1e-8)
        settings.target = None
        self.assertIsNone(settings.target)
        for target in (float("-inf"), float("inf")):
            settings.target = target
            ModularCMAES(settings)

    def test_step_size_and_vectors(self):
        for sigma0 in (0.0, -1.0, float("nan"), float("inf")):
            self.assert_rejected("sigma0", sigma0=sigma0)
        self.assert_rejected("x0", x0=[1.0])
        self.assert_rejected("x0", x0=[0.0, float("nan"), 0.0])
        self.assert_rejected("lb", lb=[0.0])
        self.assert_rejected("ub", ub=[1.0, 2.0])
        self.assert_rejected("lb must not exceed ub", lb=[0.0, 2.0, 0.0], ub=[1.0, 1.0, 1.0])
        Settings(3, lb=[0.0, -np.inf, 1.0], ub=[1.0, np.inf, 1.0])  # infinite and zero widths

    def test_budget_of_the_first_generation(self):
        # The first generation and the evaluation of the final center of mass (#111)
        self.assert_rejected("budget", lambda0=10, budget=10)
        Settings(3, lambda0=10, budget=11)
        tpa = c_maes.Modules()
        tpa.ssa = c_maes.options.StepSizeAdaptation.TPA
        self.assert_rejected("budget", modules=tpa, lambda0=10, budget=12)
        Settings(3, tpa, lambda0=10, budget=13)
        with self.assertRaisesRegex(ValueError, "budget"):
            fmin(sphere, np.ones(10), 0.5, budget=5)

    def test_setters_check_the_length(self):
        settings = Settings(3)
        for name in ("x0", "lb", "ub"):
            with self.assertRaisesRegex(ValueError, name):
                setattr(settings, name, np.zeros(2))
        settings.x0 = None
        settings.lb = np.zeros(3)

    def test_lambda0_one_turns_pairwise_off(self):
        # F17 (#75), the maintainer's decision on #136: PAIRWISE made lambda0 = 1 into 2, with
        # mu0 = 0 and NaN weights; the (1+1)-ES now runs without it, as with NONE
        runs = []
        for mirrored in ("PAIRWISE", "NONE"):
            modules = c_maes.Modules()
            modules.mirrored = getattr(c_maes.options.Mirror, mirrored)
            settings = Settings(3, modules, lambda0=1, budget=200, seed=1)
            self.assertEqual((settings.lambda0, settings.mu0, settings.one_plus_one), (1, 1, True))
            self.assertEqual(settings.modules.mirrored, c_maes.options.Mirror.NONE)
            es = ModularCMAES(settings)
            es.run(sphere)
            self.assertTrue(np.isfinite(es.p.weights.mueff))
            runs.append((es.p.stats.global_best.y, es.p.stats.evaluations))
        self.assertEqual(runs[0], runs[1])
        # MIRRORED stays (#75)
        modules = c_maes.Modules()
        modules.mirrored = c_maes.options.Mirror.MIRRORED
        settings = Settings(3, modules, lambda0=1)
        self.assertEqual(settings.modules.mirrored, c_maes.options.Mirror.MIRRORED)

    def test_internal_shapes(self):
        internal = c_maes.cmaescpp._internal
        for helper in (internal.lapack_eigh, internal.lapack_cholesky, internal.lapack_inverse):
            with self.assertRaisesRegex(ValueError, "square"):
                helper(np.eye(4)[:, :3])
        with self.assertRaisesRegex(ValueError, "inner dimensions"):
            internal.blas_gemm(np.ones((2, 3)), np.ones((2, 3)))
        population = c_maes.Population(2, 4)
        with self.assertRaises(IndexError):
            population.keep_only([100])
        with self.assertRaisesRegex(ValueError, "capacity"):
            population.n = 100
        es = ModularCMAES(Settings(3, seed=1))
        es.ask()
        with self.assertRaisesRegex(ValueError, "points"):
            es.tell(np.zeros(es.p.lamb), np.zeros((es.p.lamb, 3, 1)))


class TestSettingsSetters(unittest.TestCase):
    """#136: a setter checks its own field as the constructor does; the checks between fields run
    in the constructor and again when the optimizer is created, so the fields can be set in any
    order; the derived fields follow the setters. The bounds in either order: test_c_bounds,
    TestDegenerateBox (#133)"""

    #: Per field: values that the constructor and the setter reject (the field's own checks)
    INVALID = {
        "lambda0": [0],
        "mu0": [0],
        "target": [float("nan")],
        "sigma0": [0.0, -1.0, float("nan"), float("inf")],
        "budget": [0, 1],
        "x0": [np.zeros(2), np.array([0.0, np.nan, 0.0]), np.array([np.inf, 0.0, 0.0])],
        "lb": [np.zeros(2), np.array([0.0, np.nan, 0.0]), np.array([np.inf, 0.0, 0.0])],
        "ub": [np.zeros(4), np.array([np.nan, 0.0, 0.0]), np.array([0.0, -np.inf, 0.0])],
    }

    def assert_creation_rejects(self, settings, pattern):
        """Both ways to create an optimizer from the settings raise ValueError"""
        with self.assertRaisesRegex(ValueError, pattern):
            ModularCMAES(settings)
        with self.assertRaisesRegex(ValueError, pattern):
            c_maes.Parameters(settings)

    def test_the_setter_rejects_what_the_constructor_rejects(self):
        for name, values in self.INVALID.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        Settings(3, **{name: value})
                    settings = Settings(3)
                    before = getattr(settings, name)
                    with self.assertRaisesRegex(ValueError, name):
                        setattr(settings, name, value)
                    np.testing.assert_array_equal(getattr(settings, name), before)
                    ModularCMAES(settings)

    def test_the_setters_take_array_like(self):
        settings = Settings(2)
        settings.lb, settings.ub, settings.x0 = [0, 1], (2.0, 3.0), [1, 2]
        np.testing.assert_array_equal(settings.lb, [0.0, 1.0])
        np.testing.assert_array_equal(settings.ub, [2.0, 3.0])
        np.testing.assert_array_equal(settings.x0, [1.0, 2.0])

    def test_population_in_any_order(self):
        for order in (("mu0", "lambda0"), ("lambda0", "mu0")):
            settings = Settings(3)
            values = {"mu0": 20, "lambda0": 40}
            for name in order:  # mu0 first: mu0 = 20 > lambda0 = 7 in between
                setattr(settings, name, values[name])
            self.assertEqual((settings.mu0, settings.lambda0), (20, 40))
            ModularCMAES(settings)
        settings = Settings(3)
        settings.mu0 = 20
        self.assert_creation_rejects(settings, "mu0 = 20 exceeds lambda0 = 7")

    def test_budget_in_any_order(self):
        settings = Settings(3, lambda0=10, budget=100)
        settings.budget = 5  # smaller than the first generation until lambda0 is set
        self.assert_creation_rejects(settings, "budget 5 .* lambda0 = 10 and 1, 11 evaluations")
        settings.lambda0 = 4
        ModularCMAES(settings)
        tpa = c_maes.Modules()
        tpa.ssa = c_maes.options.StepSizeAdaptation.TPA
        settings = Settings(3, tpa, lambda0=10)
        settings.budget = 12
        self.assert_creation_rejects(settings, "budget 12 .* TPA's 2 points and 1, 13 evaluations")
        settings.budget = 13
        ModularCMAES(settings)

    def test_lambda0_and_the_adjustments_of_the_constructor(self):
        # The setter does not adjust: what the constructor would have changed is rejected
        pairwise = c_maes.Modules()
        pairwise.mirrored = c_maes.options.Mirror.PAIRWISE
        settings = Settings(3, pairwise)
        settings.lambda0 = 7
        self.assert_creation_rejects(settings, "lambda0 = 7 is odd, but mirrored PAIRWISE")
        settings.lambda0 = 8
        ModularCMAES(settings)
        settings = Settings(3)
        settings.lambda0 = 1
        self.assert_creation_rejects(settings, "give lambda0 = 1 to the constructor")
        settings = Settings(3, lambda0=1)
        settings.lambda0 = 10
        self.assert_creation_rejects(settings, "lambda0 = 10, but these settings were created")
        settings.lambda0 = 1
        ModularCMAES(settings)

    def test_mu0_follows_lambda0_until_it_is_given(self):
        sa = c_maes.Modules()
        sa.ssa = c_maes.options.StepSizeAdaptation.SA
        for modules in (None, sa):
            for lambda0 in (2, 5, 12, 100):
                with self.subTest(modules=modules, lambda0=lambda0):
                    settings = Settings(3, modules)
                    settings.lambda0 = lambda0
                    expected = Settings(3, modules, lambda0=lambda0)
                    self.assertEqual(settings.mu0, expected.mu0)
        settings = Settings(3, mu0=2)
        settings.lambda0 = 30
        self.assertEqual(settings.mu0, 2)
        settings = Settings(3)
        settings.mu0 = 3
        settings.lambda0 = 30
        self.assertEqual(settings.mu0, 3)

    def test_the_derived_values_of_the_box(self):
        lb, ub = np.array([-1.0, 0.0, 2.0]), np.array([1.0, 4.0, np.inf])
        settings = Settings(3)
        db = settings.db  # a read-only view: it follows lb and ub
        settings.lb, settings.ub = lb, ub
        expected = Settings(3, lb=lb, ub=ub)
        for name in ("db", "center", "diameter", "volume", "has_bounds"):
            np.testing.assert_array_equal(getattr(settings, name), getattr(expected, name))
        np.testing.assert_array_equal(db, ub - lb)
        for name in ("db", "center", "diameter", "volume", "has_bounds"):
            # The error names the field (the getter is named, the maintainer's decision on #136)
            with self.assertRaisesRegex(AttributeError, f"property '{name}' .* has no setter"):
                setattr(settings, name, getattr(settings, name))
        with self.assertRaisesRegex(ValueError, "read-only"):
            settings.center[0] = 1.0

    def test_a_run_after_the_setters_is_the_constructed_run(self):
        values = dict(
            x0=np.array([1.0, -2.0, 0.5]),
            lb=np.array([-3.0, -4.0, -2.0]),
            ub=np.array([3.0, 2.0, 4.0]),
            sigma0=0.7,
            budget=600,
            lambda0=12,
            target=1e-6,
        )
        sa = c_maes.Modules()
        sa.ssa = c_maes.options.StepSizeAdaptation.SA
        sa.bound_correction = c_maes.options.CorrectionMethod.SATURATE
        for modules in (None, sa):
            for mu0 in (None, 3):
                with self.subTest(modules=modules, mu0=mu0):
                    given = dict(values, **({} if mu0 is None else {"mu0": mu0}))
                    constructed = ModularCMAES(Settings(3, modules, seed=4, **given))
                    settings = Settings(3, modules, seed=4)
                    for name, value in reversed(list(given.items())):
                        setattr(settings, name, value)
                    after_setters = ModularCMAES(settings)
                    for es in (constructed, after_setters):
                        es.run(lambda x: sphere(np.asarray(x) - 1))
                    a, b = constructed.p, after_setters.p
                    self.assertEqual((a.mu, a.lamb), (b.mu, b.lamb))
                    self.assertEqual(a.stats.evaluations, b.stats.evaluations)
                    self.assertEqual(a.stats.global_best.y, b.stats.global_best.y)
                    np.testing.assert_array_equal(a.adaptation.m, b.adaptation.m)


class TestBudget(unittest.TestCase):
    """#114: run() checks the break conditions before every generation"""

    def test_run_never_exceeds_the_budget(self):
        # Every generation leaves one evaluation for the final center of mass (#111)
        for budget in (9, 16, 25, 33):
            es = ModularCMAES(Settings(4, lambda0=8, budget=budget, seed=1))
            es.run(sphere)
            self.assertLessEqual(es.p.stats.evaluations, budget)
            self.assertEqual(es.p.stats.evaluations, 8 * ((budget - 1) // 8) + 1)

    def test_a_finished_optimizer_does_not_run_again(self):
        for vectorized in (False, True):
            es = ModularCMAES(Settings(4, lambda0=8, budget=40, seed=1))
            objective = (lambda X: np.sum(X**2, axis=1)) if vectorized else sphere
            es.run(objective, vectorized=vectorized)
            used = es.p.stats.evaluations
            es.run(objective, vectorized=vectorized)
            self.assertEqual(es.p.stats.evaluations, used)

    def test_a_restart_past_the_budget_keeps_the_one_plus_one(self):
        # ask-tell that ignores break_conditions: the first generation after a restart was capped
        # to at least 2 points, which turned lambda = 1 into 2
        es = ModularCMAES(Settings(3, lambda0=1, budget=30, seed=1))
        es.p.criteria.items.append(RestartEvery(5))
        while es.p.stats.evaluations < 80:
            X = es.ask()
            es.tell([sphere(x) for x in X])
        self.assertGreater(len(es.p.stats.centers), 1)
        self.assertEqual((es.p.lamb, es.p.mu), (1, 1))


class TestViews(unittest.TestCase):
    """#115: the best solutions are copies"""

    def test_fmin_returns_a_copy(self):
        xopt, _, _, es = fmin(lambda x: sphere(np.asarray(x) - 1), np.zeros(3), 0.5, 60)
        before = xopt.copy()
        es.p.settings.budget = 2000
        es.run(lambda x: sphere(np.asarray(x) - 1))
        np.testing.assert_array_equal(xopt, before)
        self.assertLess(es.p.stats.global_best.y, sphere(before - 1))

    def test_a_stored_best_keeps_its_value(self):
        es = ModularCMAES(Settings(3, seed=1))
        history = []
        for _ in range(20):
            es.step(sphere)
            history.append(es.p.stats.current_best.x)
        self.assertGreater(len({tuple(x) for x in history}), 1)


class TestSmallPopulations(unittest.TestCase):
    """lambda0 = 2 and 3 are allowed for experiments: many modules work poorly with them, but no
    module option may crash or produce NaN"""

    def test_every_module_option(self):
        options = c_maes.get_all_module_options()
        variants = [(name, value) for name, values in options.items() for value in values]
        for lambda0 in (2, 3):
            for name, value in variants:
                with self.subTest(lambda0=lambda0, option=name, value=value):
                    es = ModularCMAES(settings_from_dict(
                        4, lambda0=lambda0, seed=1, budget=600, **{name: value}))
                    es.run(lambda X: np.sum(X**2, axis=1), vectorized=True)
                    self.assertTrue(np.isfinite(es.p.stats.global_best.y))
                    self.assertTrue(np.all(np.isfinite(es.p.adaptation.m)))


class TestActiveZeroStep(unittest.TestCase):
    """#118: a zero step with a negative weight made the active update NaN"""

    def test_elitist_active_with_one_parent(self):
        # before, over 20 seeds: C became NaN in every run, followed by 70-110 restarts in 300
        # generations. How far a run converges is not checked: at lambda0 = 2 it depends on the
        # seed and differs between platforms.
        for lambda0 in (2, 3, 4, 10):
            with self.subTest(lambda0=lambda0):
                es = ModularCMAES(settings_from_dict(
                    5, lambda0=lambda0, mu0=1, elitist=True, active=True, seed=1, budget=10**6,
                    restart_strategy="RESTART"))
                for _ in range(300):
                    es.step(sphere)
                    self.assertTrue(np.all(np.isfinite(es.p.adaptation.C)))
                self.assertLessEqual(len(es.p.stats.centers), 2)

    def test_injected_mean(self):
        for matrix_adaptation in ("COVARIANCE", "COVARIANCE_NO_EIGV"):
            es = ModularCMAES(settings_from_dict(
                4, active=True, seed=1, matrix_adaptation=matrix_adaptation,
                always_compute_eigv=matrix_adaptation == "COVARIANCE"))
            for _ in range(3):
                es.step(sphere)
            X = es.ask().copy()
            X[-1] = es.center_of_mass  # the mean, with the worst value: a negative weight, y = 0
            f = [sphere(x) for x in X]
            f[-1] = 1e9
            es.tell(f, X)
            self.assertTrue(np.all(np.isfinite(es.p.adaptation.C)))


class TestCriteriaAddedLater(unittest.TestCase):
    """#119: a built-in criterion added from Python crashed (buffers sized only at a reset)"""

    def test_built_in_criteria(self):
        for name in ("TolX", "FlatFitness", "NoImprovement", "ExceededMaxIter", "Stagnation"):
            with self.subTest(criterion=name):
                es = ModularCMAES(Settings(4, seed=1))
                criterion = getattr(restart, name)()
                es.p.criteria.items.append(criterion)
                for _ in range(30):
                    es.step(sphere)
                self.assertIsInstance(criterion.met, bool)
        criterion = restart.ExceededMaxIter()
        es = ModularCMAES(Settings(4, seed=1))
        es.p.criteria.items.append(criterion)
        es.step(sphere)
        self.assertLess(criterion.max_iter, 10**6)  # set, not uninitialised


class TestSmallFixes(unittest.TestCase):
    """#117"""

    def test_cached_sampler_resets_the_wrapped_sampler(self):
        modules = c_maes.Modules()
        cached = sampling.CachedSampler(sampling.Mirrored(sampling.Gaussian(3)))
        first = cached()
        cached.reset(modules, 4)
        # a new pair after the reset, not the mirror of the previous sample
        self.assertFalse(np.allclose(cached(), -first))

    def test_inv_root_c_without_an_eigendecomposition(self):
        es = ModularCMAES(settings_from_dict(4, ssa="MSR", seed=1))  # COVARIANCE_NO_EIGV
        rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)))
        weights = 10.0 ** np.arange(4)
        for _ in range(40):
            es.step(lambda x: float(np.sum(weights * (rotation @ x) ** 2)))
        a = es.p.adaptation
        self.assertIsInstance(a, c_maes.cmaescpp.matrix_adaptation.CovarianceNoEigvAdaptation)
        # inv_root_C moved to the metric (#123); at d = 4 every update is factorized: C is sampled
        inv_root_c = a.metric().inv_root_C()
        np.testing.assert_allclose(inv_root_c @ a.C @ inv_root_c, np.eye(4), atol=1e-8)

    def test_resize_population_keeps_the_previous_generation(self):
        es = ModularCMAES(Settings(4, lambda0=8, seed=1))
        es.step(sphere)
        es.p.resize_population(es.p.mu, es.p.lamb)
        es.ask()  # the next generation: the previous one is old_pop
        self.assertEqual(es.p.old_pop.n_finite, 8)

    def test_mu_comma_lambda_es_budget(self):
        es = c_maes.es.MuCommaLambdaES(4, np.zeros(4), budget=25)
        es(sphere)
        self.assertLessEqual(es.e, 25)
        values = iter([float("nan")] * 5 + [1.0] * 1000)
        es = c_maes.es.MuCommaLambdaES(4, np.zeros(4), budget=100)
        es(lambda x: next(values))  # NaN values sort last
        self.assertTrue(np.isfinite(es.f_min))


if __name__ == "__main__":
    unittest.main()
