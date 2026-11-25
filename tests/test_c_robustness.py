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
        for lambda0 in (0, 2, 3):
            self.assert_rejected("lambda0", lambda0=lambda0)
        self.assert_rejected("mu0", mu0=0)
        Settings(3, lambda0=1)  # the (1+1)-ES
        Settings(3, lambda0=4)

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
        self.assert_rejected("budget", lambda0=10, budget=9)
        Settings(3, lambda0=10, budget=10)
        tpa = c_maes.Modules()
        tpa.ssa = c_maes.options.StepSizeAdaptation.TPA
        self.assert_rejected("budget", modules=tpa, lambda0=10, budget=11)
        Settings(3, tpa, lambda0=10, budget=12)
        with self.assertRaisesRegex(ValueError, "budget"):
            fmin(sphere, np.ones(10), 0.5, budget=5)

    def test_setters_check_the_length(self):
        settings = Settings(3)
        for name in ("x0", "lb", "ub"):
            with self.assertRaisesRegex(ValueError, name):
                setattr(settings, name, np.zeros(2))
        settings.x0 = None
        settings.lb = np.zeros(3)

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


class TestBudget(unittest.TestCase):
    """#114: run() checks the break conditions before every generation"""

    def test_run_never_exceeds_the_budget(self):
        for budget in (8, 15, 25):
            es = ModularCMAES(Settings(4, lambda0=8, budget=budget, seed=1))
            es.run(sphere)
            self.assertLessEqual(es.p.stats.evaluations, budget)
            self.assertEqual(es.p.stats.evaluations, 8 * (budget // 8))

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


class TestActiveZeroStep(unittest.TestCase):
    """#118: a zero step with a negative weight made the active update NaN"""

    def test_elitist_active_with_one_parent(self):
        for lambda0 in (4, 10):
            es = ModularCMAES(settings_from_dict(
                5, lambda0=lambda0, mu0=1, elitist=True, active=True, seed=1, budget=10**6,
                restart_strategy="RESTART"))
            for _ in range(300):
                es.step(sphere)
            self.assertLessEqual(len(es.p.stats.centers), 2)  # was about 100 restarts
            self.assertLess(es.p.stats.global_best.y, 1e-10)

    def test_injected_mean(self):
        for matrix_adaptation in ("COVARIANCE", "COVARIANCE_NO_EIGV"):
            es = ModularCMAES(settings_from_dict(
                4, active=True, seed=1, matrix_adaptation=matrix_adaptation,
                always_compute_eigv=matrix_adaptation == "COVARIANCE"))
            for _ in range(3):
                es.step(sphere)
            X = es.ask().copy()
            X[-1] = es.xfavorite  # the mean, with the worst value: a negative weight, y = 0
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
        np.testing.assert_allclose(a.inv_root_C @ a.C @ a.inv_root_C, np.eye(4), atol=1e-8)

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
