"""Regression tests of the fixes of WP8 (dev/overhaul-2.0.md): each test fails with the code before the fix."""

import math
import unittest

import numpy as np

import modcma
from modcma import c_maes


def sphere(x):
    return float(np.sum(x * x))


def ellipsoid_rotated(d, seed=0):
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    w = 10.0 ** (6 * np.arange(d) / (d - 1))
    return lambda x: float(np.sum(w * (rotation @ x) ** 2))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


class UnableToAdaptCounter(c_maes.restart.Criterion):
    """Counts the generations in which UnableToAdapt (the first criterion) is met."""

    def __init__(self):
        super().__init__("UnableToAdaptCounter")
        self.count = 0

    def update(self, parameters):
        if parameters.criteria.items[0].met:
            self.count += 1
        self.met = False


class RestartNow(c_maes.restart.Criterion):
    """Met from the start: the next generation begins with a restart (which resets it)."""

    def __init__(self):
        super().__init__("RestartNow")
        self.met = True

    def update(self, parameters):
        pass


def restart_next_generation(es):
    es.p.criteria.items.append(RestartNow())


class TestBudget(unittest.TestCase):
    def test_a_generation_that_leaves_one_evaluation_is_run(self):
        # The last generation ends one evaluation before the budget, which the final center of
        # mass takes (#111): the budget is used exactly
        es = modcma.ModularCMAES(modcma.Settings(5, lambda0=10, budget=5001, seed=1))
        es.run(lambda x: 1.0)  # a flat function: no restarts, every generation is complete
        self.assertEqual(es.p.stats.evaluations, 5001)
        self.assertEqual(es.p.stats.t, 500)

    def test_tpa_does_not_exceed_the_budget(self):
        for budget in (1000, 1001, 1002, 1003):
            settings = modcma.settings_from_dict(5, ssa="TPA", lambda0=10, budget=budget, seed=1)
            es = modcma.ModularCMAES(settings)
            es.run(sphere)
            self.assertLessEqual(es.p.stats.evaluations, budget)

    def test_a_restart_does_not_exceed_the_budget(self):
        # F30: the generation after a restart evaluates the center first, and IPOP/BIPOP grow
        # lambda; it now takes at most the rest of the budget. A criterion forces the restart
        # where lambda more points still fit, but the center and the grown lambda do not.
        class BeforeTheEnd(c_maes.restart.Criterion):
            def __init__(self):
                super().__init__("BeforeTheEnd")

            def update(self, par):
                remaining = par.settings.budget - par.stats.evaluations
                self.met = par.lamb <= remaining < 1 + 2 * par.lamb

        configs = [
            {"restart_strategy": "IPOP"},
            {"restart_strategy": "BIPOP"},
            {"restart_strategy": "RESTART"},
            {"restart_strategy": "IPOP", "ssa": "TPA"},
            {"restart_strategy": "IPOP", "mirrored": "PAIRWISE"},
            {"restart_strategy": "IPOP", "repelling_type": "COVERAGE"},
        ]
        for config in configs:
            for budget in (1000, 1003, 1017):
                settings = modcma.settings_from_dict(3, budget=budget, seed=1, target=-1.0, **config)
                es = modcma.ModularCMAES(settings)
                es.p.criteria.items.append(BeforeTheEnd())
                es.run(rastrigin)
                self.assertGreaterEqual(len(es.p.stats.centers), 1, (config, budget))
                self.assertLessEqual(es.p.stats.evaluations, budget, (config, budget))
                # the capped generation is still run: little of the budget is left
                self.assertGreater(es.p.stats.evaluations, budget - 2 * es.p.lamb - 3,
                                   (config, budget))

    def test_repelling_does_not_exceed_the_budget(self):
        # F30: the hill-valley tests of repelling evaluate points at every restart; they stop when
        # the budget runs short. CMSA with active restarts every few generations (F6).
        for repelling_type in ("ADAPTIVE", "COVERAGE"):
            for restart_strategy in ("NONE", "IPOP"):
                for seed in range(5):
                    settings = modcma.settings_from_dict(
                        3, budget=1000, seed=seed, target=-1.0, repelling_type=repelling_type,
                        restart_strategy=restart_strategy, matrix_adaptation="CMSA", active=True,
                    )
                    es = modcma.ModularCMAES(settings)
                    es.run(rastrigin)
                    self.assertLessEqual(es.p.stats.evaluations, 1000,
                                         (repelling_type, restart_strategy, seed))


class TestInjection(unittest.TestCase):
    def test_injected_point_enters_the_update_clipped(self):
        # tell(f, X) with a point far outside the distribution (Hansen 2011, as pycma): its step
        # is clipped to sqrt(d) + 2d / (d + 2), for the mean as for the paths and the matrix; the
        # evaluated point is kept as the best solution
        d = 5
        es = modcma.ModularCMAES(modcma.Settings(d, seed=1))
        m_old = np.array(es.p.adaptation.m)
        X = es.ask()
        X[0] = 1e3
        f = [sphere(x) for x in X]
        f[0] = -1.0  # pretend it is the best: it is ranked first
        es.tell(f, X)
        limit = math.sqrt(d) + 2 * d / (d + 2)
        self.assertLessEqual(np.linalg.norm(np.asarray(es.p.pop.Z)[:, 0]), limit + 1e-9)
        np.testing.assert_allclose(es.p.stats.global_best.x, 1e3)
        step = np.linalg.norm(np.array(es.p.adaptation.m) - m_old)
        self.assertLess(step, es.p.settings.sigma0 * 2 * limit)


class TestIssue42SequentialActive(unittest.TestCase):
    def test_no_failed_adaptations(self):
        # modcma 1.2.0: 230 failed eigendecompositions in 216 BBOB runs with active updates
        functions = [ellipsoid_rotated(5), rosenbrock, rastrigin]
        for function in functions:
            for mirrored in ("NONE", "MIRRORED", "PAIRWISE"):
                for seed in range(2):
                    settings = modcma.settings_from_dict(
                        5,
                        sequential_selection=True,
                        active=True,
                        mirrored=mirrored,
                        restart_strategy="RESTART",
                        budget=3000,
                        seed=seed,
                    )
                    es = modcma.ModularCMAES(settings)
                    counter = UnableToAdaptCounter()
                    es.p.criteria.items.append(counter)
                    es.run(function)
                    self.assertEqual(counter.count, 0, (mirrored, seed))


class TestCovarianceUpdate(unittest.TestCase):
    def test_h_sigma_compensates_the_lost_variance(self):
        # C = (1 + c1 delta(h_sigma) - c1 - cmu sum w) C + ..., delta = (1 - h_sigma) cc (2 - cc)
        d = 4
        es = modcma.ModularCMAES(modcma.Settings(d, seed=1))
        es.step(sphere)
        a, w = es.p.adaptation, es.p.weights
        C_before = np.array(a.C)
        pop = es.p.pop  # sorted by the selection, (d, lambda) arrays
        Y = np.asarray(pop.Y)[:, : es.p.mu]
        rank_mu = (Y * w.positive) @ Y.T
        a.hs = False
        pc = np.array(a.pc)
        a.adapt_covariance_matrix(w, es.p.settings.modules, pop, es.p.mu)
        delta = w.cc * (2 - w.cc)
        decay = 1 + w.c1 * delta - w.c1 - w.cmu * w.positive.sum()
        expected = decay * C_before + w.c1 * np.outer(pc, pc) + w.cmu * rank_mu
        np.testing.assert_allclose(a.C, expected, rtol=1e-10, atol=1e-12)

    def test_c_is_updated_every_generation(self):
        # d = 200: the eigendecomposition is done only every few generations, C every one
        settings = modcma.Settings(200, seed=1)
        es = modcma.ModularCMAES(settings)
        self.assertGreater(es.p.weights.lazy_update_interval, 1.0)
        changes = 0
        previous = np.array(es.p.adaptation.C)
        for _ in range(6):
            es.step(sphere)
            changes += not np.array_equal(previous, es.p.adaptation.C)
            previous = np.array(es.p.adaptation.C)
        self.assertEqual(changes, 6)
        self.assertLess(es.p.stats.n_updates, 6)

    def test_negative_weights_scaled_with_c1_over_cmu(self):
        w = modcma.ModularCMAES(modcma.Settings(10, seed=1)).p.weights
        d = 10
        mueff_neg = w.negative.sum() ** 2 / np.sum(w.negative**2)
        expected = min(
            1 + w.c1 / w.cmu,
            1 + 2 * mueff_neg / (w.mueff + 2),
            (1 - w.c1 - w.cmu) / (d * w.cmu),
        )
        self.assertAlmostEqual(-w.negative.sum(), expected, places=12)

    def test_p_sigma_starts_at_zero(self):
        for option in ("COVARIANCE", "MATRIX", "CHOLESKY", "CMSA", "NATURAL_GRADIENT", "NONE"):
            es = modcma.ModularCMAES(modcma.settings_from_dict(5, matrix_adaptation=option))
            np.testing.assert_array_equal(es.p.adaptation.ps, 0.0, option)


class TestNaturalGradient(unittest.TestCase):
    def test_sigma0_sets_the_initial_spread(self):
        for sigma0 in (0.01, 1.0):
            settings = modcma.settings_from_dict(
                10, matrix_adaptation="NATURAL_GRADIENT", sigma0=sigma0, lambda0=200, seed=1
            )
            es = modcma.ModularCMAES(settings)
            X = es.ask()
            spread = float(np.std(X - es.p.adaptation.m))
            self.assertAlmostEqual(spread / sigma0, 1.0, delta=0.1)

    def test_learning_rate_of_xnes(self):
        d = 10
        es = modcma.ModularCMAES(modcma.settings_from_dict(d, matrix_adaptation="NATURAL_GRADIENT"))
        self.assertAlmostEqual(es.p.weights.cs, (9 + 3 * math.log(d)) / (5 * d * math.sqrt(d)))


class TestRestarts(unittest.TestCase):
    def test_bipop_small_populations(self):
        # lambda_small = floor(lambda_init (lambda_large / lambda_init)^(U^2)), in
        # [lambda_init, lambda_large] with lambda_large the latest large population (#140);
        # before: at most lambda_init
        settings = modcma.settings_from_dict(
            5, restart_strategy="BIPOP", budget=60_000, seed=3, x0=np.full(5, 3.0)
        )
        es = modcma.ModularCMAES(settings)
        lambdas = []
        while es.step(rastrigin):
            if not lambdas or es.p.lamb != lambdas[-1]:
                lambdas.append(es.p.lamb)
        initial = settings.lambda0
        small = [lam for lam in lambdas[1:] if lam <= max(lambdas[1:]) // 2]
        self.assertTrue(all(lam >= initial for lam in lambdas), lambdas)
        self.assertTrue(any(lam > initial for lam in small) or len(small) < 3, lambdas)

    def test_ipop_factor_need_not_be_an_integer(self):
        es = modcma.ModularCMAES(modcma.settings_from_dict(5, restart_strategy="IPOP", seed=1))
        es.p.restart_strategy.ipop_factor = 1.5
        lam = es.p.lamb
        es.p.restart_strategy.update(es.p)
        self.assertEqual(es.p.lamb, round(lam * 1.5))

    def test_tolx_relative_to_sigma0(self):
        # met when sigma sqrt(C_ii) < tol sigma0 (and sigma |pc_i| < tol sigma0)
        es = modcma.ModularCMAES(modcma.settings_from_dict(3, restart_strategy="RESTART", seed=1))
        tolx = next(c for c in es.p.criteria.items if c.name == "TolX")
        sigma0 = es.p.settings.sigma0
        es.p.mutation.sigma = 1.5 * c_maes.restart.TolX.tolerance * sigma0
        tolx.update(es.p)
        self.assertFalse(tolx.met)
        es.p.mutation.sigma = 0.5 * c_maes.restart.TolX.tolerance * sigma0
        tolx.update(es.p)
        self.assertTrue(tolx.met)

    @staticmethod
    def adaptive_repelling_after_one_run(objective):
        es = modcma.ModularCMAES(
            modcma.settings_from_dict(
                3, restart_strategy="RESTART", repelling_type="ADAPTIVE", budget=20_000, seed=1
            )
        )
        for _ in range(4):
            es.step(objective)
        restart_next_generation(es)
        es.step(objective)  # the restart: the center of mass, then the update of the archive
        return es

    def test_adaptive_repelling_counts_the_first_run(self):
        # #121: the first run entered the running average of the run lengths as 1 evaluation
        # (the average stayed 1.0 after it); now with its full length, which starts the average
        # (D2: it started at 1)
        es = self.adaptive_repelling_after_one_run(sphere)
        repelling = es.p.repelling
        alpha = repelling.restart_eval_alpha
        self.assertEqual(len(es.p.stats.centers), 1)
        n = es.p.stats.centers[0].e  # the evaluations of the first run, with its center
        self.assertEqual(repelling.last_restart_evaluations, n)
        self.assertEqual(repelling.average_restart_evaluations, n)
        # Every later run counts from the previous restart, its hill-valley points included
        average = repelling.average_restart_evaluations
        while len(es.p.stats.centers) < 2:
            es.step(sphere)
        m = repelling.last_restart_evaluations - n
        self.assertEqual(repelling.last_restart_evaluations, es.p.stats.centers[1].e)
        self.assertAlmostEqual(
            repelling.average_restart_evaluations, (1 - alpha) * average + alpha * m
        )

    def test_adaptive_repelling_counts_a_run_without_a_finite_result(self):
        # #121: a run whose center and best are not finite updates no archive, but its
        # evaluations count; the update was skipped, and the next run counted both
        state = {"values": math.inf}

        def objective(x):
            return state["values"] if state["values"] is not None else sphere(x)

        es = self.adaptive_repelling_after_one_run(objective)
        repelling = es.p.repelling
        self.assertEqual(len(es.p.stats.centers), 1)
        self.assertFalse(math.isfinite(es.p.stats.centers[0].y))
        self.assertEqual(len(repelling.archive), 0)
        n = es.p.stats.centers[0].e
        self.assertEqual(repelling.last_restart_evaluations, n)
        self.assertEqual(repelling.average_restart_evaluations, n)
        state["values"] = None
        while len(es.p.stats.centers) < 2:
            es.step(objective)
        self.assertEqual(repelling.last_restart_evaluations, es.p.stats.centers[1].e)
        self.assertEqual(len(repelling.archive), 1)

    def test_first_run_starts_at_the_center_of_the_bounds(self):
        settings = modcma.Settings(3, lb=np.zeros(3), ub=np.full(3, 10.0))
        es = modcma.ModularCMAES(settings)
        np.testing.assert_allclose(es.p.adaptation.m, 5.0)


class TestSampling(unittest.TestCase):
    def test_orthogonal_samples_are_orthogonal(self):
        # n samples per generation, in blocks of at most d from the start of each (#76, F11)
        for d, n in ((10, 10), (5, 3), (4, 10)):
            sampler = c_maes.sampling.Orthogonal(c_maes.sampling.Gaussian(d), n)
            for _ in range(3):
                Z = np.array([sampler() for _ in range(n)]).T
                for start in range(0, n, d):
                    B = Z[:, start : start + d]
                    G = B.T @ B
                    off = G - np.diag(np.diag(G))
                    self.assertLess(np.abs(off).max(), 1e-10 * np.abs(G).max(), (d, n))

    def test_wrappers_pass_on_the_expected_length(self):
        gaussian = c_maes.sampling.GaussianTransformer(c_maes.sampling.Uniform(10))
        expected = gaussian.expected_length()
        self.assertLess(expected, math.sqrt(10))
        self.assertAlmostEqual(c_maes.sampling.Mirrored(gaussian).expected_length(), expected)
        self.assertAlmostEqual(
            c_maes.sampling.Orthogonal(gaussian, 10).expected_length(), expected
        )
        for option in ("MIRRORED", "PAIRWISE"):
            es = modcma.ModularCMAES(modcma.settings_from_dict(10, mirrored=option))
            self.assertAlmostEqual(es.p.weights.expected_length_z, expected)

    def test_exponential_weights(self):
        es = modcma.ModularCMAES(modcma.settings_from_dict(10, weights="EXPONENTIAL"))
        mu = es.p.mu
        expected = 2.0 ** -np.arange(1, mu + 1) + 1 / (mu * 2.0**mu)
        np.testing.assert_allclose(es.p.weights.positive, expected)

    def test_cotn_stays_inside_the_box(self):
        settings = modcma.Settings(4, lb=np.zeros(4), ub=np.ones(4))
        cotn = c_maes.bounds.COTN(4)
        for _ in range(200):
            x = cotn.correct_x(np.array([-0.5, 1.5, 2.0, 0.5]),
                               np.array([True, True, True, False]), 50.0, settings)
            self.assertTrue(np.all((x >= 0) & (x <= 1)), x)


class TestSettingsAndHelpers(unittest.TestCase):
    def test_mu_not_larger_than_lambda_with_self_adaptation(self):
        # mu0 > lambda0 was replaced by lambda0 // 2 (not with SA, before 2.0); now it raises
        # ValueError (#136)
        with self.assertRaisesRegex(ValueError, "mu0 = 10 exceeds lambda0 = 6"):
            modcma.settings_from_dict(5, ssa="SA", lambda0=6, mu0=10)
        self.assertGreaterEqual(modcma.settings_from_dict(5, ssa="SA", lambda0=3).mu0, 1)

    def test_cpp_defaults_are_quiet(self):
        self.assertFalse(modcma.Parameters(3).settings.verbose)

    def test_set_module_accepts_enum_members(self):
        modules = modcma.Modules()
        matrix = modcma.options.MatrixAdaptationType.MATRIX
        self.assertTrue(modcma.c_maes.set_module(modules, "matrix_adaptation", matrix))
        self.assertEqual(modules.matrix_adaptation, matrix)
        self.assertTrue(modcma.c_maes.set_module(modules, "ssa", "TPA"))

    def test_population_addition(self):
        a = c_maes.Population(2, 3)
        b = c_maes.Population(2, 2)
        c = a + b
        self.assertEqual((a.n, b.n, c.n), (3, 2, 5))
        a += b
        self.assertEqual(a.n, 5)

    def test_bounds_checks(self):
        sequence = c_maes._internal.CachedShuffleSequence(2)
        with self.assertRaises((IndexError, ValueError)):
            sequence.get_index(10**9)
        with self.assertRaises(ValueError):
            c_maes.sampling.CachedSampler([])

    def test_no_improvement_tolerance_is_bound(self):
        self.assertEqual(c_maes.restart.NoImprovement.tolerance, 1e-20)

    def test_old_names_still_work(self):
        # The misspelled names of 1.x work, with a DeprecationWarning (#82)
        import warnings

        u, v = np.array([0.0, 3.0]), np.array([4.0, 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertIs(c_maes.selection.Elitsm, c_maes.selection.Elitism)
            self.assertIs(c_maes.selection.NoElitsm, c_maes.selection.NoElitism)
            self.assertEqual(c_maes.repelling.euclidian(u, v), c_maes.repelling.euclidean(u, v))
            self.assertEqual(c_maes.repelling.mahanolobis(u, v, np.eye(2)),
                             c_maes.repelling.mahalanobis(u, v, np.eye(2)))


class TestCenterOfMass(unittest.TestCase):
    def test_center_of_mass_is_the_mean(self):
        es = modcma.ModularCMAES(modcma.Settings(4, integer_variables=[1], seed=1))
        es.step(sphere)
        m = np.array(es.p.adaptation.m)
        expected = m.copy()
        expected[1] = np.round(expected[1])
        np.testing.assert_allclose(es.center_of_mass, expected)

    def test_center_of_mass_under_noise(self):
        # With noise the recorded value of the best point is biased low, and with a large
        # population the mean is closer to the optimum (median over seeds; measured 12-15 of 20
        # runs, about 1.6 times closer)
        favorite, best, biased = [], [], 0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            noisy = lambda x: sphere(x - 1) + rng.normal(scale=1.0)  # noqa: E731
            settings = modcma.Settings(
                5, x0=np.zeros(5), sigma0=0.5, lambda0=40, budget=8000, seed=seed
            )
            es = modcma.ModularCMAES(settings)
            es.run(noisy)
            favorite.append(sphere(es.center_of_mass - 1))
            best.append(sphere(es.p.stats.global_best.x - 1))
            biased += es.p.stats.global_best.y < best[-1]
        self.assertLess(np.median(favorite), np.median(best))
        self.assertGreaterEqual(biased, 9)


class TestPackaging(unittest.TestCase):
    def test_py_typed(self):
        import os

        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(modcma.__file__), "py.typed")))


class TestPerRunState(unittest.TestCase):
    """#77: state that belongs to the current run, not to all runs."""

    def test_hsigma_counts_the_generations_of_the_run(self):
        # F12: the bias correction of h_sigma uses the run's generation g, 1 - (1 - cs)^(2(g+1));
        # it used evaluations / lambda over all runs (and TPA's extra points)
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(4, ssa="TPA", seed=3, verbose=False))
        for _ in range(30):
            es.step(sphere)
        restart_next_generation(es)
        for g in range(1, 6):
            es.step(sphere)  # the first one begins with the restart
            self.assertEqual(len(es.p.stats.centers), 1)
            self.assertEqual(es.p.stats.t_run, g)
            a, w = es.p.adaptation, es.p.weights
            length = np.linalg.norm(a.ps) / math.sqrt(1 - (1 - w.cs) ** (2 * g))
            self.assertEqual(bool(a.hs), bool(length < w.expected_length_ps))

    def test_sequential_selection_after_a_restart(self):
        # F13: the reference is the best of the previous generation; with the best of all runs,
        # a new run that starts far from the old optimum never stopped a generation early
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(
            5, sequential_selection=True, mirrored="MIRRORED", seed=1, verbose=False))
        for _ in range(150):
            es.step(sphere)
        self.assertLess(es.p.stats.global_best.y, 1e-6)
        restart_next_generation(es)
        sizes = []
        for _ in range(20):
            es.step(sphere)  # the first one begins with the restart
            sizes.append(es.p.pop.n)
        self.assertEqual(len(es.p.stats.centers), 1)
        self.assertEqual(sizes[0], es.p.lamb)  # the first generation of a run is complete
        self.assertLess(min(sizes), es.p.lamb)


class TestTpaAndThreshold(unittest.TestCase):
    """#78: TPA's ties, threshold convergence per generation, relative to E||z||."""

    def test_tpa_tie_after_a_restart(self):
        # F15: after a restart dm = 0, both TPA points are the mean: no change of sigma (it grew
        # by 16 % at every restart)
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(4, ssa="TPA", seed=2, verbose=False))
        for _ in range(20):
            es.step(sphere)
        restart_next_generation(es)
        es.step(sphere)  # the restart (sigma = sigma0), then the first generation of the new run
        self.assertEqual(len(es.p.stats.centers), 1)
        self.assertEqual(es.p.mutation.rank_tpa, 0.0)
        self.assertEqual(es.p.mutation.sigma, es.p.settings.sigma0)

    def test_threshold_per_generation(self):
        # F24: the threshold is computed once per generation, relative to E||z||, and every z of
        # the generation is at least that long
        for box in (5.0, 100.0):  # does not depend on the box
            es = c_maes.ModularCMAES(c_maes.settings_from_dict(
                5, threshold_convergence=True, seed=1, verbose=False, budget=2000,
                lb=np.full(5, -box), ub=np.full(5, box)))
            for _ in range(5):
                es.step(sphere)
            es.ask()
            tc = c_maes.mutation.ThresholdConvergence()
            t = tc.threshold(es.p.sampler.expected_length(), 2000, es.p.stats.evaluations)
            self.assertEqual(es.p.mutation.generation_threshold, t)
            self.assertTrue(np.all(np.linalg.norm(es.p.pop.Z, axis=0) >= t - 1e-12))

    def test_threshold_past_the_budget(self):
        # F24: budget - evaluations wrapped around (size_t) past the budget in ask-tell
        tc = c_maes.mutation.ThresholdConvergence()
        self.assertEqual(tc.threshold(2.0, 100, 150), 0.0)
        self.assertAlmostEqual(tc.threshold(2.0, 100, 0), tc.init_threshold * 2.0)
        es = c_maes.ModularCMAES(c_maes.settings_from_dict(
            3, threshold_convergence=True, seed=1, verbose=False, budget=50))
        for _ in range(20):  # ask-tell keeps going past the budget
            X = es.ask()
            es.tell(np.array([sphere(x) for x in X]))
        self.assertGreater(es.p.stats.evaluations, 50)
        self.assertEqual(es.p.mutation.generation_threshold, 0.0)
        self.assertTrue(np.all(np.isfinite(es.ask())))


if __name__ == "__main__":
    unittest.main()
