"""The restart criteria for every matrix adaptation (F25) and BIPOP's regimes (F26), #80; the
criteria per run (#139) and mu after a restart (#150)."""

import math
import unittest

import numpy as np

import modcma
from modcma import c_maes

TYPES = [
    "COVARIANCE",
    "COVARIANCE_NO_EIGV",
    "MATRIX",
    "SEPARABLE",
    "CHOLESKY",
    "CMSA",
    "NONE",
    "NATURAL_GRADIENT",
]
KEEPS_METRIC = {"COVARIANCE", "SEPARABLE", "NONE"}
WITH_PC = {"COVARIANCE", "COVARIANCE_NO_EIGV", "SEPARABLE", "CHOLESKY"}


def sphere(x):
    return float(np.sum(x * x))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def optimizer(matrix_adaptation="COVARIANCE", d=4, **kwargs):
    kwargs.setdefault("restart_strategy", "RESTART")
    kwargs.setdefault("seed", 1)
    return modcma.ModularCMAES(
        modcma.settings_from_dict(d, matrix_adaptation=matrix_adaptation, **kwargs)
    )


def criteria(es):
    return {item.name: item for item in es.p.criteria.items}


class TestCriteriaPerAdaptation(unittest.TestCase):
    def test_created_for_every_adaptation(self):
        for name in TYPES:
            with self.subTest(name):
                es = optimizer(name)
                names = list(criteria(es))
                self.assertIn("TolX", names)
                self.assertIn("NoEffectCoord", names)
                self.assertEqual("ConditionC" in names, name in KEEPS_METRIC)
                self.assertEqual("NoEffectAxis" in names, name in KEEPS_METRIC)
                self.assertEqual(es.p.adaptation.keeps_metric(), name in KEEPS_METRIC)

    def test_covariance_with_another_step_size_adaptation(self):
        # Settings convert COVARIANCE to COVARIANCE_NO_EIGV when ssa is not CSA: TolX and
        # NoEffectCoord stay (they were lost)
        es = optimizer("COVARIANCE", ssa="MSR")
        self.assertEqual(
            es.p.settings.modules.matrix_adaptation,
            c_maes.options.MatrixAdaptationType.COVARIANCE_NO_EIGV,
        )
        names = list(criteria(es))
        self.assertIn("TolX", names)
        self.assertIn("NoEffectCoord", names)
        self.assertNotIn("ConditionC", names)

    def test_the_order_of_covariance_is_kept(self):
        names = list(criteria(optimizer("COVARIANCE")))
        self.assertEqual(names[-4:], ["TolX", "ConditionC", "NoEffectAxis", "NoEffectCoord"])

    def test_none_without_restarts(self):
        es = optimizer("MATRIX", restart_strategy="NONE")
        self.assertEqual(list(criteria(es)), ["UnableToAdapt"])

    def test_every_strategy_but_none_has_the_criteria(self):
        # Also STOP, which stops the optimizer when one is met (#192: it was a test of the order of
        # the enum, restart_strategy >= RESTART)
        expected = list(criteria(optimizer("COVARIANCE", restart_strategy="RESTART")))
        self.assertGreater(len(expected), 10)
        for strategy in c_maes.options.RestartStrategy.__members__:
            with self.subTest(strategy):
                names = list(criteria(optimizer("COVARIANCE", restart_strategy=strategy)))
                self.assertEqual(names, ["UnableToAdapt"] if strategy == "NONE" else expected)

    def test_the_default_criteria_compute_no_metric(self):
        # COVARIANCE_NO_EIGV computes its metric into its d and B only when asked (#123): with the
        # default criteria they keep their initial values during a run
        es = optimizer("COVARIANCE_NO_EIGV", ssa="MSR")
        for _ in range(30):
            es.step(sphere)
        self.assertEqual(len(es.p.stats.centers), 0)
        np.testing.assert_array_equal(es.p.adaptation.d, np.ones(4))
        self.assertFalse(np.allclose(es.p.adaptation.metric().d, 1.0))


class TestTolX(unittest.TestCase):
    def test_every_adaptation(self):
        for name in TYPES:
            with self.subTest(name):
                es = optimizer(name, d=3)
                es.step(sphere)
                tolx = criteria(es)["TolX"]
                d = 3
                self.assertEqual(len(tolx.tolx_vector), 2 * d if name in WITH_PC else d)
                sigma0 = es.p.settings.sigma0
                variances = np.array(es.p.adaptation.coordinate_wise_variances)
                # the largest sigma sqrt(C_ii) at 1.5 and 0.5 times tol sigma0 (with pc = 0)
                scale = c_maes.restart.TolX.tolerance * sigma0 / math.sqrt(variances.max())
                es.p.mutation.sigma = 1.5 * scale
                tolx.update(es.p)
                self.assertFalse(tolx.met)
                np.testing.assert_allclose(
                    tolx.tolx_vector[:d], np.sqrt(variances) * es.p.mutation.sigma
                )
                if name in WITH_PC:
                    es.p.adaptation.pc[:] = 0.0
                es.p.mutation.sigma = 0.5 * scale
                tolx.update(es.p)
                self.assertTrue(tolx.met)

    def test_pc_counts_where_there_is_one(self):
        for name in WITH_PC:
            with self.subTest(name):
                es = optimizer(name, d=3)
                es.step(sphere)
                tolx = criteria(es)["TolX"]
                sigma0 = es.p.settings.sigma0
                es.p.adaptation.coordinate_wise_variances[:] = 1.0
                es.p.mutation.sigma = 0.5 * c_maes.restart.TolX.tolerance * sigma0
                es.p.adaptation.pc[:] = 1.0
                tolx.update(es.p)
                self.assertTrue(tolx.met)
                es.p.adaptation.pc[1] = 3.0  # sigma |pc_1| = 1.5 tol sigma0
                tolx.update(es.p)
                self.assertFalse(tolx.met)


class TestNoEffect(unittest.TestCase):
    """pycma compares m + step with m in floating point: relative to m, not to 10 epsilon."""

    def test_no_effect_coord_is_relative_to_the_mean(self):
        for name in TYPES:
            with self.subTest(name):
                es = optimizer(name)
                es.step(sphere)
                item = criteria(es)["NoEffectCoord"]
                es.p.adaptation.coordinate_wise_variances[:] = 1.0
                # 0.2 sigma = 2e-10 is below half an ulp of 1e8 (7.5e-9): no effect. Absolute,
                # it was far above 10 epsilon
                es.p.mutation.sigma = 1e-9
                es.p.adaptation.m[:] = [1e8, 0.0, 0.0, 0.0]
                item.update(es.p)
                self.assertTrue(item.met)
                # At m = 0 any step has an effect, even below 10 epsilon (absolute: met)
                es.p.adaptation.m[:] = 0.0
                es.p.mutation.sigma = 1e-20
                item.update(es.p)
                self.assertFalse(item.met)

    def test_no_effect_axis_is_relative_to_the_mean(self):
        for name in KEEPS_METRIC:
            with self.subTest(name):
                es = optimizer(name)
                es.step(sphere)
                item = criteria(es)["NoEffectAxis"]
                metric = es.p.adaptation.metric()
                i = es.p.stats.t % 4
                es.p.mutation.sigma = 1e-9 / metric.d[i]  # 0.1 sigma D_i = 1e-10
                es.p.adaptation.m[:] = 1e8
                item.update(es.p)
                self.assertTrue(item.met)
                es.p.adaptation.m[:] = 1.0  # half an ulp of 1 is 1.1e-16: the step has an effect
                item.update(es.p)
                self.assertFalse(item.met)

    def test_no_effect_axis_along_a_rotated_axis(self):
        es = optimizer("COVARIANCE")
        es.step(sphere)
        item = criteria(es)["NoEffectAxis"]
        i = es.p.stats.t % 4
        rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)))
        es.p.adaptation.B[:] = rotation
        es.p.adaptation.d[:] = 1.0
        es.p.mutation.sigma = 1e-9
        es.p.adaptation.m[:] = 1e8
        item.update(es.p)
        self.assertTrue(item.met)
        # one coordinate where the step has an effect is enough to keep going
        es.p.adaptation.m[int(np.argmax(np.abs(rotation[:, i])))] = 0.0
        item.update(es.p)
        self.assertFalse(item.met)


class TestSigmaCriteria(unittest.TestCase):
    def test_max_sigma_is_relative_to_the_start_of_the_run(self):
        # TolXUp (Hansen 2016): sigma max(D) grew by more than 1e4 since the start of the run
        for sigma0, sigma, met in ((1e-3, 20.0, True), (1e-3, 5.0, False), (1e3, 2e4, False)):
            with self.subTest(sigma0=sigma0, sigma=sigma):
                es = optimizer("COVARIANCE", sigma0=sigma0)
                item = criteria(es)["MaxSigma"]
                self.assertEqual(item.start, sigma0)  # C = I at the start
                es.p.mutation.sigma = sigma
                item.update(es.p)
                self.assertEqual(item.met, met)

    def test_max_sigma_follows_c(self):
        for name in set(TYPES) - {"NONE"}:  # C = I
            with self.subTest(name):
                es = optimizer(name, sigma0=1.0)
                es.step(sphere)
                item = criteria(es)["MaxSigma"]
                es.p.mutation.sigma = 1e3
                self.set_longest_axis(es, name, 9.0)
                item.update(es.p)
                self.assertFalse(item.met)
                self.set_longest_axis(es, name, 11.0)
                item.update(es.p)
                self.assertTrue(item.met)

    @staticmethod
    def set_longest_axis(es, name, length):
        if name in KEEPS_METRIC and name != "NONE":
            es.p.adaptation.d[:] = [length, 1.0, 1.0, 0.5]
        else:
            es.p.adaptation.coordinate_wise_variances[:] = [length**2, 1.0, 1.0, 0.25]

    def test_d_sigma_follows_a_scaled_c(self):
        # MaxDSigma and MinDSigma compare sigma / sigma0 with tolerance * max(D) and min(D):
        # for the dense types sqrt(max C_ii) and sqrt(min C_ii) of the diagonal of C (decision (b)
        # of #80). Scaling the factor by s scales C by s^2, its diagonal too.
        for name in ("COVARIANCE_NO_EIGV", "MATRIX", "CHOLESKY", "CMSA", "NATURAL_GRADIENT"):
            with self.subTest(name):
                es = optimizer(name, sigma0=1.0)
                for _ in range(3):
                    es.step(sphere)
                a = es.p.adaptation
                variances = np.array(a.coordinate_wise_variances)
                factor = np.column_stack([a.compute_y(e) for e in np.eye(4)])
                np.testing.assert_allclose(variances, np.sum(factor**2, axis=1), rtol=1e-12)

                items = criteria(es)
                upper = c_maes.restart.MaxDSigma.tolerance
                lower = c_maes.restart.MinDSigma.tolerance
                for s in (1e-6, 1.0, 1e6):
                    a.coordinate_wise_variances[:] = variances * s**2
                    longest = s * math.sqrt(variances.max())
                    shortest = s * math.sqrt(variances.min())
                    for factor, met in ((0.5, False), (2.0, True)):
                        es.p.mutation.sigma = factor * upper * longest
                        items["MaxDSigma"].update(es.p)
                        self.assertEqual(items["MaxDSigma"].met, met)
                    for factor, met in ((2.0, False), (0.5, True)):
                        es.p.mutation.sigma = factor * lower * shortest
                        items["MinDSigma"].update(es.p)
                        self.assertEqual(items["MinDSigma"].met, met)

    def test_d_sigma_reads_the_metric_where_it_is_kept(self):
        es = optimizer("SEPARABLE", sigma0=1.0)
        es.step(sphere)
        item = criteria(es)["MaxDSigma"]
        es.p.adaptation.d[:] = [1e4, 1.0, 1.0, 1.0]
        es.p.mutation.sigma = c_maes.restart.MaxDSigma.tolerance * 1e3
        item.update(es.p)
        self.assertFalse(item.met)
        es.p.mutation.sigma = c_maes.restart.MaxDSigma.tolerance * 1e5
        item.update(es.p)
        self.assertTrue(item.met)


def pycma_regimes(run_lengths):
    """The regime of every run after the first, as pycma 4.6.0's fmin with bipop=True chooses it
    (cma/evolution_strategy.py): the first run is "small"; a run is small if
    sum(small_i) < max(1, sum(large_i)), else large. Also the large lambda factor and the cap of
    each small run, 0.5 * sum(large_i) evaluations (maxiter = that / popsize)."""
    small_i, large_i = [], []
    regime = "small"
    regimes, caps = [], []
    for evaluations in run_lengths:
        (small_i if regime == "small" else large_i).append(evaluations)
        regime = "small" if sum(small_i) < max(1, sum(large_i)) else "large"
        regimes.append(regime)
        caps.append(0.5 * sum(large_i) if regime == "small" else None)
    return regimes, caps


class TestBIPOP(unittest.TestCase):
    RUNS = [800, 3000, 400, 900, 2500, 10, 60_000, 7000, 7000, 7000, 7000, 30_000, 1, 1]

    def scripted(self, run_lengths, budget=10**9, **kwargs):
        es = optimizer("COVARIANCE", d=5, restart_strategy="BIPOP", budget=budget, **kwargs)
        strategy = es.p.restart_strategy
        lambda0 = es.p.settings.lambda0
        regimes, lambdas, max_iters = [], [], []
        for evaluations in run_lengths:
            es.p.stats.evaluations += evaluations
            strategy.update(es.p)
            regimes.append("large" if strategy.large_run else "small")
            lambdas.append(es.p.lamb)
            max_iters.append(strategy.max_iter)
        return es, lambda0, regimes, lambdas, max_iters

    def test_the_regimes_match_pycma(self):
        es, lambda0, regimes, lambdas, max_iters = self.scripted(self.RUNS)
        expected, caps = pycma_regimes(self.RUNS)
        self.assertEqual(regimes, expected)
        strategy = es.p.restart_strategy
        self.assertEqual(strategy.used_small + strategy.used_large, sum(self.RUNS))
        n_large = 0
        for regime, lam, max_iter, cap in zip(regimes, lambdas, max_iters, caps):
            if regime == "large":
                n_large += 1
                self.assertEqual(lam, lambda0 * 2**n_large)
                self.assertEqual(max_iter, 2**64 - 1)
            else:
                # between lambda0 and the last large population
                self.assertTrue(lambda0 <= lam <= lambda0 * 2**n_large, (lam, n_large))
                # pycma's maxiter is not rounded and a run stops at countiter >= maxiter: after
                # ceil(maxiter) generations (#139 D3; it was rounded down)
                self.assertEqual(max_iter, math.ceil(cap / lam))

    def test_a_run_longer_than_half_the_budget(self):
        # Before: the rest of the budget after the first run (9000) was split into halves; the
        # large run of 8000 left its half at 4500 - 8000, which wrapped around to about 2^64
        runs = [1000, 8000, 3000, 500, 4000]
        es, _, regimes, _, _ = self.scripted(runs, budget=10_000)
        self.assertEqual(regimes, pycma_regimes(runs)[0])
        strategy = es.p.restart_strategy
        self.assertEqual(strategy.used_large, 8000)
        self.assertEqual(strategy.used_small, 1000 + 3000 + 500 + 4000)

    def test_the_counters_never_wrap(self):
        # A run that uses more than half the remaining budget wrapped the regime's budget left
        # around (Rastrigin, d = 2, budget 20000, seed 0: 2^64 - 133); BIPOP then became IPOP
        settings = modcma.settings_from_dict(
            2, restart_strategy="BIPOP", budget=20_000, seed=0, lb=np.full(2, -5.0),
            ub=np.full(2, 5.0),
        )
        es = modcma.ModularCMAES(settings)
        regimes = set()
        while es.step(rastrigin):
            strategy = es.p.restart_strategy
            self.assertLessEqual(strategy.used_small + strategy.used_large, es.p.stats.evaluations)
            if strategy.lambda_large:
                regimes.add(strategy.large_run)
        self.assertEqual(regimes, {True, False})

    def test_lambda_small_is_rounded_down(self):
        _, lambda0, regimes, lambdas, _ = self.scripted(self.RUNS * 5)
        small = [lam for regime, lam in zip(regimes, lambdas) if regime == "small"]
        self.assertTrue(any(lam % 2 for lam in small), small)  # odd values occur

    def test_lambda_small_is_even_for_pairwise(self):
        _, _, regimes, lambdas, _ = self.scripted(self.RUNS * 5, mirrored="PAIRWISE")
        self.assertIn("small", regimes)
        self.assertTrue(all(lam % 2 == 0 for lam in lambdas), lambdas)

    def test_exceeded_max_iter_enforces_the_cap_of_a_small_run(self):
        es, _, regimes, lambdas, max_iters = self.scripted(self.RUNS[:3])
        self.assertEqual(regimes[-1], "small")
        item = criteria(es)["ExceededMaxIter"]
        item.reset(es.p)
        own = int(100 + 50 * (5 + 3) ** 2 / math.sqrt(es.p.lamb))
        self.assertEqual(item.max_iter, min(own, max_iters[-1]))
        self.assertLess(max_iters[-1], own)

    def test_small_runs_are_capped_in_a_run(self):
        # The first and the large runs end after 5 generations, so the small runs on the sphere
        # (which take far longer to converge) reach their cap, half the evaluations of the large
        # regime. A small run takes at most max_iter generations (at least one), exactly max_iter
        # when ExceededMaxIter ends it; it took max_iter + 2 (#139).
        settings = modcma.settings_from_dict(
            5, restart_strategy="BIPOP", budget=50_000, seed=2, sigma0=1e-3
        )
        es = modcma.ModularCMAES(settings)
        es.p.criteria.items.append(EndLargeRuns())
        runs = []  # (large_run, the limit of ExceededMaxIter, generation of the restart)
        used = 0
        while es.step(sphere):
            strategy = es.p.restart_strategy
            if strategy.used_budget != used:  # a restart started this generation
                used = strategy.used_budget
                start = es.p.stats.t - es.p.stats.t_run
                limit = criteria(es)["ExceededMaxIter"].max_iter
                self.assertLessEqual(limit, strategy.max_iter)
                runs.append((strategy.large_run, limit, start))
        centers = es.p.stats.centers[1:]
        capped = 0
        for (large, limit, start), center in zip(runs, centers):
            if not large:
                generations = center.t + 1 - start
                self.assertLessEqual(generations, max(limit, 1))
                if "ExceededMaxIter" in center.reasons:
                    self.assertEqual(generations, max(limit, 1))
                    capped += 1
        self.assertGreater(capped, 0)

    def test_d_sigma_compare_with_the_sigma_of_the_run(self):
        # #139: MaxDSigma and MinDSigma divide sigma by the step size at the start of the run,
        # which is smaller than Settings.sigma0 in the small runs (pycma's es.sigma0 of each run)
        settings = modcma.settings_from_dict(5, restart_strategy="BIPOP", budget=20_000, seed=2)
        es = modcma.ModularCMAES(settings)
        recorder = SigmaAtReset()
        es.p.criteria.items.append(EndLargeRuns())
        es.p.criteria.items.append(recorder)
        recorder.reset(es.p)  # the first run (the optimizer reset its criteria when created)
        items = criteria(es)
        small_runs = 0
        while es.step(sphere):
            for name in ("MaxDSigma", "MinDSigma"):
                self.assertEqual(items[name].sigma_start, recorder.sigma)
            strategy = es.p.restart_strategy
            if strategy.lambda_large and not strategy.large_run:
                small_runs += 1
                self.assertLess(recorder.sigma, settings.sigma0)
        self.assertGreater(small_runs, 0)


class RestartEvery(c_maes.restart.Criterion):
    """Met after n generations of a run."""

    def __init__(self, n):
        super().__init__("RestartEvery")
        self.n = n

    def update(self, parameters):
        self.met = parameters.stats.t + 1 - self.last_restart >= self.n


class SigmaAtReset(c_maes.restart.Criterion):
    """Records the step size at every reset, the start of a run; never met."""

    def __init__(self):
        super().__init__("SigmaAtReset")
        self.sigma = None

    def on_reset(self, parameters):
        self.sigma = parameters.mutation.sigma

    def update(self, parameters):
        self.met = False


class TestRunRelativeCriteria(unittest.TestCase):
    """#139: ExceededMaxIter, MaxDSigma and MinDSigma measure the current run."""

    def run_lengths(self, max_iter, ask_tell):
        es = optimizer("COVARIANCE", d=4, budget=600)
        # ExceededMaxIter takes the smaller of its own limit and the strategy's
        es.p.restart_strategy.max_iter = max_iter
        item = criteria(es)["ExceededMaxIter"]
        es.p.criteria.items = [item]
        item.reset(es.p)
        self.assertEqual(item.max_iter, max_iter)
        if ask_tell:
            while True:
                X = es.ask()
                if es.tell([sphere(x) for x in X]) and es.break_conditions():
                    break
        else:
            es.run(sphere)
        centers = es.p.stats.centers
        lengths = np.diff([-1] + [center.t for center in centers])
        return lengths, [center.reasons for center in centers]

    def test_a_run_takes_max_iter_generations(self):
        # pycma's maxiter: es.countiter >= opts['maxiter'], checked after each iteration
        # (cma/evolution_strategy.py:4527-4529); Hansen (2009): "the maximal number of iterations
        # in each run"
        for max_iter in (1, 3, 10):
            for ask_tell in (False, True):
                with self.subTest(max_iter=max_iter, ask_tell=ask_tell):
                    lengths, reasons = self.run_lengths(max_iter, ask_tell)
                    self.assertGreater(len(lengths), 2)
                    self.assertTrue(all(n == max_iter for n in lengths[:-1]), lengths)
                    self.assertLessEqual(lengths[-1], max_iter)  # the budget ends the last
                    self.assertTrue(all(r == ["ExceededMaxIter"] for r in reasons[:-1]), reasons)

    def test_run_and_ask_tell_agree(self):
        for max_iter in (1, 4):
            with self.subTest(max_iter=max_iter):
                np.testing.assert_array_equal(
                    self.run_lengths(max_iter, False)[0], self.run_lengths(max_iter, True)[0]
                )

    def test_max_iter_zero_still_runs_a_generation(self):
        lengths, _ = self.run_lengths(0, False)
        self.assertTrue(all(n == 1 for n in lengths), lengths)

    def test_d_sigma_relative_to_the_start_of_the_run(self):
        # A run that started at sigma = 1e-3 with Settings.sigma0 = 1 (as a small run of BIPOP):
        # sigma / sigma_start against tolerance * D, not sigma / Settings.sigma0
        es = optimizer("COVARIANCE", sigma0=1.0)
        items = criteria(es)
        es.p.mutation.sigma = 1e-3
        for name in ("MaxDSigma", "MinDSigma"):
            items[name].reset(es.p)
            self.assertEqual(items[name].sigma_start, 1e-3)
        upper = c_maes.restart.MaxDSigma.tolerance
        lower = c_maes.restart.MinDSigma.tolerance
        # D = 1 (C = I): met when sigma / 1e-3 is above upper, or below lower
        for sigma, max_met, min_met in (
            (1e-3 * upper * 10, True, False),  # sigma / sigma0 = 1e-2 upper: not met before
            (1e-3 * upper / 10, False, False),
            (1e-3 * lower * 10, False, False),  # sigma / sigma0 = 1e-2 lower: met before
            (1e-3 * lower / 10, False, True),
        ):
            with self.subTest(sigma=sigma):
                es.p.mutation.sigma = sigma
                items["MaxDSigma"].update(es.p)
                items["MinDSigma"].update(es.p)
                self.assertEqual(items["MaxDSigma"].met, max_met)
                self.assertEqual(items["MinDSigma"].met, min_met)

    def test_d_sigma_added_later_takes_the_current_sigma(self):
        es = optimizer("COVARIANCE", sigma0=1.0)
        es.step(sphere)
        for cls in (c_maes.restart.MaxDSigma, c_maes.restart.MinDSigma):
            item = cls()
            es.p.criteria.items.append(item)
            es.step(sphere)
            self.assertGreater(item.sigma_start, 0)
            self.assertNotEqual(item.sigma_start, 1.0)  # sigma after the first generations


def pycma_mu(popsize):
    """pycma 4.6.0's default mu of a population: RecombinationWeights(popsize).mu, the number of
    positive weights ln((popsize + 1) / 2) - ln(i), i = 1..popsize (recombination_weights.py), so
    popsize // 2. fmin creates every run with the default CMA_mu (evolution_strategy.py:5411-5413)."""
    return sum(math.log((popsize + 1) / 2) - math.log(i) > 0 for i in range(1, popsize + 1))


class TestMuAfterARestart(unittest.TestCase):
    """#150: mu after an IPOP or BIPOP restart is the default of the new lambda (pycma's
    popsize // 2), or the ratio of a given mu0, rounded the same way by both strategies. It kept
    mu0 / lambda0, rounded by IPOP and truncated by BIPOP (d = 3: lambda 14 gave mu 6)."""

    @staticmethod
    def restarts(strategy, d, n=8, **kwargs):
        es = optimizer("COVARIANCE", d=d, restart_strategy=strategy, budget=10**9, **kwargs)
        pairs = []
        for _ in range(n):
            es.p.stats.evaluations += 997
            es.p.restart_strategy.update(es.p)
            pairs.append((es.p.mu, es.p.lamb))
        return es, pairs

    def test_the_default_mu_is_pycmas(self):
        self.assertEqual([pycma_mu(n) for n in range(2, 40)], [n // 2 for n in range(2, 40)])
        for strategy in ("IPOP", "BIPOP"):
            for d in (2, 3, 5, 10, 20):  # lambda0 = 7, 9, 13 at d = 3, 10, 20: odd
                with self.subTest(strategy=strategy, d=d):
                    _, pairs = self.restarts(strategy, d)
                    self.assertGreater(len({lam for _, lam in pairs}), 2)
                    for mu, lam in pairs:
                        self.assertEqual(mu, pycma_mu(lam), (mu, lam))

    def test_odd_lambda0(self):
        # The evidence of #150: d = 3, lambda0 = 7, the doubled 14 has mu 7 (pycma), not 6
        _, pairs = self.restarts("IPOP", 3, n=1)
        self.assertEqual(pairs, [(7, 14)])

    def test_a_quarter_with_sa_and_cmsa(self):
        for modules in ({"ssa": "SA"}, {"matrix_adaptation": "CMSA"}):
            for strategy in ("IPOP", "BIPOP"):
                with self.subTest(strategy=strategy, **modules):
                    kwargs = dict(modules)
                    adaptation = kwargs.pop("matrix_adaptation", "COVARIANCE")
                    es = optimizer(
                        adaptation, d=3, restart_strategy=strategy, budget=10**9, **kwargs
                    )
                    for _ in range(6):
                        es.p.stats.evaluations += 997
                        es.p.restart_strategy.update(es.p)
                        self.assertEqual(es.p.mu, max(1, es.p.lamb // 4))

    def test_a_given_mu0_keeps_its_ratio(self):
        # mu0 = 2 of lambda0 = 7 (d = 3): mu0 lambda / lambda0, rounded half up, in both strategies
        for strategy in ("IPOP", "BIPOP"):
            with self.subTest(strategy=strategy):
                _, pairs = self.restarts(strategy, 3, mu0=2)
                for mu, lam in pairs:
                    self.assertEqual(mu, max(1, math.floor(2 * lam / 7 + 0.5)), (mu, lam))

    def test_both_strategies_agree(self):
        # The same lambda gives the same mu in IPOP and BIPOP, with and without mu0
        for kwargs in ({}, {"mu0": 3}, {"mu0": 5}):
            with self.subTest(**kwargs):
                seen = {}
                for strategy in ("IPOP", "BIPOP"):
                    for mu, lam in self.restarts(strategy, 3, n=10, **kwargs)[1]:
                        self.assertEqual(seen.setdefault(lam, mu), mu, (strategy, lam))
                self.assertGreater(len(seen), 3)

    def test_mu_of_a_restart_capped_at_the_budget(self):
        # The first generation after a restart near the end of the budget takes the rest of it,
        # with mu of the capped lambda by the same rule
        es = optimizer("COVARIANCE", d=10, restart_strategy="IPOP", budget=400)
        es.p.criteria.items.append(RestartEvery(3))
        lambdas = set()
        while es.step(sphere):
            lambdas.add(es.p.lamb)
            self.assertEqual(es.p.mu, pycma_mu(es.p.lamb), es.p.lamb)
        self.assertGreater(len(lambdas), 1)


class EndLargeRuns(c_maes.restart.Criterion):
    """Met after 5 generations of the first run and of the large runs of BIPOP."""

    def __init__(self):
        super().__init__("EndLargeRuns")

    def update(self, parameters):
        strategy = parameters.restart_strategy
        large = strategy.large_run or strategy.lambda_large == 0
        self.met = large and parameters.stats.t - self.last_restart >= 4


if __name__ == "__main__":
    unittest.main()
