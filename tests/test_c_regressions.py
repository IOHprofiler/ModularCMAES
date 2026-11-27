"""Regression tests for the bugs fixed in WP0 (dev/overhaul-2.0.md, section 1 and T0.3)."""

import unittest

import numpy as np

from modcma import c_maes


def sphere(x):
    return float(np.dot(x, x))


class TestIntegerMapping(unittest.TestCase):
    """B1: integer variables were mapped by loop index instead of by the given index."""

    def test_noncontiguous_integer_indices(self):
        c_maes.set_seed(1)
        settings = c_maes.Settings(5, integer_variables=[1, 3], verbose=False)
        cma = c_maes.ModularCMAES(settings)
        cma.step(sphere)
        X, Xt = cma.p.pop.X, cma.p.pop.X_transformed
        for i in (1, 3):
            np.testing.assert_array_equal(Xt[i], np.round(X[i]))
        for i in (0, 2, 4):
            np.testing.assert_array_equal(Xt[i], X[i])
        # the continuous coordinates must actually be continuous in this sample
        self.assertFalse(np.all(Xt[[0, 2, 4]] == np.round(Xt[[0, 2, 4]])))

    def test_out_of_range_index_raises(self):
        settings = c_maes.Settings(3, integer_variables=[3], verbose=False)
        with self.assertRaises(ValueError):
            c_maes.ModularCMAES(settings)


class TestActiveCovarianceUpdate(unittest.TestCase):
    """B2: the negative weights of active CMA were never rescaled by d / ||C^-1/2 y||^2 when
    lambda = 2 mu (the default), because the loop ran over [mu, lambda - mu). Once it ran, the
    rescaling also had to use invert_y: COVARIANCE_NO_EIGV never computes inv_root_C."""

    def test_rank_mu_update_matches_reference(self):
        for name in ("COVARIANCE", "COVARIANCE_NO_EIGV"):
            with self.subTest(matrix_adaptation=name):
                self.check_rank_mu_update(name)

    def check_rank_mu_update(self, name):
        c_maes.set_seed(3)
        modules = c_maes.Modules()
        modules.active = True
        modules.matrix_adaptation = getattr(c_maes.options.MatrixAdaptationType, name)
        d = 5
        settings = c_maes.Settings(d, modules, lambda0=10, mu0=5, verbose=False)
        cma = c_maes.ModularCMAES(settings)
        for _ in range(3):  # the first generation does not update C; then C moves away from I
            cma.step(sphere)

        cma.mutate(sphere)
        cma.select()
        cma.recombine()
        p, a = cma.p, cma.p.adaptation
        a.adapt_evolution_paths(p.pop, p.weights, p.stats, p.settings, p.mu)
        C0, pc = a.C.copy(), a.pc.copy()
        self.assertGreater(np.abs(C0 - np.diag(np.diag(C0))).max(), 1e-3)
        Y, hs = p.pop.Y.copy(), a.hs
        w = np.array(p.weights.weights[: Y.shape[1]])
        c1, cmu, cc = p.weights.c1, p.weights.cmu, p.weights.cc
        self.assertTrue(a.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats))

        # ||C^-1/2 y||^2 = y^T C^-1 y, with C the matrix before this update
        rank_mu_w = w.copy()
        for i in range(p.mu, len(w)):
            rank_mu_w[i] *= d / (Y[:, i] @ np.linalg.solve(C0, Y[:, i]))
        dhs = (1 - hs) * cc * (2 - cc)
        decay = 1 - c1 * dhs - c1 - cmu * w.sum()
        expected = decay * C0 + c1 * np.outer(pc, pc) + cmu * (Y * rank_mu_w) @ Y.T
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(a.C, expected, rtol=1e-10, atol=1e-12)
        # and the rescaling made a difference for this sample
        unscaled = decay * C0 + c1 * np.outer(pc, pc) + cmu * (Y * w) @ Y.T
        self.assertGreater(np.abs(a.C - 0.5 * (unscaled + unscaled.T)).max(), 1e-6)


class TestNaturalGradientInverse(unittest.TestCase):
    """B3: NaturalGradientAdaptation::invert_y recomputed the pseudo-inverse on every call."""

    def test_inverse_is_cached_and_correct(self):
        c_maes.set_seed(1)
        modules = c_maes.Modules()
        modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.NATURAL_GRADIENT
        cma = c_maes.ModularCMAES(c_maes.Settings(4, modules, verbose=False))
        for _ in range(5):
            cma.step(sphere)
        a = cma.p.adaptation
        y = np.arange(1.0, 5.0)
        z = a.invert_y(y)
        np.testing.assert_allclose(np.array(a.A) @ z, y, rtol=1e-10)
        # after the first call, the cached inverse is used: changing A without a matrix update
        # must not change the result
        a.A = np.array(a.A) * 2.0
        np.testing.assert_array_equal(a.invert_y(y), z)


class TestNoSharedSamplerState(unittest.TestCase):
    """B7: rng::normal and the Box-Muller transform kept their state in function statics, so one
    sampler object (or optimizer) changed the numbers of the next one."""

    def test_normal_does_not_leak_between_objects(self):
        c_maes.set_seed(1)
        fresh = c_maes.sampling.Gaussian(4)()
        c_maes.sampling.Gaussian(3)()  # an odd number of draws leaves half a Box-Muller pair
        c_maes.set_seed(1)
        np.testing.assert_array_equal(c_maes.sampling.Gaussian(4)(), fresh)

    def test_box_muller_does_not_leak_between_objects(self):
        default = c_maes.constants.use_box_muller
        c_maes.constants.use_box_muller = True
        try:

            def sample(d):
                return c_maes.sampling.GaussianTransformer(c_maes.sampling.Uniform(d))()

            c_maes.set_seed(1)
            fresh = sample(3)
            sample(3)  # odd dimension: keeps an extra uniform and half a pair
            c_maes.set_seed(1)
            np.testing.assert_array_equal(sample(3), fresh)
        finally:
            c_maes.constants.use_box_muller = default


def rastrigin(x):
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


class TestSequentialSelectionPopulation(unittest.TestCase):
    """B13: when sequential selection stopped a generation at individual i, the unevaluated
    columns stayed in the population: individuals of earlier generations with their old f, and
    uninitialized memory in the first generation. Now the population is 0..i (pop.n), with
    pairwise mirroring only complete pairs, and the weights are renormalised for n < lambda."""

    def run_until_stop(self, mirrored="NONE", active=False, generations=200):
        c_maes.set_seed(1)
        modules = c_maes.Modules()
        modules.sequential_selection = True
        modules.active = active
        modules.mirrored = getattr(c_maes.options.Mirror, mirrored)
        cma = c_maes.ModularCMAES(c_maes.Settings(5, modules, lambda0=12, mu0=3, verbose=False))
        generation = [0]
        evaluated = {}

        def objective(x):
            y = float(x @ x)
            evaluated[y] = generation[0]
            return y

        for g in range(1, generations):
            generation[0] = g
            before = cma.p.stats.evaluations
            cma.mutate(objective)
            n = cma.p.stats.evaluations - before
            if n < cma.p.lamb:
                yield g, n, cma, evaluated
            cma.select()
            cma.recombine()
            cma.adapt()

    def test_population_holds_only_this_generation(self):
        stops = 0
        for g, n, cma, evaluated in self.run_until_stop():
            pop = cma.p.pop
            self.assertEqual(pop.n, n)
            self.assertEqual(pop.f.shape, (n,))
            self.assertEqual(pop.X.shape, (5, n))
            self.assertEqual(pop.capacity, cma.p.lamb)
            self.assertEqual({evaluated.get(v) for v in pop.f}, {g})
            stops += 1
            if stops == 5:
                break
        self.assertEqual(stops, 5)

    def test_pairwise_stops_after_complete_pairs(self):
        stops = 0
        for g, n, cma, _ in self.run_until_stop(mirrored="PAIRWISE"):
            self.assertEqual(n % 2, 0, f"generation {g} stopped after {n} individuals")
            cma.select()  # pairwise selection needs complete pairs
            stops += 1
            if stops == 5:
                break
        self.assertEqual(stops, 5)

    def test_weights_for_population(self):
        modules = c_maes.Modules()
        modules.active = True
        w = c_maes.Parameters(c_maes.Settings(5, modules, lambda0=12, mu0=3, verbose=False)).weights
        full = np.array(w.weights)
        np.testing.assert_array_equal(w.for_population(12), full)
        for n in (4, 7, 11):
            part = np.array(w.for_population(n))
            self.assertEqual(len(part), n)
            np.testing.assert_array_equal(part[:3], full[:3])
            self.assertAlmostEqual(part[3:].sum(), full[3:].sum(), places=12)
            np.testing.assert_allclose(part[3:] / full[3:n], part[3] / full[3])

    def test_active_update_uses_renormalised_weights(self):
        for g, n, cma, _ in self.run_until_stop(active=True):
            if g < 3:
                continue
            cma.select()
            cma.recombine()
            p, a = cma.p, cma.p.adaptation
            self.assertEqual(p.pop.n, n)
            a.adapt_evolution_paths(p.pop, p.weights, p.stats, p.settings, p.mu)
            C0, pc, hs = a.C.copy(), a.pc.copy(), a.hs
            Y = p.pop.Y.copy()
            w = np.array(p.weights.for_population(n))
            c1, cmu, cc = p.weights.c1, p.weights.cmu, p.weights.cc
            p.stats.t = 10**6  # past the lazy-update interval
            self.assertTrue(a.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats))
            rank_mu_w = w.copy()
            for i in range(p.mu, n):
                rank_mu_w[i] *= 5 / (Y[:, i] @ np.linalg.solve(C0, Y[:, i]))
            dhs = (1 - hs) * cc * (2 - cc)
            decay = 1 - c1 * dhs - c1 - cmu * w.sum()
            expected = decay * C0 + c1 * np.outer(pc, pc) + cmu * (Y * rank_mu_w) @ Y.T
            np.testing.assert_allclose(a.C, 0.5 * (expected + expected.T), rtol=1e-10, atol=1e-12)
            return
        self.fail("no sequential selection stop")


class TestRepellingMetric(unittest.TestCase):
    """B4: taboo points used a never-updated identity matrix (Euclidean distance). The metric is
    now an explicit option: EUCLIDEAN (default, the behaviour since July 2026), CURRENT (the
    current search distribution, as before July 2026) and ARCHIVED (the shape of the basin's
    own run)."""

    def make(self, metric):
        c_maes.set_seed(5)
        modules = c_maes.Modules()
        modules.repelling_type = c_maes.options.RepellingType.COVERAGE
        modules.restart_strategy = c_maes.options.RestartStrategy.RESTART
        modules.repelling_metric = getattr(c_maes.options.RepellingMetric, metric)
        cma = c_maes.ModularCMAES(c_maes.Settings(3, modules, verbose=False))
        for _ in range(30):  # long enough for C to be far from the identity
            cma.step(rastrigin)
        return cma

    def taboo(self, cma, metric=None):
        solution = c_maes.parameters.Solution()
        solution.x = np.array([0.5, -1.0, 2.0])
        solution.y = 1.0
        if metric is None:
            return c_maes.repelling.TabooPoint(solution, 1.0)
        return c_maes.repelling.TabooPoint(solution, 1.0, metric)

    def test_default_is_euclidean(self):
        self.assertEqual(
            c_maes.Modules().repelling_metric, c_maes.options.RepellingMetric.EUCLIDEAN
        )

    def test_euclidean(self):
        cma = self.make("EUCLIDEAN")
        point, x = self.taboo(cma), np.array([1.0, 2.0, 3.0])
        expected = np.linalg.norm(x - point.solution.x) / cma.p.mutation.sigma
        self.assertAlmostEqual(point.distance(x, cma.p), expected, places=12)
        self.assertEqual(c_maes.repelling.archived_metric(cma.p).size, 0)

    def test_current(self):
        cma = self.make("CURRENT")
        point, x = self.taboo(cma), np.array([1.0, 2.0, 3.0])
        A = np.array(cma.p.adaptation.A)
        self.assertGreater(np.abs(A - np.eye(3) * A[0, 0]).max(), 1e-3)  # not isotropic
        expected = np.linalg.norm(np.linalg.solve(A, x - point.solution.x)) / cma.p.mutation.sigma
        self.assertAlmostEqual(point.distance(x, cma.p), expected, places=10)

    def test_archived(self):
        cma = self.make("ARCHIVED")
        metric = c_maes.repelling.archived_metric(cma.p)
        A = np.array(cma.p.adaptation.A)
        self.assertAlmostEqual(abs(np.linalg.det(metric)), 1.0, places=10)
        product = metric @ A  # proportional to the identity: metric = A^-1 / |det A^-1|^(1/d)
        np.testing.assert_allclose(product, np.eye(3) * product[0, 0], atol=1e-10)

        point, x = self.taboo(cma, metric), np.array([1.0, 2.0, 3.0])
        expected = np.linalg.norm(metric @ (x - point.solution.x)) / cma.p.mutation.sigma
        self.assertAlmostEqual(point.distance(x, cma.p), expected, places=12)
        # frozen: later adaptation does not change the stored metric
        for _ in range(5):
            cma.step(rastrigin)
        self.assertAlmostEqual(
            point.distance(x, cma.p),
            np.linalg.norm(metric @ (x - point.solution.x)) / cma.p.mutation.sigma,
            places=12,
        )

    def test_archive_stores_metric_only_for_archived(self):
        for metric, stored in (("EUCLIDEAN", False), ("CURRENT", False), ("ARCHIVED", True)):
            cma = self.make(metric)
            while not cma.p.repelling.archive and cma.step(rastrigin):
                pass
            self.assertTrue(cma.p.repelling.archive, metric)
            self.assertEqual(cma.p.repelling.archive[0].metric.size > 0, stored, metric)


class TestNonFiniteMatrixRestarts(unittest.TestCase):
    """B11: a NaN/inf matrix was reported as a successful adaptation, so the run continued with
    NaN samples. Now it is unsuccessful, which triggers the UnableToAdapt restart criterion."""

    def test_non_finite_matrix_is_unsuccessful(self):
        for name in ("COVARIANCE", "MATRIX", "SEPARABLE", "CHOLESKY", "CMSA",
                     "COVARIANCE_NO_EIGV", "NATURAL_GRADIENT"):
            c_maes.set_seed(1)
            modules = c_maes.Modules()
            modules.matrix_adaptation = getattr(c_maes.options.MatrixAdaptationType, name)
            cma = c_maes.ModularCMAES(c_maes.Settings(4, modules, verbose=False))
            cma.step(sphere)
            cma.mutate(sphere)
            cma.select()
            cma.recombine()
            p = cma.p
            p.pop.Y = np.full_like(np.array(p.pop.Y), np.nan)
            p.pop.Z = np.full_like(np.array(p.pop.Z), np.nan)
            p.stats.t = 10**6  # past any lazy-update interval
            p.adaptation.adapt_evolution_paths(p.pop, p.weights, p.stats, p.settings, p.mu)
            ok = p.adaptation.adapt_matrix(p.weights, p.pop, p.mu, p.settings, p.stats)
            self.assertFalse(ok, name)


class TestBindingArgumentNames(unittest.TestCase):
    """B12: the keyword names of Adaptation.adapt_evolution_paths were (mu, lamb) for the C++
    order (lambda, mu). lambda is gone (not used); the keywords match the positions."""

    def test_keywords_match_positional(self):
        results = []
        for keywords in (False, True):
            c_maes.set_seed(2)
            cma = c_maes.ModularCMAES(c_maes.Settings(4, lambda0=12, mu0=3, verbose=False))
            cma.mutate(sphere)
            cma.select()
            cma.recombine()
            p = cma.p
            if keywords:
                p.adaptation.adapt_evolution_paths(
                    pop=p.pop, weights=p.weights, stats=p.stats, settings=p.settings, mu=p.mu,
                )
            else:
                p.adaptation.adapt_evolution_paths(p.pop, p.weights, p.stats, p.settings, p.mu)
            results.append(np.array(p.adaptation.dz))
        np.testing.assert_array_equal(results[0], results[1])


class TestCriteriaWithoutEigendecomposition(unittest.TestCase):
    """B14: COVARIANCE_NO_EIGV is a CovarianceAdaptation, but it factorizes C with a Cholesky
    decomposition and does not keep B and d up to date (they are filled only by metric(), #123).
    The criteria that read the eigendecomposition used their initial values (identity, ones); they
    now read metric(), which computes them from the factor (#80).
    Criteria.get adds ConditionC and NoEffectAxis only for the adaptations that keep their metric
    (COVARIANCE, SEPARABLE, NONE), so this matters when they are added by hand."""

    NEEDS_EIGENDECOMPOSITION = ("ConditionC", "NoEffectAxis")

    def criteria_after_ill_conditioned_d(self, name):
        modules = c_maes.Modules()
        modules.matrix_adaptation = getattr(c_maes.options.MatrixAdaptationType, name)
        modules.restart_strategy = c_maes.options.RestartStrategy.RESTART
        cma = c_maes.ModularCMAES(c_maes.Settings(4, modules, seed=1))
        cma.step(sphere)
        p = cma.p
        names = [item.name for item in p.criteria.items]
        if "ConditionC" not in names:
            p.criteria.items.append(c_maes.restart.ConditionC())
            p.criteria.items.append(c_maes.restart.NoEffectAxis())
        # a condition of 1e20 and a vanishing axis, if the criteria read d and B
        p.adaptation.d[:] = [1e10, 1.0, 1.0, 1.0]
        p.adaptation.d[p.stats.t % 4] = 1e-30
        for item in p.criteria.items:
            if item.name in self.NEEDS_EIGENDECOMPOSITION:
                item.update(p)
        return {item.name: item.met for item in p.criteria.items}

    def test_covariance_reads_the_eigendecomposition(self):
        met = self.criteria_after_ill_conditioned_d("COVARIANCE")
        self.assertTrue(met["ConditionC"])
        self.assertTrue(met["NoEffectAxis"])

    def test_no_eigv_does_not_read_it(self):
        met = self.criteria_after_ill_conditioned_d("COVARIANCE_NO_EIGV")
        self.assertFalse(met["ConditionC"])
        self.assertFalse(met["NoEffectAxis"])



class TestCriteriaUseSquareRootOfEigenvalues(unittest.TestCase):
    """B15: d holds the square roots of the eigenvalues of C (pycma's D). MaxDSigma (tolupsigma),
    MinDSigma and NoEffectAxis took its square root again (the fourth root of the eigenvalues)."""

    def setup(self):
        modules = c_maes.Modules()
        modules.restart_strategy = c_maes.options.RestartStrategy.RESTART
        cma = c_maes.ModularCMAES(c_maes.Settings(4, modules, seed=1, sigma0=1.0))
        cma.step(sphere)
        return cma.p, {item.name: item for item in cma.p.criteria.items}

    def test_max_d_sigma(self):
        p, items = self.setup()
        p.adaptation.d[:] = [1e4, 1.0, 1.0, 1.0]
        tolerance = 1e20  # MaxDSigma.tolerance
        p.mutation.sigma = tolerance * 1e3  # above tolerance * sqrt(max D), below tolerance * max D
        items["MaxDSigma"].update(p)
        self.assertFalse(items["MaxDSigma"].met)
        p.mutation.sigma = tolerance * 1e5
        items["MaxDSigma"].update(p)
        self.assertTrue(items["MaxDSigma"].met)

    def test_min_d_sigma(self):
        p, items = self.setup()
        p.adaptation.d[:] = [1e-4, 1.0, 1.0, 1.0]
        tolerance = 1e-20  # MinDSigma.tolerance
        p.mutation.sigma = tolerance * 1e-3  # below tolerance * sqrt(min D), above tolerance * min D
        items["MinDSigma"].update(p)
        self.assertFalse(items["MinDSigma"].met)
        p.mutation.sigma = tolerance * 1e-5
        items["MinDSigma"].update(p)
        self.assertTrue(items["MinDSigma"].met)

    def test_no_effect_axis(self):
        p, items = self.setup()
        t = p.stats.t % 4
        p.adaptation.B[:] = np.eye(4)
        p.mutation.sigma = 1.0
        # 0.1 * D_t = 1e-16 has no effect on m = 1 (below half an ulp, 1.1e-16, #80);
        # 0.1 * sqrt(D_t) = 3e-9 would have
        p.adaptation.m[:] = 1.0
        p.adaptation.d[t] = 1e-15
        items["NoEffectAxis"].update(p)
        self.assertTrue(items["NoEffectAxis"].met)


if __name__ == "__main__":
    unittest.main()
