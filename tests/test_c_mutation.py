"""Tests of the step-size adaptations (mutation) of the C++ ModularCMA-ES."""

import unittest
import numpy as np
from modcma.c_maes import (
    mutation,
    Population,
    Parameters,
    parameters,
    options,
    ModularCMAES,
)


class TestMutation(unittest.TestCase):
    def setUp(self):

        pop = Population(2, 2)
        pop.Z = np.ones((2, 2)) * 0.5
        self.par = Parameters(2)
        self.par.pop = pop
        self.par.weights.beta = 1

    def test_sigma_sampler(self):
        ss = mutation.SigmaSampler(2)
        noss = mutation.NoSigmaSampler(2)

        ss.sample(2.0, self.par)
        S = self.par.pop.S
        self.assertFalse(np.all(S == 2.0))
        # One step size per individual, the same in every coordinate (#71, F1)
        np.testing.assert_array_equal(S, np.broadcast_to(S[0], S.shape))
        self.assertNotEqual(S[0, 0], S[0, 1])
        noss.sample(2.0, self.par)
        self.assertTrue(np.all(self.par.pop.S == 2.0))

    def test_threshold_convergence(self):
        tc = mutation.ThresholdConvergence()
        notc = mutation.NoThresholdConvergence()
        budget, evals, length = 100, 2, 1.5
        t = tc.threshold(length, budget, evals)
        self.assertAlmostEqual(t, tc.init_threshold * length * ((budget - evals) / budget) ** tc.decay_factor)

        z, y = np.array([0.3, 0.4]), np.array([3.0, 4.0])  # ||z|| = 0.5 < t
        notc.apply(z, y, t)
        np.testing.assert_array_equal(z, [0.3, 0.4])
        tc.apply(z, y, t)
        factor = (2 * t - 0.5) / 0.5
        np.testing.assert_allclose(z, np.array([0.3, 0.4]) * factor)
        np.testing.assert_allclose(y, np.array([3.0, 4.0]) * factor)
        self.assertAlmostEqual(np.linalg.norm(z), 2 * t - 0.5)

    def test_sequential_selection_cutoff_factor_is_checked(self):
        # A factor below 1 could stop a population after mu or fewer individuals (the updates then
        # read columns of an earlier generation); NaN and inf have no cutoff (#191)
        for mirror in (options.Mirror.NONE, options.Mirror.PAIRWISE):
            for factor in (0.5, 0.0, -1.0, float("nan"), float("inf")):
                for cls in (mutation.SequentialSelection, mutation.NoSequentialSelection):
                    with self.subTest(mirror=mirror, factor=factor, cls=cls.__name__):
                        with self.assertRaisesRegex(ValueError, "seq_cutoff_factor"):
                            cls(mirror, factor)
        mutation.SequentialSelection(options.Mirror.NONE, 1.0)
        mutation.SequentialSelection(options.Mirror.PAIRWISE, 1.5)

    def get_cma(self, ssa, adapt_sigma=True):
        modules = parameters.Modules()
        modules.ssa = ssa
        settings = parameters.Settings(2, modules)
        par = Parameters(settings)

        cma = ModularCMAES(par)
        if adapt_sigma:
            cma.mutate(sum)
            cma.select()
            cma.recombine()
            cma.p.adaptation.adapt_evolution_paths(
                cma.p.pop,
                cma.p.weights,
                cma.p.stats,
                cma.p.settings,
                cma.p.mu,
            )
            cma.p.mutation.adapt(
                cma.p.weights,
                cma.p.adaptation,
                cma.p.pop,
                cma.p.old_pop,
                cma.p.stats,
            )
        return cma

    def test_adapt_csa(self):
        cma = self.get_cma(options.StepSizeAdaptation.CSA)

        self.assertAlmostEqual(
            cma.p.mutation.sigma,
            cma.p.settings.sigma0
            * np.exp(
                (cma.p.weights.cs / cma.p.weights.damps)
                * (
                    (
                        np.linalg.norm(cma.p.adaptation.ps)
                        / cma.p.sampler.expected_length()
                    )
                    - 1
                )
            ),
        )

    def test_adapt_tpa(self):
        # In the first generation dm = 0: both TPA points are the mean, a tie, which gives no
        # signal (#78, F15; it counted as a success and sigma grew)
        cma = self.get_cma(options.StepSizeAdaptation.TPA)
        self.assertEqual(cma.p.mutation.rank_tpa, 0.0)
        self.assertAlmostEqual(cma.p.mutation.sigma, cma.p.settings.sigma0)

    def test_tpa_signal(self):
        # Once the mean moves, one of the two points is better: r is -a_tpa or a_tpa + b_tpa
        from modcma import c_maes

        es = c_maes.ModularCMAES(c_maes.settings_from_dict(3, ssa="TPA", seed=1, verbose=False))
        for _ in range(5):
            es.step(lambda x: float(x[0]))  # linear: the forward point is better
        tpa = es.p.mutation
        self.assertEqual(tpa.rank_tpa, tpa.a_tpa + tpa.b_tpa)

    def test_msr_psr_mxnes_grow_sigma_on_a_linear_function(self):
        # MSR and PSR compare with the previous generation, so they adapt from the second one; on
        # a linear function every step-size rule should increase sigma
        from modcma import c_maes

        for ssa in ("MSR", "PSR", "MXNES"):
            with self.subTest(ssa=ssa):
                settings = c_maes.settings_from_dict(5, ssa=ssa, seed=1, verbose=False)
                es = c_maes.ModularCMAES(settings)
                for _ in range(10):
                    es.step(lambda x: float(x[0]))
                self.assertGreater(es.p.mutation.sigma, 2 * es.p.settings.sigma0)

    def test_psr_ties_do_not_depend_on_the_previous_generation(self):
        # #120: the ranking started from the previous call's order, so tied values were ordered
        # by history. Now equal values share their mean rank (mid-ranks).
        es = ModularCMAES(parameters.Settings(2, lambda0=4, mu0=2))
        p = es.p

        def population(f):
            pop = Population(2, 4)
            pop.f = np.array(f, dtype=float)
            return pop

        def s_after(psr, new, old):
            psr.s, psr.sigma = 0.0, 1.0
            psr.adapt(p.weights, p.adaptation, population(new), population(old), p.stats)
            return psr.s

        def make():
            return mutation.PSR(
                mutation.NoThresholdConvergence(),
                mutation.NoSequentialSelection(options.Mirror.NONE),
                mutation.NoSigmaSampler(2),
                1.0,
            )

        tied = ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        fresh = s_after(make(), *tied)
        used = make()
        s_after(used, [1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0])  # interleaved ranks
        self.assertEqual(s_after(used, *tied), fresh)
        # all eight values tie: every rank is 3.5, the difference of the rank sums is 0
        cs = p.weights.cs
        self.assertAlmostEqual(fresh, cs * (0 - 0.25))
        # the tied parents (1 and 2, in both populations) count for neither: ranks 0.5, 2.5, 4, 5
        # against 0.5, 2.5, 6, 7, a difference of 4, and z = 4 / 16 - 0.25 = 0
        self.assertAlmostEqual(s_after(make(), [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 5.0, 6.0]), 0.0)

    def test_adapt_xnes(self):
        cma = self.get_cma(options.StepSizeAdaptation.XNES)

        w = cma.p.weights.weights.clip(0)[: cma.p.pop.n]
        z = np.power(np.linalg.norm(cma.p.pop.Z, axis=0), 2) - cma.p.settings.dim
        self.assertTrue(
            np.isclose(
                cma.p.mutation.sigma,
                cma.p.settings.sigma0
                * np.exp(
                    (cma.p.weights.cs / np.sqrt(cma.p.settings.dim)) * (w * z).sum()
                ),
            )
        )

    def test_adapt_lpxnes(self):
        cma = self.get_cma(options.StepSizeAdaptation.LPXNES)

        w = cma.p.weights.weights.clip(0)[: cma.p.pop.n]

        z = np.exp(cma.p.weights.cs * (w @ np.log(cma.p.pop.S).mean(axis=0)))

        sigma = np.power(cma.p.settings.sigma0, 1 - cma.p.weights.cs) * z
        self.assertTrue(np.isclose(cma.p.mutation.sigma, sigma))


if __name__ == "__main__":
    unittest.main()
