"""Module containing tests for ModularCMA-ES C++ Bounds."""

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
from modcma.modularcmaes import scale_with_threshold


class TestMutation(unittest.TestCase):
    def setUp(self):
        self.pop = Population(2, 2)
        self.pop.Z = np.ones((2, 2)) * 0.5

    def test_sigma_sampler(self):
        ss = mutation.SigmaSampler(2)
        noss = mutation.NoSigmaSampler(2)

        ss.sample(2.0, self.pop, 1)
        self.assertFalse(np.all(self.pop.s == 2.0))
        noss.sample(2.0, self.pop, 1)
        self.assertTrue(np.all(self.pop.s == 2.0))

    def test_threshold_convergence(self):
        tc = mutation.ThresholdConvergence()
        notc = mutation.NoThresholdConvergence()
        notc.scale(self.pop.Z[0], 10, 100, 2)
        self.assertTrue(np.all(self.pop.Z == 0.5))

        budget = 100
        evals = 2
        diam = 10

        t = tc.init_threshold * diam * pow((budget - evals) / budget, tc.decay_factor)
        norm = np.linalg.norm(self.pop.Z, axis=0)
        Z = np.ones((2, 2))
        Z[0] = tc.scale(self.pop.Z[0], diam, budget, evals)
        Z[1] = tc.scale(self.pop.Z[1], diam, budget, evals)
        self.pop.Z = Z
        expected_z = 0.5 * ((t + (t - norm)) / norm)
        self.assertTrue(np.isclose(self.pop.Z, expected_z).all())
        self.assertTrue(np.isclose(self.pop.Z, scale_with_threshold(np.ones(2) * .5, t)).all())

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
                cma.p.pop, cma.p.weights, cma.p.stats, 
                cma.p.settings, cma.p.mu, cma.p.lamb
            )
            cma.p.mutation.adapt(
                cma.p.weights, cma.p.adaptation, cma.p.pop, cma.p.old_pop, cma.p.stats, cma.p.lamb
            )
        return cma

    def test_adapt_csa(self):
        cma = self.get_cma(options.CSA)

        self.assertAlmostEqual(
            cma.p.mutation.sigma,
            cma.p.settings.sigma0
            * np.exp(
                (cma.p.weights.cs / cma.p.weights.damps)
                * ((np.linalg.norm(cma.p.adaptation.ps) / cma.p.sampler.expected_length()) - 1)
            ),
        )

    def test_adapt_tpa(self):
        cma = self.get_cma(options.TPA)
        s = ((1 - cma.p.weights.cs) * 0) + (cma.p.weights.cs * cma.p.mutation.a_tpa)
        self.assertAlmostEqual(cma.p.mutation.sigma, cma.p.settings.sigma0 * np.exp(s))

    def test_adapt_msr(self):
        cma = self.get_cma(options.MSR)

    def test_adapt_psr(self):
        cma = self.get_cma(options.PSR)
        
    def test_adapt_mxnes(self):
        cma = self.get_cma(options.MXNES)


    def test_adapt_xnes(self):
        cma = self.get_cma(options.XNES)

        w = cma.p.weights.weights.clip(0)[: cma.p.pop.n]
        z = (
            np.power(
                np.linalg.norm(cma.p.pop.Z, axis=0), 2
            )
            - cma.p.settings.dim
        )
        self.assertTrue(np.isclose(
            cma.p.mutation.sigma,
            cma.p.settings.sigma0
            * np.exp((cma.p.weights.cs / np.sqrt(cma.p.settings.dim)) * (w * z).sum()),
        ))


    def test_adapt_lpxnes(self):
        cma = self.get_cma(options.LPXNES)

        w = cma.p.weights.weights.clip(0)[: cma.p.pop.n]
        
        z = np.exp(cma.p.weights.cs * (w @ np.log(cma.p.pop.s)))
        sigma = np.power(cma.p.settings.sigma0, 1 - cma.p.weights.cs) * z
        self.assertTrue(np.isclose(cma.p.mutation.sigma, sigma))

if __name__ == "__main__":
    unittest.main()
