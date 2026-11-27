"""The presets (#92): published algorithms as a selection of modules."""

import unittest

import numpy as np

import modcma
from modcma import Modules, Settings, fmin, options, preset, preset_names


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


class TestPresets(unittest.TestCase):
    def test_names(self):
        self.assertEqual(
            preset_names(),
            (
                "cma-es",
                "active-cma-es",
                "sep-cma",
                "ma-es",
                "cholesky-cma",
                "cmsa-es",
                "xnes",
                "ipop",
                "bipop",
                "(1+1)-cma-es",
            ),
        )

    def test_modules(self):
        m = lambda name: Settings.preset(name, 5).modules
        self.assertEqual(m("cma-es").matrix_adaptation, options.MatrixAdaptationType.COVARIANCE)
        self.assertTrue(m("active-cma-es").active)
        self.assertEqual(m("sep-cma").matrix_adaptation, options.MatrixAdaptationType.SEPARABLE)
        self.assertEqual(m("ma-es").matrix_adaptation, options.MatrixAdaptationType.MATRIX)
        self.assertEqual(m("cholesky-cma").matrix_adaptation, options.MatrixAdaptationType.CHOLESKY)
        self.assertEqual(m("cmsa-es").matrix_adaptation, options.MatrixAdaptationType.CMSA)
        self.assertEqual(m("cmsa-es").ssa, options.StepSizeAdaptation.SA)
        x = m("xnes")
        self.assertEqual(x.matrix_adaptation, options.MatrixAdaptationType.NATURAL_GRADIENT)
        self.assertEqual(x.ssa, options.StepSizeAdaptation.XNES)
        self.assertEqual(x.weights, options.RecombinationWeights.UTILITY)
        self.assertEqual(m("ipop").restart_strategy, options.RestartStrategy.IPOP)
        self.assertEqual(m("bipop").restart_strategy, options.RestartStrategy.BIPOP)
        s = Settings.preset("(1+1)-cma-es", 5)
        self.assertEqual(s.lambda0, 1)
        self.assertTrue(s.modules.elitist)
        self.assertEqual(s.modules.ssa, options.StepSizeAdaptation.SR)

    def test_default_is_modules_default(self):
        a = preset("cma-es", 6, seed=1)
        b = Settings(6, Modules(), seed=1)
        self.assertEqual(str(a.modules), str(b.modules))
        self.assertEqual(a.lambda0, b.lambda0)

    def test_name_matching(self):
        for name in ("Sep_CMA", " SEP-CMA "):
            self.assertEqual(
                preset(name, 4).modules.matrix_adaptation, options.MatrixAdaptationType.SEPARABLE
            )

    def test_unknown(self):
        with self.assertRaises(ValueError) as cm:
            preset("nope", 5)
        for name in preset_names():
            self.assertIn(name, str(cm.exception))
        with self.assertRaises(ValueError):
            fmin(sphere, np.ones(3), 0.5, 500, preset="nope")

    def test_overrides(self):
        s = preset("sep-cma", 5, active=True, budget=777, lambda0=12, matrix_adaptation="MATRIX")
        self.assertTrue(s.modules.active)
        self.assertEqual(s.budget, 777)
        self.assertEqual(s.lambda0, 12)
        self.assertEqual(s.modules.matrix_adaptation, options.MatrixAdaptationType.MATRIX)
        s = preset("ipop", 5, restart_strategy="BIPOP")
        self.assertEqual(s.modules.restart_strategy, options.RestartStrategy.BIPOP)

    def test_fmin(self):
        for name in preset_names():
            with self.subTest(name=name):
                xopt, fopt, evals, es = fmin(
                    sphere, np.full(4, 1.0), 0.5, 4000, preset=name, seed=3
                )
                self.assertEqual(len(xopt), 4)
                self.assertTrue(np.isfinite(fopt))
                self.assertLessEqual(evals, 4000)
                if name != "(1+1)-cma-es":
                    self.assertLess(fopt, 1.0)

    def test_fmin_override(self):
        _, _, _, es = fmin(sphere, np.ones(4), 0.5, 2000, preset="sep-cma", active=True, seed=1)
        self.assertTrue(es.p.settings.modules.active)

    def test_ask_tell(self):
        for name in preset_names():
            with self.subTest(name=name):
                es = modcma.ModularCMAES(preset(name, 4, x0=np.ones(4), sigma0=0.5, budget=800, seed=2))
                for _ in range(5):
                    X = es.ask()
                    es.tell(np.array([sphere(x) for x in X]))
                self.assertGreater(es.p.stats.evaluations, 0)

    def test_ask_tell_equals_run(self):
        for name in ("sep-cma", "cmsa-es", "xnes", "(1+1)-cma-es"):
            with self.subTest(name=name):
                kw = dict(x0=np.ones(4), sigma0=0.5, budget=600, seed=5)
                a = modcma.ModularCMAES(preset(name, 4, **kw))
                a(sphere)
                b = modcma.ModularCMAES(preset(name, 4, **kw))
                while not b.break_conditions():
                    X = b.ask()
                    b.tell(np.array([sphere(x) for x in X]))
                self.assertEqual(a.p.stats.global_best.y, b.p.stats.global_best.y)


if __name__ == "__main__":
    unittest.main()
