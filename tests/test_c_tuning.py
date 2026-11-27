import unittest
from copy import deepcopy
from modcma import c_maes

from ConfigSpace import ConfigurationSpace

from tests.golden import module_options, pairwise_configs

NUMERIC = ("cs", "cc", "cmu", "c1", "damps")


def sphere(x):
    return sum(xi**2 for xi in x)


class TestTuning(unittest.TestCase):
    def test_module_options(self):
        options = c_maes.get_all_module_options()
        self.assertIsInstance(options, dict)
        self.assertEqual(options, {**options, "active": (False, True)})
        self.assertEqual(options, {**options, "elitist": (False, True)})
        self.assertEqual(options, {**options, "orthogonal": (False, True)})
        self.assertEqual(options, {**options, "sequential_selection": (False, True)})
        self.assertEqual(options, {**options, "threshold_convergence": (False, True)})
        self.assertEqual(options, {**options, "sample_sigma": (False, True)})
        self.assertEqual(
            options, {**options, "repelling_type": ("NONE", "COVERAGE", "ADAPTIVE")}
        )

    def test_configspace(self):
        cspace = c_maes.get_configspace(2)
        default = cspace.get_default_configuration()
        self.assertIsInstance(cspace, ConfigurationSpace)
        settings = c_maes.settings_from_config(2, default)
        self.assertIsInstance(settings, c_maes.Settings)
        self.assertEqual(settings.cc, None)
        self.assertEqual(settings.modules.sampler, c_maes.options.BaseSampler.UNIFORM)

        changed = deepcopy(default)
        changed["cc_factor"] = 0.5
        changed["sampler"] = "HALTON"
        settings_changed = c_maes.settings_from_config(2, changed)
        weights = c_maes.Parameters(settings).weights
        self.assertAlmostEqual(settings_changed.cc, 0.5 * weights.cc)
        self.assertEqual(
            settings_changed.modules.sampler, c_maes.options.BaseSampler.HALTON
        )

    def test_default_configuration_is_the_default_settings(self):
        """Over the pairwise combinations, the default numeric configuration changes nothing."""
        dim = 6
        defaults = c_maes.get_configspace(dim).get_default_configuration()
        numeric = {
            name: value
            for name, value in dict(defaults).items()
            if name in ("lambda0", "sigma0") or name.endswith("_factor")
        }
        self.assertEqual(numeric["lambda0"], c_maes.Settings(dim).lambda0)
        for modules in pairwise_configs(module_options()) + [{}]:
            with self.subTest(modules=modules):
                from_config = c_maes.settings_from_config(dim, {**modules, **numeric})
                plain = c_maes.settings_from_dict(dim, **modules)
                self.assertEqual(str(from_config.modules), str(plain.modules))
                for name in ("lambda0", "mu0", "sigma0") + NUMERIC + ("acov",):
                    self.assertEqual(getattr(from_config, name), getattr(plain, name), name)
                w_config = c_maes.Parameters(from_config).weights
                w_plain = c_maes.Parameters(plain).weights
                for name in NUMERIC + ("mueff", "lazy_update_interval"):
                    self.assertEqual(getattr(w_config, name), getattr(w_plain, name), name)

    def test_factors_are_relative_to_the_defaults_of_the_modules(self):
        dim = 8
        for ssa in ("CSA", "XNES", "SR", "SA"):
            modules = dict(ssa=ssa, matrix_adaptation="SEPARABLE")
            plain = c_maes.Parameters(c_maes.settings_from_dict(dim, **modules)).weights
            config = dict(modules, cs_factor=2.0, c1_factor=0.5, damps_factor=3.0, mu0_factor=0.5)
            settings = c_maes.settings_from_config(dim, config)
            weights = c_maes.Parameters(settings).weights
            self.assertAlmostEqual(weights.cs, 2 * plain.cs)
            self.assertAlmostEqual(weights.c1, 0.5 * plain.c1)
            self.assertAlmostEqual(weights.damps, 3 * plain.damps)
            reference = c_maes.settings_from_dict(dim, **modules)
            self.assertEqual(settings.mu0, max(1, round(0.5 * reference.mu0)))
            self.assertIsNone(settings.cc)  # no factor: the default, from the new cs

    def test_mu_is_between_1_and_lambda(self):
        for factor in (0.1, 2.0):
            settings = c_maes.settings_from_config(
                5, {"lambda0": 3, "mu0_factor": factor, "ssa": "SA"}
            )
            self.assertTrue(1 <= settings.mu0 <= 3)

    def test_sampled_mu_equal_to_the_default_is_kept(self):
        # lambda0 = 20, mu0 = 3 ran with mu0 = 10; mu0 = 10 is the default of lambda0 = 20
        settings = c_maes.settings_from_config(5, {"lambda0": 20, "mu0": 3})
        self.assertEqual((settings.lambda0, settings.mu0), (20, 3))
        settings = c_maes.settings_from_config(5, {"lambda0": 20, "mu0": 10, "ssa": "SA"})
        self.assertEqual(settings.mu0, 10)  # the default of SA is 5
        settings = c_maes.settings_from_config(5, {"lambda0": 20, "mu0": 10, "cs": 0.3})
        self.assertEqual((settings.mu0, settings.cs), (10, 0.3))

    def test_no_none_sample_transformation(self):
        cspace = c_maes.get_configspace(3)
        self.assertNotIn("NONE", cspace["sample_transformation"].choices)
        self.assertEqual(cspace["sample_transformation"].default_value, "GAUSSIAN")
        self.assertIn("NONE", c_maes.get_all_module_options()["sample_transformation"])

    def test_conditions(self):
        cspace = c_maes.get_configspace(4)
        for _ in range(300):
            config = dict(cspace.sample_configuration())
            names = set(config)
            restarts = config["restart_strategy"] in ("RESTART", "IPOP", "BIPOP")
            self.assertEqual("center_placement" in names, restarts)
            self.assertEqual("repelling_type" in names, restarts)
            self.assertEqual(
                "repelling_metric" in names,
                restarts and config["repelling_type"] != "NONE",
            )
            self.assertEqual("active" in names, config["matrix_adaptation"] != "NONE")
            self.assertEqual("mu0_factor" in names, config["lambda0"] > 1)
            self.assertEqual(
                "damps_factor" in names, config["ssa"] in ("CSA", "MSR", "PSR", "SR")
            )
            ma, ssa = config["matrix_adaptation"], config["ssa"]
            self.assertEqual("cc_factor" in names, ma not in ("NONE", "MATRIX", "CMSA"))
            self.assertEqual("c1_factor" in names, ma not in ("NONE", "CMSA", "NATURAL_GRADIENT"))
            self.assertEqual("cmu_factor" in names, "c1_factor" in names)
            self.assertEqual(
                "cs_factor" in names,
                ssa not in ("SR", "SA") or ma in ("MATRIX", "NATURAL_GRADIENT"),
            )
            self.assertNotEqual(config["sample_transformation"], "NONE")
            if config["matrix_adaptation"] == "COVARIANCE":
                self.assertEqual(config["ssa"], "CSA")

    def test_sampled_configurations_run(self):
        dim = 3
        cspace = c_maes.get_configspace(dim)
        cspace.seed(7)
        for config in cspace.sample_configuration(60):
            with self.subTest(config=dict(config)):
                settings = c_maes.settings_from_config(dim, config, budget=300, seed=1)
                es = c_maes.ModularCMAES(settings)
                es.run(sphere)
                self.assertGreater(es.p.stats.evaluations, 0)

    def test_the_inactive_parameters_have_no_effect(self):
        """A parameter that the conditions leave out does not change the run."""
        for modules, name, value in [
            (dict(ssa="TPA"), "damps", 2.0),
            (dict(matrix_adaptation="NONE"), "cc", 0.1),
            (dict(matrix_adaptation="CMSA"), "c1", 0.1),
            (dict(ssa="SR", matrix_adaptation="CHOLESKY"), "cs", 0.9),
        ]:
            runs = []
            for extra in ({}, {name: value}):
                s = c_maes.settings_from_dict(4, **modules, **extra, budget=600, seed=2)
                es = c_maes.ModularCMAES(s)
                es.run(sphere)
                runs.append((es.p.stats.global_best.y, es.p.stats.evaluations))
            self.assertEqual(runs[0], runs[1], (modules, name))

    def test_from_dict(self):
        settings = c_maes.settings_from_dict(2, active=True, cc=1)
        self.assertEqual(settings.modules.active, True)
        self.assertEqual(settings.cc, 1)

    def test_fmin(self):
        xopt, fopt, evals, es = c_maes.fmin(
            sphere, [1, 2], 0.2, 300, matrix_adaptation="NONE"
        )
        self.assertLess(fopt, 1e-4)
        self.assertLessEqual(evals, 300)
        self.assertEqual(sphere(xopt), fopt)


if __name__ == "__main__":
    unittest.main()
