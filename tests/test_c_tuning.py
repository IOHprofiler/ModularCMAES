import random
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
        changed["cc"] = 0.1
        changed["sampler"] = "HALTON"
        settings_changed = c_maes.settings_from_config(2, changed)
        self.assertEqual(settings_changed.cc, 0.1)
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
            if name in ("lambda0", "mu0", "sigma0") + NUMERIC
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

    def test_a_default_value_is_the_default_of_the_modules(self):
        """The numeric names, ranges and defaults are those of master's space, and its rule holds:
        a value equal to the default of the space is not passed, any other is."""
        dim = 8
        cspace = c_maes.get_configspace(dim)
        for name in ("lambda0", "mu0", "sigma0") + NUMERIC:
            self.assertIn(name, cspace)
        # the lower bounds of cs and damps moved from 0 to 1e-6 (#169)
        self.assertEqual((cspace["cs"].lower, cspace["cs"].upper), (1e-6, 1))
        self.assertEqual((cspace["damps"].lower, cspace["damps"].upper), (1e-6, 10))
        self.assertEqual((cspace["lambda0"].lower, cspace["lambda0"].upper), (1, 50 * dim))
        defaults = dict(cspace.get_default_configuration())
        rng = random.Random(3)
        for _ in range(60):
            modules = {
                name: rng.choice(options)
                for name, options in c_maes.get_all_module_options().items()
            }
            numeric = {}
            for name in NUMERIC + ("sigma0",):
                if rng.random() < 0.5:
                    numeric[name] = rng.uniform(0.01, 0.45)  # c1 + cmu <= 1 (#147)
                else:
                    numeric[name] = defaults[name]  # the default of the modules
            settings = c_maes.settings_from_config(dim, {**modules, **numeric})
            # master's reference: only what differs from the default goes to Settings
            passed = {k: v for k, v in numeric.items() if v != defaults[k]}
            reference = c_maes.settings_from_dict(dim, **modules, **passed)
            self.assertEqual(str(settings.modules), str(reference.modules))
            for name in ("lambda0", "mu0", "sigma0") + NUMERIC:
                self.assertEqual(getattr(settings, name), getattr(reference, name), name)

    def test_sampled_mu_equal_to_the_default_is_kept(self):
        # lambda0 = 20, mu0 = 3 ran with mu0 = 10; mu0 = 10 is the default of lambda0 = 20
        settings = c_maes.settings_from_config(5, {"lambda0": 20, "mu0": 3})
        self.assertEqual((settings.lambda0, settings.mu0), (20, 3))
        settings = c_maes.settings_from_config(5, {"lambda0": 20, "mu0": 10, "ssa": "SA"})
        self.assertEqual(settings.mu0, 10)  # the default of SA is 5
        settings = c_maes.settings_from_config(5, {"lambda0": 20, "mu0": 10, "cs": 0.3})
        self.assertEqual((settings.mu0, settings.cs), (10, 0.3))
        # mu0 equal to its default (lambda0 // 2 of the default lambda0 = 7) with another lambda0
        default = c_maes.get_configspace(5).get_default_configuration()
        settings = c_maes.settings_from_config(5, {"lambda0": 20, "mu0": default["mu0"]})
        self.assertEqual(settings.mu0, default["mu0"])
        # no lambda0, mu0 equal to its default: the default of the modules (lambda / 4 for SA)
        settings = c_maes.settings_from_config(5, {"mu0": default["mu0"], "ssa": "SA"})
        self.assertEqual(settings.mu0, c_maes.settings_from_dict(5, ssa="SA").mu0)

    def test_no_none_sample_transformation(self):
        cspace = c_maes.get_configspace(3)
        self.assertNotIn("NONE", cspace["sample_transformation"].choices)
        self.assertEqual(cspace["sample_transformation"].default_value, "GAUSSIAN")
        self.assertIn("NONE", c_maes.get_all_module_options()["sample_transformation"])

    def test_conditions(self):
        cspace = c_maes.get_configspace(4)
        for _ in range(400):
            config = dict(cspace.sample_configuration())
            names = set(config)
            ma, ssa = config["matrix_adaptation"], config["ssa"]
            restarts = config["restart_strategy"] != "STOP"
            self.assertEqual("center_placement" in names, restarts)
            self.assertEqual("repelling_type" in names, restarts)
            self.assertEqual(
                "repelling_metric" in names,
                restarts and config["repelling_type"] != "NONE",
            )
            self.assertEqual("active" in names, ma != "NONE" and config["lambda0"] > 1)
            self.assertEqual("mu0" in names, config["lambda0"] > 1)
            one = config["lambda0"] == 1
            for name in ("elitist", "sequential_selection", "weights"):
                self.assertEqual(name in names, not one, name)
            self.assertEqual(
                "damps" in names, one or ssa in ("CSA", "MSR", "PSR", "SR")
            )
            if one:
                self.assertNotIn(config["restart_strategy"], ("IPOP", "BIPOP"))
                self.assertNotEqual(config["mirrored"], "PAIRWISE")
            self.assertEqual("cc" in names, ma not in ("NONE", "MATRIX", "CMSA"))
            self.assertEqual(
                "cs" in names,
                ssa not in ("SR", "SA") or ma not in ("NONE", "CHOLESKY", "CMSA"),
            )
            c1_active = ma in ("COVARIANCE", "COVARIANCE_NO_EIGV", "SEPARABLE", "MATRIX", "CHOLESKY")
            c1_active |= ma in ("CMSA", "NATURAL_GRADIENT") and bool(config.get("active"))
            self.assertEqual("c1" in names, bool(c1_active))
            self.assertEqual("cmu" in names, "c1" in names)
            self.assertNotEqual(config["sample_transformation"], "NONE")
            if ma == "COVARIANCE":
                self.assertEqual(ssa, "CSA")

    def test_sampled_configurations_run(self):
        dim = 3
        cspace = c_maes.get_configspace(dim)
        cspace.seed(7)
        for config in cspace.sample_configuration(60):
            with self.subTest(config=dict(config)):
                try:
                    settings = c_maes.settings_from_config(dim, config, budget=300, seed=1)
                except ValueError as error:
                    # the only rejection left in the space (#169)
                    self.assertIn("c1 + cmu", str(error))
                    self.assertGreater(config["c1"] + config["cmu"], 1)
                    continue
                es = c_maes.ModularCMAES(settings)
                es.run(sphere)
                self.assertGreater(es.p.stats.evaluations, 0)

    def test_the_lower_bounds_of_cs_cc_damps_are_positive(self):
        # #169: Settings rejects cs, cc and damps <= 0, so no sample may be there. The defaults
        # stay, and c1 and cmu keep the lower bound 0.
        for dim in (2, 10):
            cspace = c_maes.get_configspace(dim)
            for name in ("cs", "cc", "damps"):
                self.assertGreater(cspace[name].lower, 0)
            for name in ("c1", "cmu"):
                self.assertEqual(cspace[name].lower, 0)
        cspace = c_maes.get_configspace(4)
        cspace.seed(3)
        for config in cspace.sample_configuration(300):
            for name in ("cs", "cc", "damps"):
                if name in config:
                    self.assertGreater(config[name], 0)
                    # Settings accepts the value
                    c_maes.settings_from_config(4, {name: config[name]})
        # the values at the lower bound are accepted, a value of 0 is not
        for name in ("cs", "cc", "damps"):
            c_maes.settings_from_config(4, {name: cspace[name].lower})
            with self.assertRaises(ValueError):
                c_maes.settings_from_config(4, {name: 0.0})

    def test_c1_plus_cmu_above_one_is_rejected(self):
        # #169: the ranges of c1 and cmu are kept; the sum is checked by Settings, which
        # settings_from_config builds, so the error is raised there
        with self.assertRaisesRegex(ValueError, r"c1 \+ cmu.*c1 = 0\.7.*cmu = 0\.6"):
            c_maes.settings_from_config(5, {"c1": 0.7, "cmu": 0.6})
        c_maes.settings_from_config(5, {"c1": 0.4, "cmu": 0.6})

    def test_the_inactive_parameters_have_no_effect(self):
        """Where the space has no hyperparameter, changing the parameter changes nothing: every
        step-size rule with every matrix adaptation, with and without `active`, for the weights
        `DEFAULT`, `EQUAL` and `UTILITY` (the conditions are derived from the code, see the
        comment of `_INACTIVE_FOR`; this is the probe that checks them)."""
        options = c_maes.get_all_module_options()
        cspace = c_maes.get_configspace(4)
        probes = {"cs": 0.5, "cc": 0.5, "cmu": 0.5, "c1": 0.5, "damps": 2.0}
        checked = 0
        for ssa in options["ssa"]:
            for ma in options["matrix_adaptation"]:
                for active in (False, True):
                    for weights in ("DEFAULT", "EQUAL", "UTILITY"):
                        modules = dict(
                            ssa=ssa, matrix_adaptation=ma, active=active, weights=weights
                        )
                        plain = c_maes.settings_from_dict(4, **modules, budget=500, seed=2)
                        # the modules as Settings resolves them: what the conditions are about
                        resolved = {
                            "ssa": ssa,
                            "matrix_adaptation": ma,
                            "active": plain.modules.active,
                        }
                        base = self._run(plain)
                        for name, value in probes.items():
                            if self._has_hyperparameter(cspace, name, resolved):
                                continue
                            default = getattr(c_maes.Parameters(plain).weights, name)
                            changed = c_maes.settings_from_dict(
                                4, **modules, budget=500, seed=2, **{name: default * value}
                            )
                            self.assertEqual(self._run(changed), base, (modules, name))
                            checked += 1
        self.assertGreater(checked, 500)

    def test_lambda_1_overrides_have_no_effect(self):
        """lambda0 = 1 is a (1+1)-ES: the modules that the space leaves out change nothing."""
        runs = set()
        for elitist in (False, True):
            for sequential in (False, True):
                for weights in ("DEFAULT", "EQUAL", "UTILITY"):
                    for ssa in ("CSA",):
                        settings = c_maes.settings_from_dict(
                            4, lambda0=1, elitist=elitist, sequential_selection=sequential,
                            weights=weights, ssa=ssa, budget=300, seed=2,
                        )
                        runs.add(self._run(settings))
        self.assertEqual(len(runs), 1)
        runs = {
            self._run(
                c_maes.settings_from_dict(4, lambda0=1, restart_strategy=r, budget=300, seed=2)
            )
            for r in ("RESTART", "IPOP", "BIPOP")
        }
        self.assertEqual(len(runs), 1)
        # PAIRWISE becomes NONE (#136): forbidden in the space
        runs = {
            self._run(
                c_maes.settings_from_dict(4, lambda0=1, mirrored=m, budget=300, seed=2)
            )
            for m in ("NONE", "PAIRWISE")
        }
        self.assertEqual(len(runs), 1)
        for remove_duplicates in (True, False):
            cspace = c_maes.get_configspace(4, remove_duplicates=remove_duplicates)
            config = dict(cspace.get_default_configuration())
            for name in ("mu0", "active", "elitist", "sequential_selection", "weights"):
                config.pop(name, None)  # inactive for lambda0 = 1
            for mirrored in ("NONE", "MIRRORED", "PAIRWISE"):
                config.update(lambda0=1, mirrored=mirrored)
                forbidden = any(f.is_forbidden_value(config) for f in cspace.forbidden_clauses)
                self.assertEqual(forbidden, remove_duplicates and mirrored == "PAIRWISE")

    @staticmethod
    def _run(settings):
        es = c_maes.ModularCMAES(settings)
        es.run(sphere)
        return (es.p.stats.global_best.y, es.p.stats.evaluations, es.p.mutation.sigma)

    @staticmethod
    def _has_hyperparameter(cspace, name, modules):
        """Whether `<name>` is active for the modules, by the conditions of the space."""
        child = f"{name}"
        for condition in cspace.conditions:
            children = {c.child.name for c in getattr(condition, "components", [condition])}
            if child in children:
                return condition.satisfied_by_value(
                    {"lambda0": 8, **modules}
                )
        return True

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
