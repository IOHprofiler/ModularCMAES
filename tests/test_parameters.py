"""Module containing tests for ModularCMA-ES Parameters."""

from email.mime import base
import os
import unittest
import warnings
import pickle

import numpy as np

from modcma.parameters import Parameters, BIPOPParameters
from modcma.utils import AnyOf
from modcma.population import Population


class TestParameters(unittest.TestCase):
    """Test case for Parameters object of Modular CMA-ES."""

    def setUp(self):
        """Test setup method."""
        np.random.seed(42)
        self.p = Parameters(5)

    def try_wrong_types(self, p, name, type_):
        """Test for wrong input types."""
        for x in (1, 1.0, "x", True, np.ndarray,):
            if type(x) != type_:
                with self.assertRaises(TypeError, msg=f"{name} {type_} {x}"):
                    setattr(p, name, x)

    def test_updating(self):
        """Test the updating of parameters."""
        self.p.update(dict(mirrored="mirrored"))
        self.assertEqual(self.p.mirrored, "mirrored")
        self.assertEqual(self.p.sampler.__name__, "mirrored_sampling")

        with self.assertRaises(ValueError):
            self.p.update(dict(nonexist=10))

        self.p.update(dict(active=True))
        self.assertEqual(self.p.active, True)
        self.assertEqual(self.p.mirrored, "mirrored")
        self.assertEqual(self.p.sampler.__name__, "mirrored_sampling")

        old_mueff = self.p.mueff
        self.p.update(dict(weights_option="equal"), reset_default_modules=True)
        self.assertEqual(self.p.active, False)
        self.assertEqual(self.p.mirrored, None)
        self.assertEqual(self.p.weights_option, "equal")
        self.assertNotEqual(self.p.mueff, old_mueff)

    def test_bipop_parameters(self):
        """Test BIPOPParameters."""
        self.p.local_restart = "BIPOP"
        self.p.used_budget += 11
        self.p.bipop_parameters.adapt(self.p.used_budget)
        self.assertEqual(self.p.bipop_parameters.large, True)
        bp = self.p.bipop_parameters 
        self.assertEqual(bp.lambda_, self.p.lambda_ * 2)
        self.assertEqual(bp.mu, self.p.mu * 2)
        self.assertEqual(bp.sigma, 2)
        self.p.used_budget += 11
        bp.adapt(self.p.used_budget)
        self.assertEqual(self.p.bipop_parameters.large, False)
        self.assertLessEqual(bp.lambda_, self.p.lambda_)
        self.assertLessEqual(bp.mu, self.p.mu)
        self.assertLessEqual(bp.sigma, self.p.sigma0)
        self.p.used_budget += 11
        bp.adapt(self.p.used_budget)
        self.assertEqual(bp.used_budget, 33)

    def test_sampler(self):
        """Test different samplers."""
        for orth in (False, True):
            self.p.mirrored = None
            self.p.orthogonal = orth
            sampler = self.p.get_sampler()
            self.assertEqual(sampler.__name__ == "orthogonal_sampling", orth)
            self.p.mirrored = "mirrored"
            sampler = self.p.get_sampler()
            self.assertEqual(sampler.__name__, "mirrored_sampling")
            self.p.mirrored = "mirrored pairwise"
            self.assertEqual(sampler.__name__, "mirrored_sampling")

    def test_wrong_parameters(self):
        """Test whether warnings are produced correctly."""
        with self.assertWarns(RuntimeWarning):
            Parameters(1, mu=3, lambda_=2)

    def test_options(self):
        """Test setting of options."""
        for module in Parameters.__modules__:
            m = getattr(Parameters, module)
            if type(m) == AnyOf:
                for o in m.options:
                    setattr(self.p, module, o)
                    Parameters(1, **{module: o})

    def step(self):
        """Test a single iteration of the mechanism."""
        y = np.random.rand(self.p.lambda_, self.p.d).T
        x = self.p.m.reshape(-1, 1) * y
        f = np.array(list(map(sum, x)))
        self.p.used_budget += self.p.lambda_
        self.p.population = Population(x, y, f)
        self.p.m_old = self.p.m.copy()
        self.p.m *= np.linalg.norm(y, axis=1).reshape(-1, 1)
        self.p.adapt()
        self.p.old_population = self.p.population.copy()

    def set_parameter_and_step(self, pname, value, nstep=2, warning_action="default"):
        """Test a single iteration of the mechanism. after setting a parameter."""
        setattr(self.p, pname, value)
        with warnings.catch_warnings():
            warnings.simplefilter(warning_action)
            for _ in range(nstep):
                self.step()

    def test_tpa(self):
        """Test TPA."""
        self.p.rank_tpa = 0.3
        self.set_parameter_and_step("step_size_adaptation", "tpa")

    def test_msr(self):
        """Test MSR."""
        self.set_parameter_and_step("step_size_adaptation", "msr")

    def test_active(self):
        """Test active."""
        self.set_parameter_and_step("active", True)

    def test_reset(self):
        """Test if C is correctly reset if it has inf."""
        self.p.C[0][0] = np.inf
        self.step()

    def test_warning(self):
        """Test whether warnings are produced correctly."""
        self.p.compute_termination_criteria = True
        self.set_parameter_and_step("max_iter", True, 5, "ignore")

    def test_threshold(self):
        """Test treshold mutation."""
        self.step()
        self.assertEqual(type(self.p.threshold), np.float64)

    def test_from_arary(self):
        """Test instantiation from a config array."""
        c_array = [0] * 11

        with self.assertRaises(AttributeError):
            _c_array = c_array[1:].copy()
            _ = Parameters.from_config_array(5, _c_array)

        with self.assertRaises(AttributeError):
            _c_array = c_array + [0]
            _ = Parameters.from_config_array(5, _c_array)

        with self.assertRaises(AttributeError):
            _c_array = c_array.copy()
            _c_array[0] = 2
            _ = Parameters.from_config_array(5, _c_array)

        _ = Parameters.from_config_array(5, c_array)

    def test_save_load(self):
        """Test pickle save and load mechanism."""
        tmpfile = os.path.join(os.path.dirname(__file__), "tmp.pkl")
        self.p.save(tmpfile)
        _ = Parameters.load(tmpfile)
        os.remove(tmpfile)
        with self.assertRaises(OSError):
            self.p.load("__________")

        with open(tmpfile, "wb") as f:
            pickle.dump({}, f)
        with self.assertRaises(AttributeError):
            self.p.load(tmpfile)
        os.remove(tmpfile)

    def test_save_load_samplers(self):
        tmpfile = os.path.join(os.path.dirname(__file__), "tmp.pkl")
        for base_sampler in getattr(Parameters, "base_sampler").options:
            p = Parameters(2, base_sampler=base_sampler)
            p.save(tmpfile)
            sample = next(p.sampler)
            p2 = Parameters.load(tmpfile)
            sample2 = next(p2.sampler)
            self.assertTrue(np.all(sample == sample2))
        os.remove(tmpfile)

    def test_fix_lambda_even(self):
        self.p.lambda_ = 11
        self.p.mirrored = 'mirrored pairwise'
        self.assertEqual(self.p.lambda_, 11)
        self.p.init_selection_parameters()
        self.assertEqual(self.p.lambda_, 12)

        for weights_option in ("equal", "1/2^lambda"):
            self.p.weights_option = weights_option
            self.p.lambda_ = 11
            self.p.mu = 5
            self.p.init_adaptation_parameters()
            self.assertEqual(len(self.p.weights), 11)

        b = BIPOPParameters(7, 20, .5)
        b.adapt(11)
        self.assertEqual(b.lambda_small, 8)

if __name__ == "__main__":
    unittest.main()
