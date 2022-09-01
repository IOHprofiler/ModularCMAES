"""Module containing tests for ModularCMA-ES."""

import os
import shutil
import io
import json
import unittest
import unittest.mock

import numpy as np
import ioh

from modcma import parameters, utils, modularcmaes


class TestModularCMAESMeta(type):
    """Metaclass for generating test-cases."""

    def __new__(cls, name, bases, clsdict):
        """Method for generating new classes."""

        def make_test_fid(module, value, fid):
            return {
                f"test_{module}_{value}_f{fid}": lambda self: self.run_bbob_function(
                    module, value, fid
                )
            }
            
        def make_test_option(module, value):
            return {
                f"test_{module}_{value}": lambda self: self.run_module(
                    module, value
                )
            }

        for module in parameters.Parameters.__modules__:
            m = getattr(parameters.Parameters, module)
            if type(m) == utils.AnyOf:
                for o in filter(None, m.options):
                    for fid in range(1, 25):
                        clsdict.update(make_test_fid(module, o, fid))
                    clsdict.update(make_test_option(module, o))

            elif type(m) == utils.InstanceOf:
                for fid in range(1, 25):
                    clsdict.update(make_test_fid(module, True, fid))

                clsdict.update(make_test_option(module, True))

        return super().__new__(cls, name, bases, clsdict)


class TestModularCMAES(unittest.TestCase, metaclass=TestModularCMAESMeta):
    """Test case for ModularCMAES Object. Gets applied for all Parameters.__modules__."""

    _dim = 2
    _budget = int(1e1 * _dim)
    
    def __init__(self, args, **kwargs):
        """Initializes the expected function value dictionary."""
        with open("tests/expected.json", "r") as f:
            self.bbob2d_per_module = json.load(f)
        super().__init__(args, **kwargs)

    def run_module(self, module, value):
        """Test a single run of the mechanism with a given module active."""
        self.p = parameters.Parameters(
            self._dim, budget=self._budget, **{module: value}
        )
        self.c = modularcmaes.ModularCMAES(sum, parameters=self.p).run()

    def run_bbob_function(self, module, value, fid):
        """Expects the output to be consistent with BBOB_2D_PER_MODULE_20_ITER."""
        np.random.seed(42)
        f = ioh.get_problem(fid, dimension=self._dim, instance=1)
        self.p = parameters.Parameters(
            self._dim, budget=self._budget, **{module: value}
        )
        self.c = modularcmaes.ModularCMAES(f, parameters=self.p).run()
        self.assertAlmostEqual(
            f.state.current_best_internal.y,
            self.bbob2d_per_module[f"{module}_{value}"][fid - 1],
        )

    def test_select_raises(self):
        """Test whether errors are produced correctly."""
        c = modularcmaes.ModularCMAES(sum, 5, mirrored="mirrored pairwise")
        c.mutate()
        c.parameters.population = c.parameters.population[:3]
        with self.assertRaises(ValueError):
            c.select()

    def test_local_restart(self):
        """Test a single iteration of the mechanism with a given local restart active."""
        for lr in filter(None, parameters.Parameters.local_restart.options):
            c = modularcmaes.ModularCMAES(sum, self._dim, local_restart=lr)
            for _ in range(10):
                c.step()

            c.parameters.max_iter = 5
            c.step()


class TestModularCMAESSingle(unittest.TestCase):
    """Test case for ModularCMAES Object, holds custom tests."""

    def test_tpa_threshold_cov_sequential(self):
        c = modularcmaes.ModularCMAES(sum, 2,
            threshold_convergence=True, sequential=True, 
            step_size_adaptation='tpa', budget=10).run()
        self.assertLess(c.parameters.fopt, 0.)

    def test_str_repr(self):
        """Test the output of repr and str."""
        c = modularcmaes.ModularCMAES(sum, 5)
        self.assertIsInstance(str(c), str)
        self.assertIsInstance(repr(c), str)

    def test_n_generations(self):
        """Test n iterations of the mechanism."""
        c = modularcmaes.ModularCMAES(sum, 5, n_generations=5)
        self.assertEqual(1, len(c.break_conditions))

        for _ in range(5):
            c.step()

        self.assertTrue(any(c.break_conditions))

        c = modularcmaes.ModularCMAES(sum, 5)
        self.assertEqual(2, len(c.break_conditions))

    def testtpa_mutation(self):
        """Test tpa mutation."""

        class TpaParameters:
            sigma = 0.4
            rank_tpa = None
            a_tpa = 0.3
            b_tpa = 0

            def __init__(self, m_factor=1.1):
                self.m = np.random.rand(5, 1)
                self.m_old = self.m * m_factor

        p = TpaParameters()
        y, x, f = modularcmaes.tpa_mutation(sum, p)
        for _, l in enumerate([x, y, f]):
            self.assertIn(2, l.shape)

        self.assertListEqual((-y[:, 0]).tolist(), y[:, 1].tolist())

        for xi, fi in zip(x.T, f):
            self.assertEqual(sum(xi), fi)

        self.assertEqual(p.rank_tpa, p.a_tpa + p.b_tpa)

        p = TpaParameters(-2)
        y, x, f = modularcmaes.tpa_mutation(sum, p)
        self.assertEqual(p.rank_tpa, -p.a_tpa)

    def test_scale_with_treshold(self):
        """Test threshold mutations."""
        threshold = 5
        z = np.ones(20)
        new_z = modularcmaes.scale_with_threshold(z.copy(), threshold)
        new_z_norm = np.linalg.norm(new_z)
        self.assertNotEqual((z == new_z).all(), True)
        self.assertNotEqual(np.linalg.norm(z), new_z_norm)
        self.assertGreater(new_z_norm, threshold)

    def testcorrect_bounds(self):
        """Test bound correction."""
        x = np.ones(5) * np.array([2, 4, 6, -7, 3])
        ub, lb = np.ones(5) * 5, np.ones(5) * -5
        disabled, *correction_methods = parameters.Parameters.__annotations__.get(
            "bound_correction"
        )
        new_x, corrected = modularcmaes.correct_bounds(x.copy(), ub, lb, disabled)

        self.assertEqual((x == new_x).all(), True)
        self.assertEqual(corrected, True)

        for correction_method in correction_methods:
            new_x, corrected = modularcmaes.correct_bounds(
                x.copy(), ub, lb, correction_method
            )
            self.assertEqual(corrected, True)
            self.assertNotEqual((x == new_x).all(), True)
            self.assertGreaterEqual(np.min(new_x), -5)
            self.assertLessEqual(np.max(new_x), 5)
            self.assertEqual((x[[0, 1, 4]] == new_x[[0, 1, 4]]).all(), True)

        with self.assertRaises(ValueError):
            modularcmaes.correct_bounds(x.copy(), ub, lb, "something_undefined")
            
    def test_popsize_changes(self):
        """Test manual changes to population size."""
        c = modularcmaes.ModularCMAES(sum, 2)
        c.step()
        c.parameters.update_popsize(40)
        c.step()
        self.assertEqual(c.parameters.population.n, 40)
        c.parameters.update_popsize(6)
        c.step()
        self.assertEqual(c.parameters.population.n, 6)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_evaluate_bbob(self, mock_std):
        """Test the mechanism of evaluate_bbob."""
        data_folder = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)
        self.assertTrue(os.path.isdir(data_folder))
        modularcmaes.evaluate_bbob(1, 2, 1, logging=True, data_folder=data_folder)
        shutil.rmtree(data_folder)
        modularcmaes.evaluate_bbob(1, 2, 2)
        


if __name__ == "__main__":
    unittest.main()
