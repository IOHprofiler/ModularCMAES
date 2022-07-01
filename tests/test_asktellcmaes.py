"""Module containing tests for Ask-tell interface of ModularCMA-ES."""

import unittest

import numpy as np
from modcma import asktellcmaes
import ioh

class AskTellCMAESTestCase(unittest.TestCase):
    """Test case for ask-tell interface of Modular CMA-ES."""

    def setUp(self):
        """Test setup method."""
        self.d = 5
        self.fid = 1
        self.func = ioh.get_problem(1, dimension=5, instance=1)
        self.opt = asktellcmaes.AskTellCMAES(self.d, target=79.48)

    def test_sequential_selection_disabled(self):
        """Test whether sequential is disabled."""
        self.opt.parameters.sequential = True
        with self.assertRaises(NotImplementedError):
            _ = self.opt.ask()

    def test_unkown_xi(self):
        """Test whether errors are produced correctly."""
        with self.assertRaises(RuntimeError):
            self.opt.tell(np.random.uniform(size=(self.d, 1)), 90.0)
        _ = self.opt.ask()
        with self.assertRaises(ValueError):
            self.opt.tell(np.random.uniform(size=(self.d, 1)), 90.0)

    def test_warns_on_repeated_xi(self):
        """Test whether warnings are produced correctly."""
        xi = self.opt.ask()
        self.opt.tell(xi, self.func(xi.flatten()))
        with self.assertWarns(UserWarning):
            self.opt.tell(xi, self.func(xi.flatten()))

    def test_ask(self):
        """Test ask mechanism."""
        xi = self.opt.ask()
        self.assertIsInstance(xi, np.ndarray)
        self.assertEqual(len(xi), self.d)

    def test_tell(self):
        """Test tell mechanism."""
        xi = self.opt.ask()
        fi = self.func(xi.flatten())
        self.opt.tell(xi, fi)
        self.assertEqual(self.opt.parameters.population.f[0], fi)

    def test_single_run(self):
        """Test a single run of the mechanism."""
        while True:
            try:
                xi = self.opt.ask()
                self.opt.tell(xi, self.func(xi.flatten()))
            except StopIteration:
                break
        self.assertNotEqual(self.opt.parameters.fopt, None)
        self.assertNotEqual(len(self.opt.ask_queue), 0)

    def test_disabled_functions(self):
        """Test whether errors are produced correctly."""    
        with self.assertRaises(NotImplementedError):
            self.opt.run()
        with self.assertRaises(NotImplementedError):
            self.opt.step()


if __name__ == "__main__":
    unittest.main()