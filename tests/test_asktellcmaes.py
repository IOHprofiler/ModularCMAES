import unittest

import numpy as np
from modcma import asktellcmaes
from IOHexperimenter import IOH_function

class AskTellCMAESTestCase(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.fid = 1
        self.func = IOH_function(1, 5, 1, suite = "BBOB")
        self.opt = asktellcmaes.AskTellCMAES(self.d, target = 79.48)

    def test_sequential_selection_disabled(self):
        self.opt.parameters.sequential = True
        with self.assertRaises(NotImplementedError):
            _ = self.opt.ask()

    def test_unkown_xi(self):
        with self.assertRaises(RuntimeError):
            self.opt.tell(np.random.random((self.d, 1)), 90.)
        _ = self.opt.ask()
        with self.assertRaises(ValueError):
            self.opt.tell(np.random.random((self.d, 1)), 90.)
        

    def test_warns_on_repeated_xi(self):
        xi = self.opt.ask()
        self.opt.tell(xi, self.func(xi.flatten()))
        with self.assertWarns(UserWarning):
            self.opt.tell(xi, self.func(xi.flatten()))

    def test_ask(self):
        xi = self.opt.ask()
        self.assertIsInstance(xi, np.ndarray)
        self.assertEqual(len(xi), self.d)

    def test_tell(self):
        xi = self.opt.ask()
        fi = self.func(xi.flatten())
        self.opt.tell(xi, fi)
        self.assertEqual(self.opt.parameters.population.f[0], fi)
    
    def test_single_run(self):
        while True:
            try:
                xi = self.opt.ask()
                self.opt.tell(xi, self.func(xi.flatten()))
            except StopIteration:
                break
        self.assertNotEqual(self.opt.parameters.fopt, None)
        self.assertNotEqual(len(self.opt.ask_queue), 0)

    def test_disabled_functions(self):
        with self.assertRaises(NotImplementedError):
            self.opt.run()
        with self.assertRaises(NotImplementedError):
            self.opt.step()

if __name__ == '__main__':
    unittest.main()