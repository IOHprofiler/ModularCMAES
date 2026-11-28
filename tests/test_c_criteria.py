"""Tests of the restart criteria of the C++ ModularCMA-ES."""

import unittest

import modcma.c_maes as ccma


class MyCriterion(ccma.restart.Criterion):
    def __init__(self, name):
        super().__init__(f"MyCriterion{name}")

    def on_reset(self, par: ccma.Parameters):
        """Called when a restart happens (also at the start)"""

    def update(self, par: ccma.Parameters):
        """Called after each iteration, needs to modify self.met"""
        self.met = True


class TestCriteria(unittest.TestCase):
    def setUp(self):
        mod = ccma.parameters.Modules()
        mod.restart_strategy = ccma.options.RestartStrategy.RESTART
        settings = ccma.Settings(dim=3, budget=1000, verbose=True, modules=mod)
        self.cma = ccma.ModularCMAES(settings)

    def test_modify(self):
        self.cma.p.criteria.items = self.cma.p.criteria.items[1:3]
        self.assertEqual(len(self.cma.p.criteria.items), 2)
        c1 = MyCriterion("c1")
        c2 = MyCriterion("c2")
        self.cma.p.criteria.items[1] = c1
        self.assertEqual(len(self.cma.p.criteria.items), 2)
        self.assertEqual(self.cma.p.criteria.items[1].name, "MyCriterionc1")

        self.cma.p.criteria.items.append(c2)
        self.assertEqual(len(self.cma.p.criteria.items), 3)
        self.assertEqual(self.cma.p.criteria.items[2].name, "MyCriterionc2")

        self.cma.p.criteria.items.append(MyCriterion("c3"))
        self.assertEqual(len(self.cma.p.criteria.items), 4)
        self.assertEqual(self.cma.p.criteria.items[-1].name, "MyCriterionc3")

    def test_the_criteria_relative_to_the_start_of_the_run_show_sigma_start(self):
        # TolX, MaxDSigma and MinDSigma compare with the step size at the start of the run
        items = {c.name: c for c in self.cma.p.criteria.items}
        self.cma.p.mutation.sigma = 0.25
        for name in ("TolX", "MaxDSigma", "MinDSigma"):
            with self.subTest(name=name):
                items[name].reset(self.cma.p)
                self.assertEqual(items[name].sigma_start, 0.25)


if __name__ == "__main__":
    unittest.main()
