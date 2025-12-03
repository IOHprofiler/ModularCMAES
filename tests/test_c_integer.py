import unittest
from modcma import c_maes
import numpy as np

def sphere(x):
    return sum(round(xi + 7)**2  for xi in x)


class TestInteger(unittest.TestCase):
    def test_int(self):
        dim = 10
        settings = c_maes.settings_from_dict(
            dim, 
            integer_variables=list(range(dim)), 
            target=0,
            lb=[-50] * dim,
            ub=[50] * dim,
            # active=True,
            # lambda0=1,
            sample_transformation='LAPLACE',
            ssa="SA"
        )
        cma = c_maes.ModularCMAES(settings)
        c_maes.utils.set_seed(100)
        while not cma.break_conditions():            
            cma.mutate(sphere)
            cma.select()
            cma.recombine()
            cma.adapt()
            
        self.assertEqual(cma.p.stats.global_best.y, 0.0)
        
