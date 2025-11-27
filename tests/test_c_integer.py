import unittest
from modcma import c_maes
import numpy as np

def sphere(x):
    return sum(round(xi + 7)**2  for xi in x)


class TestInteger(unittest.TestCase):
    def test_int(self):
        dim = 4
        settings = c_maes.settings_from_dict(
            dim, 
            integer_variables=list(range(dim)), 
            target=0,
            lb=[-50] * dim,
            ub=[50] * dim,
            # active=True,
            # sample_transformation='LAPLACE'
        )
        cma = c_maes.ModularCMAES(settings)
        c_maes.utils.set_seed(100)
        while not cma.break_conditions():            
            cma.mutate(sphere)
            
            print(cma.p.stats.t, cma.p.stats.evaluations, cma.p.pop.f)
            breakpoint()
            
            # if cma.p.pop.s[0] < (cma.p.weights.mueff / dim):
                # breakpoint()
            
            cma.select()
            cma.recombine()
            cma.adapt()
            
        # breakpoint()
        
