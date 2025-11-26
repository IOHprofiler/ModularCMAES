import unittest
from modcma import c_maes


def sphere(x):
    return sum(xi**2 for xi in x)


class TestInteger(unittest.TestCase):
    def test_int(self):
        settings = c_maes.settings_from_dict(2, integer_variables=[0])
        cma = c_maes.ModularCMAES(settings)
        
        for _ in range(3):
            cma.mutate(sphere)
            cma.select()
            cma.recombine()
            cma.adapt()
            
        # breakpoint()
        
