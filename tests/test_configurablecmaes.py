import unittest
from functools import partial

import numpy as np

from ccmaes import parameters, utils, configurablecmaes


class TestConfigurableCMAESMeta(type):
    def __new__(classes, name, bases,  clsdict):
        def gen_test(module, value):
            def do_test(self):
                return self.run_module(module, value)
            return do_test

        for module in  parameters.Parameters.__modules__:
            m = getattr(parameters.Parameters, module)
            if type(m) == utils.AnyOf:
                for o in filter(None, m.options):
                    clsdict[f"test_{module}_{o}"] = gen_test(module, o)
            elif type(m) == utils.InstanceOf:
                clsdict[f"test_{module}_True"] = gen_test(module, True)
                
        clsdict[f"test_standard"] = gen_test('active', True)
        return super().__new__(classes, name, bases, clsdict)



class TestConfigurableCMAES(
        unittest.TestCase, 
        metaclass=TestConfigurableCMAESMeta):

    _dim = 2 
    _budget = int(1e4 * _dim)
    _target = 79.48 + 1e-08

    def run_module(self, module, value):
        self.p = parameters.Parameters(
                self._dim, self._target, self._budget,
                **{module:value}
        ) 
        self.c = configurablecmaes.ConfigurableCMAES(
                    utils.sphere_function, parameters=self.p).run()
 

if __name__ == '__main__':
    unittest.main()
