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
    _budget = int(1e2 * _dim)
    _target = 79.48 + 1e-08

    def run_module(self, module, value):
        self.p = parameters.Parameters(
                self._dim, self._target, self._budget,
                **{module:value}
        ) 
        self.c = configurablecmaes.ConfigurableCMAES(
                    utils.sphere_function, parameters=self.p).run()

    def test_select_raises(self):
        c = configurablecmaes.ConfigurableCMAES(
                    utils.sphere_function, 5, 
                    mirrored='mirrored pairwise'
                )
        
        c.mutate()
        c.parameters.population = c.parameters.population[:3]
        with self.assertRaises(ValueError):
            c.select()


    def test_local_restart(self):
        for lr in filter(None, parameters.Parameters.local_restart.options):
            c = configurablecmaes.ConfigurableCMAES(
                    utils.sphere_function, 5, 
                    local_restart=lr
                )
            for i in range(10):
                c.step()
            
            c.parameters.max_iter = 5
            c.step()






if __name__ == '__main__':
    unittest.main()
