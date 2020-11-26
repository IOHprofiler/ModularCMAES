import os
import shutil
import io
import unittest
import unittest.mock

import numpy as np
from modcma import parameters, utils, modularcmaes


class TestModularCMAESMeta(type):
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

class TestModularCMAES(
        unittest.TestCase, 
        metaclass=TestModularCMAESMeta):

    _dim = 2 
    _budget = int(1e2 * _dim)

    def run_module(self, module, value):
        self.p = parameters.Parameters(
                self._dim, budget = self._budget,
                **{module:value}
        ) 
        self.c = modularcmaes.ModularCMAES(
                    sum, parameters=self.p).run()

    def test_select_raises(self):
        c = modularcmaes.ModularCMAES(sum, 5, 
            mirrored='mirrored pairwise'
        )
        c.mutate()
        c.parameters.population = c.parameters.population[:3]
        with self.assertRaises(ValueError):
            c.select()

    def test_local_restart(self):
        for lr in filter(None, parameters.Parameters.local_restart.options):
            c = modularcmaes.ModularCMAES(
                    sum, 5, local_restart=lr)
            for _ in range(10):
                c.step()
            
            c.parameters.max_iter = 5
            c.step()
    
    
class TestModularCMAESSingle(unittest.TestCase):
    def test_str_repr(self):
        c = modularcmaes.ModularCMAES(sum, 5)
        self.assertIsInstance(str(c), str)
        self.assertIsInstance(repr(c), str)

    def test_n_generations(self):
        c = modularcmaes.ModularCMAES(sum, 5, n_generations = 5)
        self.assertEqual(1, len(c.break_conditions))

        for i in range(5):
            c.step()

        self.assertTrue(any(c.break_conditions))

        c = modularcmaes.ModularCMAES(sum, 5)
        self.assertEqual(2, len(c.break_conditions))


    def testtpa_mutation(self):
        class TpaParameters:
            sigma = .4
            rank_tpa = None
            a_tpa = .3
            b_tpa = 0
            def __init__(self, m_factor=1.1):
                self.m =  np.ones(5) * .5
                self.m_old = self.m * m_factor
        
        p = TpaParameters()
        x, y, f = [], [], []
        modularcmaes.tpa_mutation(sum, p, x, y, f)
        for _, l in enumerate([x,y,f]):
            self.assertEqual(len(l), 2)
        
        self.assertListEqual((-y[0]).tolist(), y[1].tolist())
       
        for xi, fi in zip(x, f):
            self.assertEqual(sum(xi), fi)
        
        self.assertEqual(p.rank_tpa, p.a_tpa + p.b_tpa) 

        p = TpaParameters(-2)
        x, y, f = [], [], []
        modularcmaes.tpa_mutation(sum, p, x, y, f)
        self.assertEqual(p.rank_tpa, -p.a_tpa)

    def test_scale_with_treshold(self):
        threshold = 5
        z = np.ones(20)
        new_z = modularcmaes.scale_with_threshold(z.copy(), threshold)
        new_z_norm = np.linalg.norm(new_z)
        self.assertNotEqual((z == new_z).all(), True)
        self.assertNotEqual(np.linalg.norm(z), new_z_norm)
        self.assertGreater(new_z_norm, threshold)

    def testcorrect_bounds(self):
        x = np.ones(5) * np.array([2, 4, 6, -7, 3])
        ub, lb = np.ones(5) * 5, np.ones(5) * -5
        disabled, *correction_methods = parameters.Parameters.__annotations__\
            .get("bound_correction")
        new_x, corrected = modularcmaes.correct_bounds(x.copy(), ub, lb, disabled)

        self.assertEqual((x == new_x).all(), True)
        self.assertEqual(corrected, True)
        
        for correction_method in correction_methods:
            new_x, corrected = modularcmaes.\
                correct_bounds(x.copy(), ub, lb, correction_method)     
            self.assertEqual(corrected, True)
            self.assertNotEqual((x == new_x).all(), True)
            self.assertGreaterEqual( np.min(new_x), -5)
            self.assertLessEqual(np.max(new_x), 5)
            self.assertEqual((x[[0, 1, 4]] == new_x[[0, 1, 4]]).all(), True)

        with self.assertRaises(ValueError):
            modularcmaes.correct_bounds(x.copy(), ub, lb, "something_undefined")
            
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_evaluate_bbob(self, mock_std):
        data_folder = os.path.join(os.path.dirname(__file__), 'tmp')
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)
        modularcmaes.evaluate_bbob(1, 1, 1, logging=True, data_folder=data_folder)
        shutil.rmtree(data_folder) 
        modularcmaes.evaluate_bbob(1, 1, 1)
        



if __name__ == '__main__':
    unittest.main()
