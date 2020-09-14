import os
import io
import shutil
import unittest
import unittest.mock

import numpy as np

from ccmaes import utils
from ccmaes.parameters import Parameters
from ccmaes.configurablecmaes import ConfigurableCMAES


class TestUtils(unittest.TestCase):
    
    def setUp(self):
        class Foo(utils.AnnotatedStruct):
            x: int
            y: float = 0.
            z: np.ndarray = np.ones(5)
            c: (None, "x", "y", 1) = None
        self.fooclass = Foo



    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_timeit(self, mock_stdout):
        @utils.timeit
        def f():
            pass
        f()
        self.assertIn("Time elapsed", mock_stdout.getvalue())

    def test_anyof(self):
        foo = self.fooclass(1)
        self.assertEqual(foo.c, None)
        with self.assertRaises(ValueError):
            foo.c = 'z'
            foo.c = 10
            foo.c = 1.
        foo.c = 'x'
        self.assertEqual(foo.c, 'x')

    def test_instanceof(self):
        foo = self.fooclass(1)
        self.assertEqual(int, type(foo.x))
        self.assertEqual(float, type(foo.y))
        self.assertEqual(np.ndarray, type(foo.z))

        x = np.zeros(1)
        foo.z = x 
        self.assertListEqual(foo.z.tolist(), x.tolist())
        self.assertNotEqual(id(foo.z), id(x))
        
        with self.assertRaises(TypeError):
            bar = self.fooclass(None) 
            bar = self.fooclass('')
            bar = self.fooclass('x')
            bar = self.fooclass(1.)

            foo.y = 1
            foo.y = 'z'
            foo.z = 1
            foo.z = 'z'


    def test_metaclass_raises(self):
        with self.assertRaises(TypeError):
            class Foo(utils.AnnotatedStruct):
                x: 'x'
    
    def test_repr(self):
        self.assertEqual(type(repr(self.fooclass(1))), str)


    def test_tpa_mutation(self):
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
        utils._tpa_mutation(utils.sphere_function, p, x, y, f)
        for i, l in enumerate([x,y,f]):
            self.assertEqual(len(l), 2)
        
        self.assertListEqual((-y[0]).tolist(), y[1].tolist())
       
        for xi, fi in zip(x, f):
            self.assertEqual(utils.sphere_function(xi), fi)
        
        self.assertEqual(p.rank_tpa, p.a_tpa + p.b_tpa) 

        p = TpaParameters(-2)
        x, y, f = [], [], []
        utils._tpa_mutation(utils.sphere_function, p, x, y, f)
        self.assertEqual(p.rank_tpa, -p.a_tpa)


    def test_descriptor(self):
        class Foo:
            x = utils.Descriptor()

        self.assertIsInstance(Foo.x, utils.Descriptor)
        foo = Foo()
        foo.x = 1
        self.assertEqual(foo.x, 1)
        del foo.x
        self.assertNotIn('x', foo.__dict__)     
            

    def test_scale_with_treshold(self):
        threshold = 5
        z = np.ones(20)
        new_z = utils._scale_with_threshold(z.copy(), threshold)
        new_z_norm = np.linalg.norm(new_z)
        self.assertNotEqual((z == new_z).all(), True)
        self.assertNotEqual(np.linalg.norm(z), new_z_norm)
        self.assertGreater(new_z_norm, threshold)

    def test_correct_bounds(self):
        x = np.ones(5) * np.array([2, 4, 6, -7, 3])
        ub, lb = np.ones(5) * 5, np.ones(5) * -5
        correction_methods = Parameters.__annotations__.get("bound_correction")
        new_x, corrected = utils._correct_bounds(x.copy(), ub, lb, correction_methods[0])

        self.assertEqual((x == new_x).all(), True)
        self.assertEqual(corrected, True)
        
        for correction_method in correction_methods[1:]:
            new_x, corrected = utils._correct_bounds(x.copy(), ub, lb, correction_method)     
            self.assertEqual(corrected, True)
            self.assertNotEqual((x == new_x).all(), True)
            self.assertGreaterEqual( np.min(new_x), -5)
            self.assertLessEqual(np.max(new_x), 5)
            self.assertEqual((x[[0, 1, 4]] == new_x[[0, 1, 4]]).all(), True)

    def test_ert(self):
        evals = [5000, 45000, 1000, 100, 10]
        budget = 10000
        ert, ert_sd, n_succ = utils.ert(evals, budget)
        self.assertEqual(n_succ, 4)
        self.assertAlmostEqual( ert, 12777.5)
        self.assertAlmostEqual(ert_sd, 17484.642861665)

        for evals in ([50000], [], [int(1e10)]):    
            ert, ert_sd, n_succ = utils.ert(evals, budget)
            self.assertEqual(ert, float("inf"))
            self.assertEqual(np.isnan(ert_sd), True)
            self.assertEqual(n_succ, 0)


    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_evaluate(self, mock_std):
        data_folder = os.path.join(os.path.dirname(__file__), 'tmp')
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)
        utils.evaluate(ConfigurableCMAES, 1, 1, 1, logging=True, data_folder=data_folder)
        shutil.rmtree(data_folder) 
        utils.evaluate(ConfigurableCMAES, 1, 1, 1)

if __name__ == "__main__":
    unittest.main()
