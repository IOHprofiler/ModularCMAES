import os
import io
import unittest
import unittest.mock

import numpy as np

from modcma import utils


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

    def test_descriptor(self):
        class Foo:
            x = utils.Descriptor()

        self.assertIsInstance(Foo.x, utils.Descriptor)
        foo = Foo()
        foo.x = 1
        self.assertEqual(foo.x, 1)
        del foo.x
        self.assertNotIn('x', foo.__dict__)     

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


  

if __name__ == "__main__":
    unittest.main()
