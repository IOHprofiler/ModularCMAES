import unittest
import numpy as np
from modcma.c_maes import es, parameters, utils

def sphere(x: np.ndarray) -> float:
    return np.linalg.norm(x)


class TestES(unittest.TestCase):
    def test_1p1(self):
        d = 2
        utils.set_seed(1)
        x0 = np.ones(d)
        
        alg = es.OnePlusOneES(
            d, 
            x0, 
            sphere(x0)
        )
        alg(sphere)
        self.assertLessEqual(alg.f, 1e-8)
        self.assertLessEqual(alg.t, 500)

    def test_mu_comma_lamb(self):
        d = 2
        x0 = np.ones(d)
        utils.set_seed(1)
        alg = es.MuCommaLambdaES(
            d, 
            x0,
            
        )
        alg(sphere)
        self.assertLessEqual(alg.f_min, 1e-8)
        self.assertLessEqual(alg.e, 1000)

if __name__ == "__main__":
    unittest.main()