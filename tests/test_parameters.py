import os
import unittest
import warnings
import pickle


import numpy as np

from modcma.parameters import Parameters
from modcma.utils import AnyOf
from modcma.population import Population


class TestParameters(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.p = Parameters(5)
    
    def try_wrong_types(self, p, name, type_):
        for x in (1, 1., 'x', True, np.ndarray,):
            if type(x) != type_:
                with self.assertRaises(TypeError, msg=f"{name} {type_} {x}"):
                    setattr(p, name, x)

    def test_bipop_parameters(self):
        self.p.local_restart = 'BIPOP'
        self.p.used_budget  += 11
        self.p.bipop_parameters.adapt(self.p.used_budget)
        self.assertEqual(self.p.bipop_parameters.large, True)
        bp = self.p.bipop_parameters 
        self.assertEqual(bp.lambda_, self.p.lambda_*2)
        self.assertEqual(bp.mu, self.p.mu*2)
        self.assertEqual(bp.sigma, 2)
        self.p.used_budget += 11
        bp.adapt(self.p.used_budget) 
        self.assertEqual(self.p.bipop_parameters.large, False)
        self.assertLessEqual(bp.lambda_, self.p.lambda_)
        self.assertLessEqual(bp.mu, self.p.mu)
        self.assertLessEqual(bp.sigma, self.p.init_sigma)
        self.p.used_budget += 11
        bp.adapt(self.p.used_budget) 
        self.assertEqual(bp.used_budget, 33)


    def test_sampler(self):
        for orth in (False, True):
            self.p.mirrored = None
            self.p.orthogonal = orth
            sampler = self.p.get_sampler()
            self.assertEqual(sampler.__name__ == 'orthogonal_sampling', orth) 
            self.p.mirrored = 'mirrored' 
            sampler = self.p.get_sampler() 
            self.assertEqual(sampler.__name__, 'mirrored_sampling')
            self.p.mirrored = 'mirrored pairwise'
            self.assertEqual(sampler.__name__, 'mirrored_sampling')

    def test_wrong_parameters(self):
        with self.assertWarns(RuntimeWarning):
            Parameters(1, mu=3, lambda_=2)

    def test_options(self):
        for module in Parameters.__modules__:
            m = getattr(Parameters, module)
            if type(m) == AnyOf:
                for o in m.options:
                    setattr(self.p, module, o)
                    Parameters(1, **{module:o})

    def step(self):     
        y = np.random.rand(self.p.lambda_, self.p.d).T 
        x = self.p.m.reshape(-1,1) * y
        f = np.array(list(map(sum, x)))
        self.p.used_budget += self.p.lambda_
        self.p.population = Population(x, y, f)
        self.p.m_old = self.p.m.copy()
        self.p.m *= np.linalg.norm(y, axis=1).reshape(-1, 1)
        self.p.adapt()
        self.p.old_population = self.p.population.copy()
    
    def set_parameter_and_step(self, pname, value, nstep=2, warning_action="default"):
        setattr(self.p, pname, value)
        with warnings.catch_warnings():
            warnings.simplefilter(warning_action)
            for _ in range(nstep):
                self.step()

    def test_tpa(self):
        self.p.rank_tpa = .3
        self.set_parameter_and_step('step_size_adaptation', 'tpa')

    def test_msr(self):
        self.set_parameter_and_step('step_size_adaptation', 'msr')
    
    def test_active(self):
        self.set_parameter_and_step('active', True)
    
    def test_reset(self):
        self.p.C[0][0] = np.inf
        self.step()

    def test_warning(self):
        self.p.compute_termination_criteria = True
        self.set_parameter_and_step('max_iter', True, 5, 'ignore')
    
    def test_threshold(self):
        self.step()
        self.assertEqual(type(self.p.threshold), np.float64)

    
    def test_from_arary(self):
        c_array = [0] * 11

        with self.assertRaises(AttributeError):
            _c_array = c_array[1:].copy()
            p = Parameters.from_config_array(5, _c_array)

        with self.assertRaises(AttributeError):
            _c_array = c_array + [0]
            p = Parameters.from_config_array(5, _c_array)

        with self.assertRaises(AttributeError):
            _c_array = c_array.copy()
            _c_array[0] = 2
            p = Parameters.from_config_array(5, _c_array)
    
        p = Parameters.from_config_array(5, c_array)

    
    def test_save_load(self):
        tmpfile = os.path.join(os.path.dirname(__file__), 'tmp.pkl')
        self.p.save(tmpfile)
        p = Parameters.load(tmpfile)
        os.remove(tmpfile)
        with self.assertRaises(OSError):
            self.p.load("__________")
        
        
        with open(tmpfile, "wb") as f:
            pickle.dump({}, f)
        with self.assertRaises(AttributeError):
            self.p.load(tmpfile)
        os.remove(tmpfile)
             


if __name__ == '__main__':
    unittest.main()
