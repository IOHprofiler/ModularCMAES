import unittest
from copy import deepcopy
from modcma import c_maes

from ConfigSpace import ConfigurationSpace

def sphere(x):
    return sum(xi**2 for xi in x)

class TestTuning(unittest.TestCase):
    def test_module_options(self):
        options = c_maes.get_all_module_options()
        self.assertIsInstance(options, dict)
        self.assertEqual(options, {**options, "active": (False, True)})
        self.assertEqual(options, {**options, "elitist": (False, True)})
        self.assertEqual(options, {**options, "orthogonal": (False, True)})
        self.assertEqual(options, {**options, "sequential_selection": (False, True)})
        self.assertEqual(options, {**options, "threshold_convergence": (False, True)})
        self.assertEqual(options, {**options, "sample_sigma": (False, True)})
        self.assertEqual(options, {**options, "repelling_restart": (False, True)})


    def test_configspace(self):
        cspace = c_maes.get_configspace(2)
        default = cspace.get_default_configuration()
        self.assertIsInstance(cspace, ConfigurationSpace)
        settings = c_maes.settings_from_config(2, default)
        self.assertIsInstance(settings, c_maes.Settings)
        self.assertEqual(settings.cc, None)
        self.assertEqual(settings.modules.sampler, c_maes.options.BaseSampler.UNIFORM)
        
        changed = deepcopy(default)
        changed['cc'] = 0.1
        changed['sampler'] = "HALTON"
        settings_changed = c_maes.settings_from_config(2, changed)
        self.assertEqual(settings_changed.cc, 0.1)
        self.assertEqual(settings_changed.modules.sampler, c_maes.options.BaseSampler.HALTON)
        
    def test_from_dict(self):
        settings = c_maes.settings_from_dict(2, active=True, cc=1)
        self.assertEqual(settings.modules.active, True)
        self.assertEqual(settings.cc, 1)
        
        
    def test_fmin(self):
        xopt, fopt, evals, es = c_maes.fmin(sphere, [1, 2], 0.2, 100, active=True, matrix_adaptation='NONE')    
        self.assertLess(fopt, 1e-4)
        self.assertLessEqual(evals, 100)
        self.assertEqual(sphere(xopt), fopt)
        

if __name__ == "__main__":
    unittest.main()
    