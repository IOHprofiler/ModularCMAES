import unittest
from copy import deepcopy
from modcma import c_maes
from ConfigSpace import ConfigurationSpace

class TestTuning(unittest.TestCase):
    def test_module_options(self):
        options = c_maes.get_all_module_options()
        self.assertIsInstance(options, dict)
        self.assertEqual(options, options | {"active": (False, True)})
        self.assertEqual(options, options | {"elitist": (False, True)})
        self.assertEqual(options, options | {"orthogonal": (False, True)})
        self.assertEqual(options, options | {"sequential_selection": (False, True)})
        self.assertEqual(options, options | {"threshold_convergence": (False, True)})
        self.assertEqual(options, options | {"sample_sigma": (False, True)})
        self.assertEqual(options, options | {"repelling_restart": (False, True)})


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

if __name__ == "__main__":
    unittest.main()