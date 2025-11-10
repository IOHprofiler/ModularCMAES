import unittest
from modcma import c_maes


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

    def test_numeric_options(self):
        options = c_maes.get_all_numeric_options()
        self.assertIsInstance(options, dict)

    def test_configspace(self):
        cspace = c_maes.get_configspace()
        print(cspace)


if __name__ == "__main__":
    unittest.main()