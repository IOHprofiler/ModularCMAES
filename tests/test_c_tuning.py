import unittest
from modcma import c_maes


class TestTuning(unittest.TestCase):
    def test_module_options(self):
        self.assertIsInstance(c_maes.get_all_module_options(), dict)

if __name__ == "__main__":
    unittest.main()