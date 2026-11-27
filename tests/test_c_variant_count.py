"""The count of distinct module configurations in the guide matches the code (#130)."""

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location("count_variants", ROOT / "tools" / "count_variants.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVariantCount(unittest.TestCase):
    def test_guide_states_the_count_of_the_code(self):
        """A new module or option changes the count: rerun tools/count_variants.py, update the docs."""
        n = load_tool().count(check_samples=2000)
        modules = (ROOT / "docs" / "guide" / "modules.md").read_text()
        tuning = (ROOT / "docs" / "guide" / "tuning.md").read_text()
        self.assertIn(f"{n['all']:,} combinations", modules)
        self.assertIn(f"{n['distinct']:,} distinct variants", modules)
        self.assertIn(f"{n['distinct']:,}", tuning)
        self.assertIn(f"{len(n['options'])} modules", modules)


if __name__ == "__main__":
    unittest.main()
