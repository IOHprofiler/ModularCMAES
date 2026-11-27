"""The counts of variants in the guide match the code (#130)."""

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


@unittest.skipUnless(
    (ROOT / "tools" / "count_variants.py").exists() and (ROOT / "docs").is_dir(),
    "the tool and the documentation are not available (the tests of a wheel)",
)
class TestVariantCount(unittest.TestCase):
    def test_guide_states_the_counts_of_the_code(self):
        """A new module or option changes the counts: run tools/count_variants.py, update the guide."""
        tool = load_tool()
        n = tool.count(check_samples=2000)
        self.assertEqual(n["paper"], 14_859_936)
        guide = (ROOT / "docs" / "guide" / "modules.md").read_text()
        section = guide[guide.index("## How many variants"):]
        for key in (
            "paper",
            "modules_14",
            "modules_14_combinations",
            "modules_14_repelling_none",
            "modules_13",
            "modules_13_combinations",
            "modules_13_repelling_none",
            "fields_combinations",
            "fields_after_settings",
            "fields_distinct",
        ):
            with self.subTest(key=key):
                self.assertIn(f"{n[key]:,}", section)
        self.assertIn(f"All {n['fields']} fields of `Modules`", section)
        self.assertEqual(len(tool.MODULES), 13)


if __name__ == "__main__":
    unittest.main()
