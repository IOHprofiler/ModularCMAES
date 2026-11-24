"""Every public name of the extension has a docstring in its type stubs.

The docstrings are written in the bindings (bindings/*.cpp) and reach help(), editors and the API
reference of the documentation through the stubs (tools/gen_stubs.py). This test reads the stubs
committed in modcma/c_maes/cmaescpp/*.pyi.
"""

import ast
import os
import unittest

import modcma.c_maes

# The stubs next to the extension (the source tree, or an installed wheel)
STUBS = os.path.join(os.path.dirname(os.path.abspath(modcma.c_maes.__file__)), "cmaescpp")


def undocumented(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    missing = []

    def visit(body, prefix):
        for i, node in enumerate(body):
            if isinstance(node, ast.ClassDef):
                if not node.name.startswith("_") and ast.get_docstring(node) is None:
                    missing.append(prefix + node.name)
                visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                setter = any(
                    isinstance(d, ast.Attribute) and d.attr == "setter" for d in node.decorator_list
                )
                dunder = node.name.startswith("__") and node.name != "__init__"
                private = node.name.startswith("_") and not node.name.startswith("__")
                if not (setter or dunder or private) and ast.get_docstring(node) is None:
                    missing.append(prefix + node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                if isinstance(node.value, ast.Name):
                    continue  # an alias of a documented name (an old spelling)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                follows = i + 1 < len(body) and isinstance(body[i + 1], ast.Expr) and isinstance(
                    getattr(body[i + 1], "value", None), ast.Constant
                )
                missing.extend(prefix + n for n in names if not n.startswith("_") and not follows)

    visit(tree.body, "")
    return missing


@unittest.skipUnless(os.path.isdir(STUBS), "stubs not installed")
class TestStubs(unittest.TestCase):
    def test_every_public_name_has_a_docstring(self):
        for name in sorted(os.listdir(STUBS)):
            if name.endswith(".pyi"):
                with self.subTest(stub=name):
                    self.assertEqual(undocumented(os.path.join(STUBS, name)), [])


GEN_DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "gen_docs.py"
)


@unittest.skipUnless(os.path.isfile(GEN_DOCS), "documentation not available")
class TestDocsTables(unittest.TestCase):
    def test_option_tables_are_up_to_date(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, GEN_DOCS, "--check"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, "run python tools/gen_docs.py:\n" + result.stdout)


if __name__ == "__main__":
    unittest.main()
