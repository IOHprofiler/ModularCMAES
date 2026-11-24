"""The code in the documentation runs.

Every ```python block of a page in docs/ (and of README.md) is executed, the blocks of one page in
order in one namespace. A block right after the comment <!-- no-test --> is skipped, and so is a
block that includes a file (pymdownx.snippets: --8<--); the example scripts in examples/ are run as
programs instead. Skipped when the documentation is not there (the tests of a wheel).
"""

import contextlib
import io
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
EXAMPLES = os.path.join(ROOT, "examples")
BLOCK = re.compile(r"(<!--\s*no-test\s*-->\s*\n)?^```python[^\n]*\n(.*?)^```", re.M | re.S)


def pages():
    found = [os.path.join(ROOT, "README.md")]
    for directory, _, names in os.walk(DOCS):
        found += [os.path.join(directory, n) for n in sorted(names) if n.endswith(".md")]
    return found


def blocks(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for match in BLOCK.finditer(text):
        skip, code = match.groups()
        if skip or "--8<--" in code:
            continue
        yield text.count("\n", 0, match.start()) + 2, code


@unittest.skipUnless(os.path.isdir(DOCS), "documentation not available")
class TestDocs(unittest.TestCase):
    def test_pages(self):
        for path in pages():
            namespace = {"__name__": "__docs__"}
            for line, code in blocks(path):
                with self.subTest(page=os.path.relpath(path, ROOT), line=line):
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(code, f"{path}:{line}", "exec"), namespace)  # noqa: S102

    def test_examples(self):
        for name in sorted(os.listdir(EXAMPLES)) if os.path.isdir(EXAMPLES) else []:
            if not name.endswith(".py"):
                continue
            with self.subTest(example=name):
                result = subprocess.run(
                    [sys.executable, os.path.join(EXAMPLES, name)],
                    capture_output=True,
                    text=True,
                    cwd=EXAMPLES,
                    env={**os.environ, "MODCMA_EXAMPLE_QUICK": "1"},
                    timeout=600,
                )
                if result.returncode != 0 and "pip install" in result.stderr[-200:]:
                    self.skipTest(result.stderr.strip())  # an optional dependency is missing
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
