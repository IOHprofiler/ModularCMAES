"""Fill the generated parts of the documentation from the type stubs.

The guide pages show the options of a module as a table, taken from the docstrings of the enums in
modcma/c_maes/cmaescpp/options.pyi (which come from bindings/*.cpp). A table sits between

    <!-- options: EnumName -->
    ...
    <!-- /options -->

Run after regenerating the stubs (tools/gen_stubs.py):

    python tools/gen_docs.py          # rewrite the tables
    python tools/gen_docs.py --check  # exit 1 if a table is out of date (tests/test_c_stubs.py)
"""

import argparse
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
OPTIONS_STUB = os.path.join(ROOT, "modcma", "c_maes", "cmaescpp", "options.pyi")
REGION = re.compile(
    r"(<!-- options: (?P<name>[A-Za-z]+) -->\n)(?:.*?\n)?(<!-- /options -->)", re.S
)


def enum_tables(stub=OPTIONS_STUB):
    with open(stub, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    tables = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        rows = []
        for i, item in enumerate(node.body):
            if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
                following = node.body[i + 1] if i + 1 < len(node.body) else None
                doc = ""
                if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
                    doc = " ".join(str(following.value.value).split())
                doc = doc.replace("|", "\\|")
                rows.append(f"| `{item.targets[0].id}` | {doc} |")
        intro = " ".join((ast.get_docstring(node) or "").split())
        tables[node.name] = "\n".join(
            [f"**`modcma.options.{node.name}`**: {intro}", "", "| value | |", "|---|---|", *rows]
        )
    return tables


def render(text, tables):
    return REGION.sub(lambda m: f"{m.group(1)}{tables[m.group('name')]}\n{m.group(3)}", text)


def pages():
    for directory, _, names in os.walk(DOCS):
        for name in sorted(names):
            if name.endswith(".md"):
                yield os.path.join(directory, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="only check, do not write")
    args = parser.parse_args()
    tables = enum_tables()
    stale = []
    for path in pages():
        with open(path, encoding="utf-8") as f:
            text = f.read()
        new = render(text, tables)
        if new != text:
            stale.append(os.path.relpath(path, ROOT))
            if not args.check:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
    for path in stale:
        print(("out of date: " if args.check else "updated: ") + path)
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
