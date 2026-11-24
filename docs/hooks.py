"""MkDocs hooks: pages that are files at the repository root, and the figures they show.

PERFORMANCE.md, REPRODUCIBILITY.md, CHANGELOG.md and CONTRIBUTING.md live at the root (GitHub shows
them there); the site shows the same files. A page in docs/ that contains only the marker
``<!-- root: NAME.md -->`` gets the content of that file, with its links rewritten: to the
corresponding page of the site, to a figure of the site, or to the file on GitHub.
"""

import ast
import os
import re

from mkdocs.structure.files import File

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITHUB = "https://github.com/IOHprofiler/ModularCMAES/blob/master/"
# Root files shown as pages of the site
PAGES = {
    "PERFORMANCE.md": "performance.md",
    "REPRODUCIBILITY.md": "reproducibility.md",
    "CHANGELOG.md": "changelog.md",
    "CONTRIBUTING.md": "contributing.md",
    "README.md": "index.md",
}
# Directories of figures the root pages show, copied into the site at the same path
FIGURES = ["scripts/performance/figures"]
MARKER = re.compile(r"<!--\s*root:\s*([A-Z]+\.md)\s*-->")
# <!-- options: EnumName --> becomes a table of the values of modcma.options.EnumName and their
# docstrings, read from the stubs (the reference has the same text, with its own anchors)
OPTIONS = re.compile(r"<!--\s*options:\s*([A-Za-z]+)\s*-->")
OPTIONS_STUB = os.path.join(ROOT, "modcma", "c_maes", "cmaescpp", "options.pyi")
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")


def on_files(files, config):
    for directory in FIGURES:
        for name in sorted(os.listdir(os.path.join(ROOT, directory))):
            if name.lower().endswith((".png", ".svg")):
                files.append(
                    File(
                        f"{directory}/{name}",
                        src_dir=ROOT,
                        dest_dir=config["site_dir"],
                        use_directory_urls=config["use_directory_urls"],
                    )
                )
    return files


def _rewrite(match, page_dir):
    prefix, target, suffix = match.groups()
    if re.match(r"^[a-z]+:|^#|^/", target):
        return match.group(0)
    path, _, anchor = target.partition("#")
    anchor = f"#{anchor}" if anchor else ""
    up = "../" * page_dir.count("/") if page_dir else ""
    if path in PAGES:
        return f"{prefix}{up}{PAGES[path]}{anchor}{suffix}"
    if any(path.startswith(directory + "/") for directory in FIGURES):
        return f"{prefix}{up}{path}{suffix}"
    return f"{prefix}{GITHUB}{path}{anchor}{suffix}"


def _enum_table(name):
    with open(OPTIONS_STUB, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            rows = []
            for i, item in enumerate(node.body):
                if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
                    doc = ""
                    if i + 1 < len(node.body) and isinstance(node.body[i + 1], ast.Expr):
                        doc = " ".join(str(node.body[i + 1].value.value).split())
                    doc = doc.replace("|", "\\|")
                    rows.append(f"| `{item.targets[0].id}` | {doc} |")
            intro = " ".join((ast.get_docstring(node) or "").split())
            return "\n".join(
                [f"**`modcma.options.{name}`**: {intro}", "", "| value | |", "|---|---|", *rows]
            )
    raise KeyError(f"no enum {name} in {OPTIONS_STUB}")


def on_page_markdown(markdown, page, config, files):
    markdown = OPTIONS.sub(lambda m: _enum_table(m.group(1)), markdown)
    match = MARKER.search(markdown)
    if match is None:
        return markdown
    with open(os.path.join(ROOT, match.group(1)), encoding="utf-8") as f:
        content = f.read()
    page_dir = os.path.dirname(page.file.src_uri)
    content = LINK.sub(lambda m: _rewrite(m, page_dir), content)
    return MARKER.sub(lambda _: content, markdown, count=1)
