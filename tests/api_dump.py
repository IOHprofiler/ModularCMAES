"""API dump of the C++ extension ``modcma.c_maes.cmaescpp`` (TASKS.md, T1.1).

Records every submodule, class, enum, function, method and property with, for every overload, the
argument names and which arguments have defaults (and the default's text). The format does not
depend on the binding library: signatures are parsed from the docstrings that pybind11 and nanobind
generate (``name(arg: type = default, ...) -> type``), so a dump made with pybind11 can be compared
with the extension after the nanobind port.

    python -m tests.api_dump write [--output tests/data/api/pybind11.json]
    python -m tests.api_dump compare tests/data/api/pybind11.json

``compare`` reports what the installed extension lacks compared with the reference: a missing name,
a property that became read-only, a reference overload that no current overload accepts (same
argument names in the same order, the same arguments optional). Additions are allowed.
"""

import argparse
import enum
import json
import os
import re
import sys
import types

from modcma.c_maes import cmaescpp

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "data", "api", "pybind11.json")


def split_top_level(text: str) -> list:
    parts, depth, current = [], 0, []
    for char in text:
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def parse_signatures(name: str, doc: str) -> list:
    """All overloads of ``name`` in a pybind11/nanobind docstring, as lists of [arg, default]."""
    signatures = []
    for line in (doc or "").splitlines():
        line = re.sub(r"^\s*\d+\.\s*", "", line.strip())
        if not line.startswith(name + "("):
            continue
        depth, end = 0, None
        for i, char in enumerate(line[len(name):], start=len(name)):
            depth += char == "("
            depth -= char == ")"
            if depth == 0:
                end = i
                break
        if end is None:
            continue
        args = []
        for part in split_top_level(line[len(name) + 1 : end]):
            part = part.strip()
            if part in ("*", "/", "") or part.startswith(("*args", "**kwargs")):
                continue
            arg_name = re.split(r"[:=]", part, maxsplit=1)[0].strip()
            default = None
            if "=" in part:
                default = re.sub(r"\s+", "", part.split("=", 1)[1])
            if arg_name == "self":
                continue
            args.append([arg_name, default])
        signatures.append(args)
    return signatures


def describe(obj, name: str):
    if isinstance(obj, type) and issubclass(obj, enum.Enum):
        return {"kind": "enum", "members": [m.name for m in obj]}
    if isinstance(obj, property) or type(obj).__name__ in ("property", "nb_property"):
        return {"kind": "property", "readonly": getattr(obj, "fset", None) is None}
    if callable(obj) and not isinstance(obj, type):
        return {"kind": "function", "overloads": parse_signatures(name, getattr(obj, "__doc__", ""))}
    return {"kind": type(obj).__name__}


def dump_class(cls) -> dict:
    members = {}
    for name, value in sorted(vars(cls).items()):
        if name.startswith("_") and name not in ("__init__", "__call__", "__add__", "__repr__"):
            continue
        if name == "__init__" and (getattr(value, "__doc__", "") or "").startswith("Initialize self"):
            continue  # no constructor bound: object.__init__
        members[name] = describe(value, name)
    bases = [b.__name__ for b in cls.__bases__ if b is not object]
    return {"kind": "class", "bases": bases, "members": members}


def dump_module(module, prefix: str) -> dict:
    result = {}
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        value = getattr(module, name)
        qualified = f"{prefix}.{name}"
        if isinstance(value, types.ModuleType):
            if value.__name__.startswith(cmaescpp.__name__):
                result.update(dump_module(value, qualified))
            continue
        if isinstance(value, type) and issubclass(value, enum.Enum):
            result[qualified] = describe(value, name)
        elif isinstance(value, type):
            result[qualified] = dump_class(value)
        else:
            result[qualified] = describe(value, name)
    return result


def dump() -> dict:
    return dump_module(cmaescpp, "cmaescpp")


def accepts(current: list, reference: list) -> bool:
    """Whether the current overload accepts every call the reference overload accepts."""
    cur_names = [a for a, _ in current]
    ref_names = [a for a, _ in reference]
    if cur_names[: len(ref_names)] != ref_names:
        return False
    # extra current arguments must be optional; optional reference arguments must stay optional
    if any(d is None for _, d in current[len(ref_names) :]):
        return False
    return all(not (r is not None and c is None) for (_, r), (_, c) in zip(reference, current))


def compare(reference: dict, current: dict) -> list:
    problems = []
    for name, ref in reference.items():
        cur = current.get(name)
        if cur is None:
            problems.append(f"missing: {name}")
            continue
        problems += compare_entry(name, ref, cur)
        if ref["kind"] == "class":
            for member, ref_member in ref["members"].items():
                cur_member = cur.get("members", {}).get(member)
                if cur_member is None:
                    problems.append(f"missing: {name}.{member}")
                else:
                    problems += compare_entry(f"{name}.{member}", ref_member, cur_member)
    return problems


def compare_entry(name: str, ref: dict, cur: dict) -> list:
    if ref["kind"] != cur["kind"]:
        return [f"kind changed: {name}: {ref['kind']} -> {cur['kind']}"]
    if ref["kind"] == "enum" and set(ref["members"]) - set(cur["members"]):
        return [f"enum members missing: {name}: {sorted(set(ref['members']) - set(cur['members']))}"]
    if ref["kind"] == "property" and cur["readonly"] and not ref["readonly"]:
        return [f"became read-only: {name}"]
    if ref["kind"] == "function":
        return [
            f"overload not accepted: {name}({', '.join(a + ('=' + d if d else '') for a, d in overload)})"
            for overload in ref["overloads"]
            if not any(accepts(c, overload) for c in cur["overloads"])
        ]
    return []


def main():
    parser = argparse.ArgumentParser(description="API dump of modcma.c_maes.cmaescpp")
    parser.add_argument("command", choices=("write", "compare"))
    parser.add_argument("reference", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "write":
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(dump(), f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.output}")
    else:
        with open(args.reference) as f:
            problems = compare(json.load(f), dump())
        for problem in problems:
            print(problem)
        print(f"{len(problems)} incompatibilities with {args.reference}")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
