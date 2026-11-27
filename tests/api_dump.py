"""API dump of the C++ extension ``modcma.c_maes.cmaescpp`` (dev/overhaul-2.0.md, T1.1).

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

#: Differences from the pybind11 reference that are accepted, with the reason
ACCEPTED = {
    "overload not accepted: cmaescpp.restart.CriteriaVector.pop(i)":
        "nanobind's bind_vector names the argument index (pop(i=...) by keyword is not supported)",
    "missing: cmaescpp.matrix_adaptation.CovarianceAdaptation.inv_root_C":
        "internal: moved to Adaptation.metric().inv_root_C(), for every matrix adaptation (#123)",
    # Internals (#80): BIPOP's regimes count the evaluations they used (the budgets left wrapped)
    "missing: cmaescpp.restart.BIPOP.budget": "internal: the regimes no longer split the budget",
    "missing: cmaescpp.restart.BIPOP.budget_large": "internal: now BIPOP.used_large",
    "missing: cmaescpp.restart.BIPOP.budget_small": "internal: now BIPOP.used_small",
    "missing: cmaescpp.restart.BIPOP.mu_factor":
        "internal: mu after a restart follows the new lambda, as in IPOP (#150)",
    "became read-only: cmaescpp.matrix_adaptation.NaturalGradientAdaptation.G":
        "WP4: the xNES update uses the low-rank factor of G (gradient_samples, gradient_weights, "
        "gradient_shift); the property computes G from it",
    # 2.0: the utils submodule is gone. set_seed, build_info and compute_ert are in the main namespace;
    # modcma.c_maes.utils (Python) forwards them with a DeprecationWarning; the rest was removed
    # or moved to the private _internal (tests only)
    "missing: cmaescpp.utils.set_seed": "now cmaescpp.set_seed; deprecated alias modcma.c_maes.utils",
    "missing: cmaescpp.utils.build_info": "now cmaescpp.build_info; deprecated alias modcma.c_maes.utils",
    "missing: cmaescpp.utils.cdf": "removed (scipy.stats.norm.cdf)",
    "missing: cmaescpp.utils.ppf": "removed (scipy.stats.norm.ppf)",
    "missing: cmaescpp.utils.random_uniform": "removed: the global generator only seeds optimizers",
    "missing: cmaescpp.utils.random_normal": "removed: the global generator only seeds optimizers",
    "missing: cmaescpp.utils.compute_ert": "now cmaescpp.compute_ert; deprecated alias modcma.c_maes.utils",
    "missing: cmaescpp.utils.i8_sobol": "internal: cmaescpp._internal",
    "missing: cmaescpp.utils.Shuffler": "internal: cmaescpp._internal",
    "missing: cmaescpp.utils.CachedShuffleSequence": "internal: cmaescpp._internal",
    "missing: cmaescpp.sampling.Sobol.cache": "internal: the points of the Sobol sampler",
    # Internals (#78): threshold convergence works on whole generations
    "missing: cmaescpp.mutation.ThresholdConvergence.scale":
        "internal: replaced by threshold() and apply(); the threshold is computed per generation",
    # The misspelled names of 1.x warn (DeprecationWarning) since #82
    "missing: cmaescpp.selection.Elitsm":
        "deprecated: still available through the module's __getattr__ (now selection.Elitism)",
    "missing: cmaescpp.selection.NoElitsm":
        "deprecated: still available through the module's __getattr__ (now selection.NoElitism)",
    "overload not accepted: cmaescpp.mutation.SequentialSelection.__init__(mirror, seq_cuttoff_factor=1.0)":
        "the keyword is seq_cutoff_factor; seq_cuttoff_factor= still works (a keyword-only "
        "overload with a DeprecationWarning); positional calls are unchanged",
    "overload not accepted: cmaescpp.mutation.NoSequentialSelection.__init__(mirror, seq_cuttoff_factor=1.0)":
        "as SequentialSelection: the keyword is seq_cutoff_factor, the old one is deprecated",
    "missing: cmaescpp.mutation.Strategy.mutate":
        "internal: ModularCMAES.advance samples and queues the population (#84)",
    "missing: cmaescpp.repelling.hill_valley_test":
        "internal: the optimizer evaluates the hill-valley points itself, one at a time (#84)",
    "missing: cmaescpp.repelling.TabooPoint.shares_basin": "internal: as hill_valley_test (#84)",
    "missing: cmaescpp.repelling.Repelling.update_archive":
        "internal: the optimizer's restart step runs the archive update (#84)",
    "missing: cmaescpp.Parameters.start":
        "internal: ModularCMAES.advance begins a generation and runs a restart (#84)",
    "missing: cmaescpp.Parameters.perform_restart":
        "internal: a restart is evaluated only by the optimizer (#84); a restart criterion that "
        "is met starts one at the next generation",
    # Internals: unused parameters and members removed (#84, third review)
    "missing: cmaescpp.matrix_adaptation.Adaptation.expected_length_z":
        "internal: not used by the adaptations; Weights.expected_length_z is the one CSA uses",
    "overload not accepted: cmaescpp.matrix_adaptation.Adaptation.adapt_matrix(weights, modules, population, mu, settings, stats)":
        "internal: modules is settings.modules",
    "overload not accepted: cmaescpp.matrix_adaptation.Adaptation.restart(settings, sigma)":
        "internal: sigma was not used",
    "overload not accepted: cmaescpp.matrix_adaptation.NaturalGradientAdaptation.compute_gradients(pop, weights, stats, settings, mu, lamb)":
        "internal: stats, mu and lamb were not used",
    "overload not accepted: cmaescpp.parameters.CoordinateMapping.__init__(arg0, arg1, arg2)":
        "internal: mueff was not used",
    "overload not accepted: cmaescpp.matrix_adaptation.Adaptation.adapt_evolution_paths(pop, weights, stats, settings, mu, lamb)":
        "internal: lamb was not used",
    "overload not accepted: cmaescpp.mutation.Strategy.adapt(weights, dynamic, population, old_population, stats, lamb)":
        "internal: lamb was not used",
}
# #136: derived from lb and ub, which recompute them; a value set directly was inconsistent with
# the bounds (db is the width the bound corrections use, center the default x0)
for name in ("db", "center", "diameter", "has_bounds"):
    ACCEPTED[f"became read-only: cmaescpp.parameters.Settings.{name}"] = (
        "derived from lb and ub (#136): set lb or ub; x0 sets the starting point")
for name in ("Cholesky", "Covariance", "Matrix", "Self", "Separable"):
    ACCEPTED[f"overload not accepted: cmaescpp.matrix_adaptation.{name}Adaptation.__init__"
             "(dimension, x0, expected_length_z)"] = "internal: expected_length_z was not used"
ACCEPTED["overload not accepted: cmaescpp.matrix_adaptation.NoAdaptation.__init__"
         "(dimension, x0, expected_length_z)"] = "internal: expected_length_z was not used"
ACCEPTED["overload not accepted: cmaescpp.matrix_adaptation.NaturalGradientAdaptation.__init__"
         "(dimension, x0, expected_length_z, sigma)"] = (
    "internal: expected_length_z and sigma were not used")


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
    """All overloads of ``name`` in a pybind11/nanobind docstring, as lists of
    [arg, default, kind] with kind "positional" (before a ``/``), "keyword" (after a ``*``) or ""
    (either). The generic ``name(*args, **kwargs)`` line of overloaded functions is skipped."""
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
        args, generic, keyword_only = [], False, False
        for part in split_top_level(line[len(name) + 1 : end]):
            part = part.strip()
            if part.startswith(("*args", "**kwargs")):
                generic = True
                continue
            if part == "/":
                for arg in args:
                    arg[2] = "positional"
                continue
            if part == "*":
                keyword_only = True
                continue
            if not part:
                continue
            arg_name = re.split(r"[:=]", part, maxsplit=1)[0].strip()
            default = None
            if "=" in part:
                default = re.sub(r"\s+", "", part.split("=", 1)[1])
            if arg_name == "self":
                continue
            args.append([arg_name, default, "keyword" if keyword_only else ""])
        if not generic:
            signatures.append(args)
    return signatures


def describe(obj, name: str):
    if isinstance(obj, type) and issubclass(obj, enum.Enum):
        return {"kind": "enum", "members": [m.name for m in obj]}
    if isinstance(obj, property) or type(obj).__name__ in ("property", "nb_property"):
        return {"kind": "property", "readonly": getattr(obj, "fset", None) is None}
    if callable(obj) and not isinstance(obj, type):
        # An alias (an old name bound to the same function) documents its signature under the
        # function's own name
        own = getattr(obj, "__name__", name)
        return {"kind": "function", "overloads": parse_signatures(own, getattr(obj, "__doc__", ""))}
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


def same_name(ref: list, cur: list) -> bool:
    # pybind11 names unnamed arguments arg0, arg1, ...: they can only be passed by position
    return ref[0] == cur[0] or re.fullmatch(r"arg\d+", ref[0]) is not None or cur[2] == "positional"


def accepts(current: list, reference: list) -> bool:
    """Whether the current overload accepts the calls of the reference overload: the same
    arguments (by position; keyword-only arguments by name, in any order), the optional ones
    still optional, any extra arguments optional."""
    positional = [a for a in current if a[2] != "keyword"]
    keyword = {a[0]: a for a in current if a[2] == "keyword"}
    matched = 0
    for i, ref in enumerate(reference):
        if i < len(positional) and same_name(ref, positional[i]):
            cur = positional[i]
            matched += 1
        elif ref[0] in keyword:
            cur = keyword.pop(ref[0])
        else:
            return False
        if ref[1] is not None and cur[1] is None:
            return False
    extra = positional[matched:] + list(keyword.values())
    return all(a[1] is not None for a in extra)


def notes(reference: dict, current: dict) -> list:
    """Accepted differences that change how a call can be written: an argument that became
    positional-only or keyword-only, or was renamed where it is positional-only."""
    out = []
    for name, ref in reference.items():
        cur = current.get(name, {})
        members = ref.get("members", {}) if ref["kind"] == "class" else {"": ref}
        for member, ref_member in members.items():
            cur_member = cur.get("members", {}).get(member) if member else cur
            if not cur_member or ref_member.get("kind") != "function":
                continue
            for overload in ref_member["overloads"]:
                for cur_overload in cur_member["overloads"]:
                    if accepts(cur_overload, overload):
                        keyword = [c[0] for c in cur_overload if c[2] == "keyword"]
                        changed = [
                            f"{r[0]}->{c[0]} (positional-only)" if c[2] == "positional" else f"{r[0]}->{c[0]}"
                            for r, c in zip(overload, cur_overload)
                            if c[2] != "keyword" and (c[2] == "positional" or r[0] != c[0])
                        ]
                        changed = [c for c in changed if not re.match(r"arg\d+->", c)]
                        if keyword:
                            changed.append("keyword-only: " + ", ".join(keyword))
                        if changed:
                            out.append(f"{name}{'.' + member if member else ''}: {', '.join(changed)}")
                        break
    return sorted(set(out))


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
            f"overload not accepted: {name}({', '.join(a[0] + ('=' + a[1] if a[1] else '') for a in overload)})"
            for overload in ref["overloads"]
            if not any(accepts(c, overload) for c in cur["overloads"])
        ]
    return []


def main():
    parser = argparse.ArgumentParser(description="API dump of modcma.c_maes.cmaescpp")
    parser.add_argument("command", choices=("write", "compare"))
    parser.add_argument("reference", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--notes", action="store_true", help="also list accepted call-style changes")
    args = parser.parse_args()
    if args.command == "write":
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(dump(), f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.output}")
    else:
        with open(args.reference) as f:
            reference = json.load(f)
        # dumps written before the kind field was recorded
        for entry in reference.values():
            members = entry["members"].values() if entry["kind"] == "class" else []
            for member in [entry, *members]:
                for overload in member.get("overloads", []):
                    for arg in overload:
                        if len(arg) == 2:
                            arg.append("")
        current = dump()
        problems = [p for p in compare(reference, current) if p not in ACCEPTED]
        if args.notes:
            for note in notes(reference, current):
                print(f"note: {note}")
        for problem in problems:
            print(problem)
        print(f"{len(problems)} incompatibilities with {args.reference}")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
