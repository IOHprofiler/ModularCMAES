"""Count the distinct module configurations of modcma (#130).

The framework paper counts the "distinct variants with the default parameters", leaving out the
combinations where an option has no effect. This script recomputes that count from the code:

* every option of every module of `Modules` (`get_all_module_options`);
* two configurations are the same variant when `Settings` turns them into the same modules (its
  own adjustments, read back from `Settings.modules`), or when they differ only in an option that
  has no effect there (the rules below, from docs/guide/compatibility.md, each checked in the code);
* the numerical parameters keep their defaults (lambda > 1, mu = lambda / 2, or lambda / 4 for SA
  and CMSA); equivalences that depend on the problem (x0, a symmetric or infinite box) are not
  applied, so `X0`, `ZERO` and `CENTER` count as different placements and every bound
  correction counts.

Equivalences that are not documented are not detected: the count is the number of
configurations that are not known to coincide.

    python tools/count_variants.py
"""

import itertools
import math
import random

from modcma import c_maes

DIM = 10  # default lambda = 10, mu = 5; no module adjustment depends on the dimension

# The modules that Settings adjusts together (include/settings.hpp) and those that run-time rules
# couple to them; enumerated through Settings
GROUP_A = (
    "active",
    "matrix_adaptation",
    "ssa",
    "weights",
    "sample_sigma",
    "elitist",
    "sequential_selection",
    "mirrored",
)
# Restarts: not adjusted by Settings with lambda > 1; the no-effect rules below
GROUP_B = ("restart_strategy", "center_placement", "repelling_type", "repelling_metric")
# Independent of the others: every option acts in every configuration (verified below that
# Settings leaves them unchanged)
GROUP_C = ("bound_correction", "orthogonal", "sample_transformation", "sampler", "threshold_convergence")


def options() -> dict:
    """The options of every module, as Settings accepts them (enum members, or True/False)."""
    return {k: list(v) for k, v in c_maes.get_all_module_options().items()}


def name(value) -> str:
    return str(value).split(".")[-1]


def canonical_a(combo: dict):
    """The modules of group A as Settings makes them, then the run-time no-effect rules."""
    s = c_maes.settings_from_dict(DIM, **combo)
    m = {k: name(getattr(s.modules, k)) for k in GROUP_A}
    # sample_sigma is forced on for SA and LPXNES (src/mutation.cpp:434)
    if m["ssa"] in ("SA", "LPXNES"):
        m["sample_sigma"] = "True"
    # Sequential selection with PAIRWISE stops at 2 mu at the earliest (include/mutation.hpp:58,
    # src/mutation.cpp:43): the whole generation when 2 mu >= lambda
    if m["mirrored"] == "PAIRWISE" and 2 * s.mu0 >= s.lambda0:
        m["sequential_selection"] = "False"
    return tuple(m[k] for k in GROUP_A)


def count_b(opts: dict, matrix_none: bool) -> int:
    """Distinct (restart, center placement, repelling type, repelling metric)."""
    seen = set()
    for r, c, t, m in itertools.product(*([name(o) for o in opts[k]] for k in GROUP_B)):
        # STOP never restarts: the placement and repelling have no effect (NONE still restarts
        # when UnableToAdapt fires; modcma/c_maes/__init__.py, the conditions of #81)
        if r == "STOP":
            c, t, m = "-", "-", "-"
        # the metric acts only with repelling; CURRENT and ARCHIVED are EUCLIDEAN without a matrix
        if t in ("-", "NONE"):
            m = "-"
        elif matrix_none:
            m = "EUCLIDEAN"
        # NOVELTY_WEIGHTED and MAXIMIN_TABOO are UNIFORM with an empty archive (no repelling)
        if t == "NONE" and c in ("NOVELTY_WEIGHTED", "MAXIMIN_TABOO"):
            c = "UNIFORM"
        seen.add((r, c, t, m))
    return len(seen)


def check_factorization(opts: dict, samples: int = 20000, seed: int = 1) -> None:
    """Settings leaves groups B and C unchanged, and its adjustment of A depends on A only."""
    rng = random.Random(seed)
    names = list(opts)
    for _ in range(samples):
        combo = {k: rng.choice(opts[k]) for k in names}
        s = c_maes.settings_from_dict(DIM, **combo)
        for k in GROUP_B + GROUP_C:
            assert name(getattr(s.modules, k)) == name(combo[k]), (k, combo)
        only_a = c_maes.settings_from_dict(DIM, **{k: combo[k] for k in GROUP_A})
        for k in GROUP_A:
            assert getattr(s.modules, k) == getattr(only_a.modules, k), (k, combo)


def count(check_samples: int = 20000) -> dict:
    """The counts: all combinations, after the adjustments of Settings, and distinct."""
    opts = options()
    groups = set(GROUP_A + GROUP_B + GROUP_C)
    assert set(opts) == groups, f"modules not assigned to a group: {set(opts) ^ groups}"
    check_factorization(opts, check_samples)

    classes_a, after_settings_a = set(), set()
    for values in itertools.product(*(opts[k] for k in GROUP_A)):
        combo = dict(zip(GROUP_A, values))
        s = c_maes.settings_from_dict(DIM, **combo)
        after_settings_a.add(tuple(name(getattr(s.modules, k)) for k in GROUP_A))
        classes_a.add(canonical_a(combo))

    b = {flag: count_b(opts, flag) for flag in (False, True)}
    c = math.prod(len(opts[k]) for k in GROUP_C)
    i_matrix = GROUP_A.index("matrix_adaptation")
    raw_b = math.prod(len(opts[k]) for k in GROUP_B)
    return {
        "options": {k: len(v) for k, v in opts.items()},
        "all": math.prod(len(v) for v in opts.values()),
        "after_settings": len(after_settings_a) * raw_b * c,
        "distinct": c * sum(b[a[i_matrix] == "NONE"] for a in classes_a),
        "group_a": (len(classes_a), math.prod(len(opts[k]) for k in GROUP_A)),
        "group_b": (b[False], b[True], raw_b),
        "group_c": c,
    }


def main() -> None:
    n = count()
    print(f"modules: {len(n['options'])}; options: "
          + ", ".join(f"{k} {v}" for k, v in n["options"].items()))
    print(f"all combinations of the options:            {n['all']:>15,}")
    print(f"after the adjustments of Settings:          {n['after_settings']:>15,}")
    print(f"distinct (also without no-effect options):  {n['distinct']:>15,}")
    a, b, c = n["group_a"], n["group_b"], n["group_c"]
    print(f"  group A: {a[0]} classes of {a[1]}; group B: {b[0]} (with a matrix), "
          f"{b[1]} (matrix NONE) of {b[2]}; group C: {c}")


if __name__ == "__main__":
    main()
