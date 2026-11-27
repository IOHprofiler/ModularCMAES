"""Count the distinct variants of modcma (#130); the numbers of docs/guide/modules.md.

A *variant* is a combination of module options with the numerical parameters at their defaults
(lambda > 1, mu = lambda / 2, or lambda / 4 for SA and CMSA). Two combinations are the same variant
when they cannot give different runs on any problem:

* `Settings` turns them into the same modules (its own adjustments, read back from
  `Settings.modules`: COVARIANCE becomes COVARIANCE_NO_EIGV unless ssa is CSA, UTILITY weights
  switch the active update on, NATURAL_GRADIENT with active switches to UTILITY, NONE switches
  active off); or
* they differ only in an option that has no effect there (docs/guide/compatibility.md, each rule
  checked in the code, see `RULES` below).

Equivalences that depend on the problem (X0 = CENTER without an x0, ZERO = CENTER for a symmetric
box, bound correction with an infinite box) are not applied. Equivalences that are not documented
are not detected: a count is the number of combinations not known to coincide.

The counts (docs/guide/modules.md, "How many variants"):

* `paper`: the framework paper's arithmetic, its 14 modules with its options and its two
  dependencies (active needs a matrix adaptation, repelling needs a restart strategy), for
  comparison: 14,859,936. The paper does not apply the adjustments of Settings.
* `modules`: the algorithmic modules of the current code (`MODULES`), with their algorithmic
  options, the other fields of `Modules` at their defaults; with or without bound correction as a
  module, and with repelling inactive (the paper's rule) or active under restart_strategy=NONE
  (which still restarts when UnableToAdapt fires).
* `fields`: every field of `Modules` with every value.

    python tools/count_variants.py
"""

import itertools
import math
import random

from modcma import c_maes

DIM = 10  # default lambda = 10, mu = 5; no module adjustment depends on the dimension

# The fields of Modules, in three groups that are counted separately (verified by
# `check_factorization`): A is adjusted by Settings, B are the restarts (not adjusted with lambda
# > 1), C act independently of all others.
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
GROUP_B = ("restart_strategy", "center_placement", "repelling_type", "repelling_metric")
GROUP_C = ("bound_correction", "orthogonal", "sample_transformation", "sampler", "threshold_convergence")

# The algorithmic modules: components of the search, whose options are alternative mechanisms
# from the literature and act on any problem. The values left out are not algorithmic options:
# COVARIANCE_NO_EIGV is COVARIANCE computed with a Cholesky factor (Settings switches to it), the
# NONE transformation is a testing option, STOP is run control (stop instead of continue).
MODULES = {
    "sampler": None,
    "sample_transformation": ("NONE",),
    "mirrored": None,
    "orthogonal": None,
    "threshold_convergence": None,
    "elitist": None,
    "sequential_selection": None,
    "weights": None,
    "ssa": None,
    "active": None,
    "matrix_adaptation": ("COVARIANCE_NO_EIGV",),
    "restart_strategy": ("STOP",),
    "repelling_type": None,
}
# A module in the framework paper; here counted both ways: it is constraint handling, which acts
# only on a bounded problem (as the integer variables, which are not a module)
BOUND_CORRECTION = "bound_correction"
# Settings of another module, at their defaults in the module counts: where a restart begins (of
# the restart strategy), the distance of repelling, per-individual step sizes (of the
# self-adaptive step-size rules)
OPTION_FIELDS = ("center_placement", "repelling_metric", "sample_sigma")

RULES = """\
sample_sigma is forced on for SA and LPXNES (src/mutation.cpp:434)
sequential selection with PAIRWISE stops at 2 mu at the earliest, the whole generation when
  2 mu >= lambda (include/mutation.hpp:58, src/mutation.cpp:43)
STOP never restarts: center placement and repelling have no effect (NONE restarts when
  UnableToAdapt fires; the conditions of get_configspace, #81)
repelling_metric acts only with repelling; CURRENT and ARCHIVED are EUCLIDEAN without a matrix
NOVELTY_WEIGHTED and MAXIMIN_TABOO are UNIFORM without repelling (an empty archive)"""


def options() -> dict:
    """The values of every field of Modules, as Settings accepts them (enum members or bools)."""
    return {k: list(v) for k, v in c_maes.get_all_module_options().items()}


def name(value) -> str:
    return str(value).split(".")[-1]


def check_factorization(opts: dict, samples: int = 20000, seed: int = 1) -> None:
    """Settings leaves groups B and C unchanged, and its adjustment of A depends on A only."""
    groups = set(GROUP_A + GROUP_B + GROUP_C)
    assert set(opts) == groups, f"fields of Modules not assigned to a group: {set(opts) ^ groups}"
    rng = random.Random(seed)
    for _ in range(samples):
        combo = {k: rng.choice(v) for k, v in opts.items()}
        s = c_maes.settings_from_dict(DIM, **combo)
        for k in GROUP_B + GROUP_C:
            assert name(getattr(s.modules, k)) == name(combo[k]), (k, combo)
        only_a = c_maes.settings_from_dict(DIM, **{k: combo[k] for k in GROUP_A})
        for k in GROUP_A:
            assert getattr(s.modules, k) == getattr(only_a.modules, k), (k, combo)


def classes_a(opts: dict, merge_no_eigv: bool) -> set:
    """The distinct combinations of group A: Settings, then the run-time rules."""
    seen = set()
    for values in itertools.product(*(opts[k] for k in GROUP_A)):
        s = c_maes.settings_from_dict(DIM, **dict(zip(GROUP_A, values)))
        m = {k: name(getattr(s.modules, k)) for k in GROUP_A}
        if merge_no_eigv and m["matrix_adaptation"] == "COVARIANCE_NO_EIGV":
            m["matrix_adaptation"] = "COVARIANCE"
        if m["ssa"] in ("SA", "LPXNES"):
            m["sample_sigma"] = "True"
        if m["mirrored"] == "PAIRWISE" and 2 * s.mu0 >= s.lambda0:
            m["sequential_selection"] = "False"
        seen.add(tuple(m[k] for k in GROUP_A))
    return seen


def classes_b(opts: dict, matrix_none: bool, repelling_with_none: bool = True) -> int:
    """The distinct (restart strategy, center placement, repelling type, repelling metric)."""
    seen = set()
    for r, c, t, m in itertools.product(*([name(o) for o in opts[k]] for k in GROUP_B)):
        if r == "STOP" or (r == "NONE" and not repelling_with_none):
            c, t, m = "-", "-", "-"
        if t in ("-", "NONE"):
            m = "-"
        elif matrix_none:
            m = "EUCLIDEAN"
        if t == "NONE" and c in ("NOVELTY_WEIGHTED", "MAXIMIN_TABOO"):
            c = "UNIFORM"
        seen.add((r, c, t, m))
    return len(seen)


def restrict(opts: dict, with_bounds: bool) -> dict:
    """The algorithmic modules with their algorithmic options; the other fields at the default."""
    default = c_maes.Modules()
    out = {}
    for k, v in opts.items():
        if k in MODULES or (with_bounds and k == BOUND_CORRECTION):
            left_out = MODULES.get(k) or ()
            out[k] = [o for o in v if name(o) not in left_out]
        else:
            out[k] = [getattr(default, k)]
    return out


def count_distinct(opts: dict, merge_no_eigv: bool, repelling_with_none: bool) -> int:
    a = classes_a(opts, merge_no_eigv)
    b = {flag: classes_b(opts, flag, repelling_with_none) for flag in (False, True)}
    c = math.prod(len(opts[k]) for k in GROUP_C)
    i_matrix = GROUP_A.index("matrix_adaptation")
    return c * sum(b[x[i_matrix] == "NONE"] for x in a)


def count_paper() -> int:
    """The framework paper's 14 modules, options and two dependencies, as plain arithmetic."""
    sampling = 3 * 6 * 3 * 2 * 2 * 7  # sampler, transformation, mirrored, orthogonal, threshold, bounds
    selection = 2 * 2 * 3 * 9  # elitism, sequential selection, weights, ssa
    matrix = 7 * 2 - 1  # active has no effect with NONE
    restarts = 4 * 2 - 1  # repelling (on, off) has no effect with restart strategy None
    return sampling * selection * matrix * restarts


def count(check_samples: int = 20000) -> dict:
    """Every count of docs/guide/modules.md."""
    opts = options()
    check_factorization(opts, check_samples)
    n = {"paper": count_paper()}
    for bounds, label in ((True, "14"), (False, "13")):
        r = restrict(opts, bounds)
        n[f"modules_{label}"] = count_distinct(r, merge_no_eigv=True, repelling_with_none=False)
        n[f"modules_{label}_repelling_none"] = count_distinct(
            r, merge_no_eigv=True, repelling_with_none=True
        )
        n[f"modules_{label}_combinations"] = math.prod(len(v) for v in r.values())
    after_settings = {
        tuple(name(getattr(c_maes.settings_from_dict(DIM, **dict(zip(GROUP_A, v))).modules, k))
              for k in GROUP_A)
        for v in itertools.product(*(opts[k] for k in GROUP_A))
    }
    n["fields"] = len(opts)
    n["fields_combinations"] = math.prod(len(v) for v in opts.values())
    n["fields_after_settings"] = (
        len(after_settings)
        * math.prod(len(opts[k]) for k in GROUP_B)
        * math.prod(len(opts[k]) for k in GROUP_C)
    )
    n["fields_distinct"] = count_distinct(opts, merge_no_eigv=False, repelling_with_none=True)
    return n


def main() -> None:
    n = count()
    rows = (
        ("the framework paper: 14 modules, its options and dependencies", "paper"),
        ("14 modules (13 + bound correction): combinations", "modules_14_combinations"),
        ("14 modules: distinct variants", "modules_14"),
        ("  the same, repelling active with restart_strategy=NONE", "modules_14_repelling_none"),
        ("13 modules (bound correction an option): combinations", "modules_13_combinations"),
        ("13 modules: distinct variants", "modules_13"),
        ("  the same, repelling active with restart_strategy=NONE", "modules_13_repelling_none"),
        (f"all {n['fields']} fields of Modules: combinations", "fields_combinations"),
        ("  after the adjustments of Settings", "fields_after_settings"),
        ("  distinct", "fields_distinct"),
    )
    for label, key in rows:
        print(f"{label:<64s} {n[key]:>15,}")
    print("\nrun-time rules applied besides the adjustments of Settings:\n" + RULES)


if __name__ == "__main__":
    main()
