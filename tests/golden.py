"""Golden trajectory fixtures for the C++ ModularCMAES (TASKS.md, T0.1).

A fixture holds, for a fixed seed, the trajectory of many configurations: every module option one at
a time, a pairwise covering set of all module options, and special cases (1+1, integer variables,
tight bounds, restarts, repelling, the global ``constants``). For each run it stores a checkpoint every
``CHECKPOINT_EVERY`` generations: a SHA-256 over the full state of all generations since the previous
checkpoint (``pop.X``, ``pop.f``, ``m``, ``sigma``, evaluations) plus a few readable numbers, so that a
mismatch shows where a trajectory starts to differ.

Trajectories are compared bitwise, so a fixture is only valid on the platform (OS, CPU, compiler,
libm) that produced it: ``std::shuffle``/``std::uniform_*_distribution`` differ between libc++,
libstdc++ and MSVC, and libm, FMA contraction and SIMD reductions change the last bits. Fixtures are
stored per platform in ``tests/data/golden/<platform>.json``; a platform without one is skipped.

Usage (each run executes in this process, so call it in a fresh interpreter, as the test does):
    python -m tests.golden write [--platform NAME]     # (re)generate the fixture of this platform
    python -m tests.golden check [--platform NAME]     # compare with the fixture, exit 1 on mismatch
    python -m tests.golden list                        # print the run ids
    --fixture PATH reads/writes another file (e.g. a baseline of the commit before a change)

Never regenerate a fixture to make a change pass: regenerate it from the commit *before* the change,
on the same machine, and only with the maintainer's approval (TASKS.md, section 3).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from typing import Callable, Dict, List

import numpy as np

from modcma import c_maes

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "golden")
CHECKPOINT_EVERY = 10
SEED = 42
#: The fields of a checkpoint row
CHECKPOINT_FIELDS = ("t", "evaluations", "sigma", "f0", "restarts", "digest")

# ---------------------------------------------------------------------------------------------------
# Objective functions: plain Python arithmetic, so that they give the same bits on every platform
# (apart from math.cos in rastrigin, which is libm's, as is everything the optimizer computes).
# ---------------------------------------------------------------------------------------------------


def sphere(x) -> float:
    return sum(float(xi) * float(xi) for xi in x)


def ellipsoid(x) -> float:
    n = len(x)
    if n == 1:
        return float(x[0]) ** 2
    return sum(
        10.0 ** (6.0 * i / (n - 1)) * float(xi) * float(xi) for i, xi in enumerate(x)
    )


def rosenbrock(x) -> float:
    x = [float(xi) for xi in x]
    return sum(
        100.0 * (x[i + 1] - x[i] * x[i]) ** 2 + (1.0 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


def rastrigin(x) -> float:
    return 10.0 * len(x) + sum(
        float(xi) * float(xi) - 10.0 * math.cos(2.0 * math.pi * float(xi)) for xi in x
    )


FUNCTIONS: Dict[str, Callable] = {
    "sphere": sphere,
    "ellipsoid": ellipsoid,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
}

# ---------------------------------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------------------------------


def module_options() -> Dict[str, tuple]:
    """Module name -> (default, other options...), as names/bools (stable, sorted by name)."""
    options = c_maes.get_all_module_options()
    return {name: tuple(options[name]) for name in sorted(options)}


def pairwise_configs(options: Dict[str, tuple]) -> List[Dict]:
    """A deterministic greedy pairwise covering set of all module options."""
    names = list(options)
    uncovered = set()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for va in range(len(options[a])):
                for vb in range(len(options[b])):
                    uncovered.add(((a, va), (b, vb)))

    def key(p, q):
        return (p, q) if names.index(p[0]) < names.index(q[0]) else (q, p)

    configs = []
    while uncovered:
        first, second = min(uncovered)
        chosen = {first[0]: first[1], second[0]: second[1]}
        for name in names:
            if name in chosen:
                continue
            best, best_gain = 0, -1
            for v in range(len(options[name])):
                gain = sum(
                    key((name, v), (other, ov)) in uncovered
                    for other, ov in chosen.items()
                )
                if gain > best_gain:
                    best, best_gain = v, gain
            chosen[name] = best
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                uncovered.discard(((a, chosen[a]), (b, chosen[b])))
        configs.append(
            {
                name: options[name][v]
                for name, v in chosen.items()
                if v != 0  # only the non-defaults, which keeps the fixture readable
            }
        )
    return configs


def make_runs() -> List[Dict]:
    """The list of runs of a fixture. Stored in the fixture itself, so changing this function only
    matters when a fixture is regenerated."""
    options = module_options()
    runs = []

    def add(run_id, modules=None, settings=None, constants=None, functions=None, dims=(5,), budget=None):
        for function in functions or ("ellipsoid", "rastrigin"):
            for dim in dims:
                runs.append(
                    {
                        "id": f"{run_id}/{function}/d{dim}",
                        "modules": modules or {},
                        "settings": settings or {},
                        "constants": constants or {},
                        "function": function,
                        "dim": dim,
                        "budget": budget or (400 * dim if function != "rastrigin" else 1200 * dim),
                    }
                )

    add("default", dims=(2, 5, 10), functions=("sphere", "ellipsoid", "rosenbrock", "rastrigin"))

    for name, values in options.items():
        for value in values[1:]:
            add(f"one/{name}={value}", modules={name: value})

    for i, config in enumerate(pairwise_configs(options)):
        add(f"pair/{i:03d}", modules=config, functions=("rastrigin",))

    special = [
        ("one_plus_one", {}, {"lambda0": 1}),
        ("one_plus_one_cholesky", {"matrix_adaptation": "CHOLESKY"}, {"lambda0": 1}),
        ("integer_noncontiguous", {}, {"integer_variables": [1, 3]}),
        ("integer_all", {}, {"integer_variables": [0, 1, 2, 3, 4]}),
        ("large_lambda", {"active": True}, {"lambda0": 40, "mu0": 10}),
        ("small_sigma", {}, {"sigma0": 1e-3}),
        ("x0_given", {}, {"x0": [1.0, -2.0, 0.5, 3.0, -4.0]}),
        ("tight_bounds_mirror", {"bound_correction": "MIRROR"}, {"lb": [-1.0] * 5, "ub": [0.5] * 5}),
        ("tight_bounds_cotn", {"bound_correction": "COTN"}, {"lb": [-1.0] * 5, "ub": [0.5] * 5}),
        ("tight_bounds_resample", {"bound_correction": "RESAMPLE"}, {"lb": [-1.0] * 5, "ub": [0.5] * 5}),
        ("restart_ipop_active", {"restart_strategy": "IPOP", "active": True}, {}),
        ("restart_bipop_elitist", {"restart_strategy": "BIPOP", "elitist": True}, {}),
        ("repelling_coverage", {"restart_strategy": "RESTART", "repelling_type": "COVERAGE"}, {}),
        ("repelling_adaptive", {"restart_strategy": "IPOP", "repelling_type": "ADAPTIVE"}, {}),
        ("repelling_matrix", {"restart_strategy": "RESTART", "repelling_type": "COVERAGE", "matrix_adaptation": "MATRIX"}, {}),
        ("repelling_current", {"restart_strategy": "RESTART", "repelling_type": "COVERAGE", "repelling_metric": "CURRENT"}, {}),
        ("repelling_archived", {"restart_strategy": "RESTART", "repelling_type": "COVERAGE", "repelling_metric": "ARCHIVED"}, {}),
        ("repelling_adaptive_archived_cholesky", {"restart_strategy": "IPOP", "repelling_type": "ADAPTIVE", "repelling_metric": "ARCHIVED", "matrix_adaptation": "CHOLESKY"}, {}),
        ("tpa_threshold_sequential", {"ssa": "TPA", "threshold_convergence": True, "sequential_selection": True}, {}),
        ("pairwise_sequential", {"mirrored": "PAIRWISE", "sequential_selection": True}, {}),
        ("orthogonal_mirrored_halton", {"orthogonal": True, "mirrored": "MIRRORED", "sampler": "HALTON"}, {}),
    ]
    for run_id, modules, settings in special:
        add(f"special/{run_id}", modules=modules, settings=settings, budget=None)

    add("constants/box_muller", constants={"use_box_muller": True})
    add("constants/cache_samples", constants={"cache_samples": True})
    add("constants/clip_sigma", constants={"clip_sigma": True})
    add("constants/box_muller_cotn", modules={"bound_correction": "COTN"}, constants={"use_box_muller": True})
    return runs


# ---------------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------------

CONSTANT_NAMES = ("cache_max_doubles", "cache_min_samples", "cache_samples", "clip_sigma", "use_box_muller")
DEFAULT_CONSTANTS = {name: getattr(c_maes.constants, name) for name in CONSTANT_NAMES}


def make_settings(run: Dict) -> "c_maes.Settings":
    modules = c_maes.Modules()
    for name, value in run["modules"].items():
        if not c_maes.set_module(modules, name, value):
            raise ValueError(f"unknown module {name}")
    kwargs = dict(run["settings"])
    for key in ("x0", "lb", "ub"):
        if key in kwargs:
            kwargs[key] = np.array(kwargs[key], dtype=float)
    return c_maes.Settings(run["dim"], modules, budget=run["budget"], verbose=False, **kwargs)


def execute(run: Dict) -> Dict:
    for name, value in DEFAULT_CONSTANTS.items():
        setattr(c_maes.constants, name, value)
    for name, value in run["constants"].items():
        setattr(c_maes.constants, name, value)

    c_maes.utils.set_seed(SEED)
    cma = c_maes.ModularCMAES(make_settings(run))
    objective = FUNCTIONS[run["function"]]

    checkpoints = []
    digest = hashlib.sha256()
    generation = 0
    try:
        while True:
            running = cma.step(objective)
            generation += 1
            p = cma.p
            digest.update(np.ascontiguousarray(p.pop.X).tobytes())
            digest.update(np.ascontiguousarray(p.pop.f).tobytes())
            digest.update(np.ascontiguousarray(p.adaptation.m).tobytes())
            digest.update(np.float64(p.mutation.sigma).tobytes())
            digest.update(np.int64(p.stats.evaluations).tobytes())
            if generation % CHECKPOINT_EVERY == 0 or not running:
                checkpoints.append(
                    [
                        generation,
                        int(p.stats.evaluations),
                        float(p.mutation.sigma),
                        float(p.pop.f[0]),
                        len(p.stats.centers),
                        digest.hexdigest()[:16],
                    ]
                )
                digest = hashlib.sha256()
            if not running:
                break
        result = {
            "best_y": float(cma.p.stats.global_best.y),
            "evaluations": int(cma.p.stats.evaluations),
            "generations": generation,
            "restarts": len(cma.p.stats.centers),
            "checkpoints": checkpoints,
        }
    except Exception as error:  # a crash is part of the behaviour we pin
        result = {"error": f"{type(error).__name__}: {error}", "checkpoints": checkpoints}
    return result


def platform_name() -> str:
    return f"{sys.platform}-{platform.machine().lower()}"


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURE_DIR, f"{name}.json")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=os.path.dirname(__file__),
        ).stdout.strip()
    except Exception:
        return "unknown"


def write(name: str, path: str = ""):
    runs = make_runs()
    for run in runs:
        run["result"] = execute(run)
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {
        "description": "Golden trajectories of the C++ ModularCMAES; see tests/golden.py",
        "platform": name,
        "generated_from_commit": git_commit(),
        "python": platform.python_version(),
        "seed": SEED,
        "checkpoint_every": CHECKPOINT_EVERY,
        "runs": runs,
    }
    path = path or fixture_path(name)
    header = {key: value for key, value in fixture.items() if key != "runs"}
    header["checkpoint_fields"] = list(CHECKPOINT_FIELDS)
    with open(path, "w") as f:
        # one run per line keeps the file small and its diffs readable
        f.write(json.dumps(header, indent=1)[:-2] + ',\n "runs": [\n')
        f.write(",\n".join(json.dumps(run, separators=(",", ":")) for run in runs))
        f.write("\n ]\n}\n")
    n_errors = sum("error" in run["result"] for run in runs)
    print(f"wrote {len(runs)} runs ({n_errors} raising) to {path}")


def compare(expected: Dict, actual: Dict) -> str:
    """Empty string when equal, else a description of the first difference."""
    if expected.get("error") != actual.get("error"):
        return f"error: expected {expected.get('error')!r}, got {actual.get('error')!r}"
    for e, a in zip(expected["checkpoints"], actual["checkpoints"]):
        if json.dumps(list(e)) != json.dumps(list(a)):  # json: NaN == NaN
            readable = lambda row: dict(zip(CHECKPOINT_FIELDS, row))  # noqa: E731
            return (
                f"first differs at generation <= {e[0]}: "
                f"expected {readable(e)}, got {readable(a)}"
            )
    if len(expected["checkpoints"]) != len(actual["checkpoints"]):
        return (
            f"number of checkpoints: expected {len(expected['checkpoints'])}, "
            f"got {len(actual['checkpoints'])}"
        )
    for key in ("best_y", "evaluations", "generations", "restarts"):
        if json.dumps(expected.get(key)) != json.dumps(actual.get(key)):
            return f"{key}: expected {expected.get(key)}, got {actual.get(key)}"
    return ""


def check(name: str, only: str = "", path: str = "") -> int:
    path = path or fixture_path(name)
    if not os.path.exists(path):
        print(f"SKIP: no golden fixture for platform {name} ({path})")
        return 0
    with open(path) as f:
        fixture = json.load(f)
    failures = []
    runs = [run for run in fixture["runs"] if only in run["id"]]
    for run in runs:
        message = compare(run["result"], execute(run))
        if message:
            failures.append((run["id"], message))
    for run_id, message in failures:
        print(f"MISMATCH {run_id}: {message}")
    print(f"{len(runs) - len(failures)}/{len(runs)} runs match the {name} fixture "
          f"(generated from {fixture['generated_from_commit']})")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="Golden trajectories of the C++ ModularCMAES")
    parser.add_argument("command", choices=("write", "check", "list"))
    parser.add_argument("--platform", default=platform_name())
    parser.add_argument("--only", default="", help="check only runs whose id contains this text")
    parser.add_argument("--fixture", default="", help="fixture file instead of tests/data/golden/<platform>.json")
    args = parser.parse_args()
    if args.command == "write":
        write(args.platform, args.fixture)
    elif args.command == "check":
        sys.exit(check(args.platform, args.only, args.fixture))
    else:
        for run in make_runs():
            print(run["id"])


if __name__ == "__main__":
    main()
