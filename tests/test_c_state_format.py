"""The saved state of format 1 (#91, #192): states written by modcma 2.0.0 still load.

tests/data/states/format1.xz holds 218 states that `write()` saved with the C++ serialization of
master at commit 52079ed, before the serialization was rewritten (#192): the optimizer between
ask() and tell(), in the configurations of the golden fixture's pairwise covering set that cover
every option of every module, its special cases and the process-wide constants; also between the
hill-valley points of a restart, and at the end of a run; and the objects of those runs on their own
(Parameters, Settings, the modules, the criteria, ...), and standalone samplers. Each state must
load, and the loaded object must save to the same bytes: the reader reads format 1 as it was
written, and the writer still writes it. This holds on every platform (the state holds the bits of
every number; nothing in it depends on the compiler).

To keep the file small, `write()` leaves out the states larger than 12 kB, and prints them: 11 of
them, the large populations of special/large_lambda (ask3, end) and the states of
special/orthogonal_mirrored_halton (ask3, end), the end of 5 runs with restarts and of
constants/cache_samples, and that run's CachedSampler on its own. Every option of every module is
still in a kept state, and so are the orthogonal and mirrored samplers, Halton and CachedSampler
(standalone samplers).

The file is the record of format 1: it is not regenerated. A new format version keeps reading it,
and this test then checks that a format-1 state loads (and no longer that it saves to the same
bytes). `python -m tests.test_c_state_format write` wrote it; `... write <path>` writes the
states of the current code elsewhere, to compare with `read(<path>)`.
"""

import json
import lzma
import os
import sys
import unittest

import numpy as np

from modcma import c_maes
from modcma.c_maes import cmaescpp

# An xz file: a line of JSON (format_version, generated_from_commit, and the name, class and length
# of every state), then the states, one after the other
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "states", "format1.xz")
PREFIX = "modcma.c_maes.cmaescpp."


def class_name(obj) -> str:
    """The name of obj's class below modcma.c_maes.cmaescpp, e.g. "mutation.CSA" """
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"[len(PREFIX):]


def find_class(name: str):
    cls = cmaescpp
    for part in name.split("."):
        cls = getattr(cls, part)
    return cls


def load(name: str, state: bytes):
    cls = find_class(name)
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj


def write(path: str = FIXTURE):
    """Write the fixture (once, with the serialization of format 1; see the module docstring).
    Prints the states it leaves out for their size."""
    from tests import golden

    # The pairwise configurations that cover every option of every module (a greedy cover), the
    # special cases and the process-wide constants
    pairs = [run for run in golden.make_runs() if run["id"].startswith("pair/")]
    uncovered = {(name, value) for run in pairs for name, value in run["modules"].items()}
    cover = set()
    while uncovered:
        best = max(pairs, key=lambda run: len(uncovered & set(run["modules"].items())))
        cover.add(best["id"])
        uncovered -= set(best["modules"].items())
    runs = []
    for run in golden.make_runs():
        group = run["id"].split("/")[0]
        if group == "pair" or (group == "constants" and run["function"] == "ellipsoid"):
            runs.append(dict(run, dim=3, budget=1200))
        elif group == "special":
            # The runs with restarts long enough for two, so that the repelling archive is tested
            if "repelling" in run["id"] or "restart" in run["id"]:
                if run["function"] == "rastrigin":
                    runs.append(dict(run, dim=3, budget=3000))
            elif run["function"] == "ellipsoid":
                runs.append(dict(run, budget=1500))

    records, objects = [], {}
    max_size = 12000

    def keep(name, obj, max_size=max_size):
        state = obj.__getstate__()
        if len(state) <= max_size:
            records.append((name, class_name(obj), state))
        else:
            print(f"left out (larger than {max_size} bytes): {name}, {len(state)} bytes")

    def keep_object(obj):
        states = objects.setdefault(class_name(obj), [])
        state = obj.__getstate__()
        if len(states) < 2 and state not in states:
            if len(state) <= max_size:
                states.append(state)
            else:
                print(f"left out (larger than {max_size} bytes): object/{class_name(obj)}, "
                      f"{len(state)} bytes")

    try:
        for run in runs:
            for name, value in golden.DEFAULT_CONSTANTS.items():
                setattr(c_maes.constants, name, value)
            for name, value in run["constants"].items():
                setattr(c_maes.constants, name, value)
            c_maes.set_seed(golden.SEED)
            es = c_maes.ModularCMAES(golden.make_settings(run))
            objective = golden.FUNCTIONS[run["function"]]
            # The id of the golden run, with the dimension of this run
            run_id = f"{run['id'].rsplit('/', 1)[0]}/d{run['dim']}"
            asks, previous_single, hill_valley = 0, False, False
            while not es.break_conditions():
                X = es.ask()
                asks += 1
                if asks == 3 and (run["id"] in cover or not run["id"].startswith("pair/")):
                    keep(f"{run_id}/ask3", es)
                # Two single points in a row after a run: the center of mass, then a hill-valley
                # point of the repelling archive
                single = len(X) == 1
                if single and previous_single and not hill_valley and len(es.p.repelling.archive):
                    keep(f"{run_id}/hill-valley", es, max_size=16000)
                    hill_valley = True
                previous_single = single
                es.tell([objective(x) for x in X])
            if not run["id"].startswith("pair/"):
                keep(f"{run_id}/end", es)

            p = es.p
            parts = [p, p.settings, p.settings.modules, p.stats, p.weights, p.pop, p.old_pop,
                     p.sampler, p.adaptation, p.mutation, p.mutation.threshold_convergence,
                     p.mutation.sequential_selection, p.mutation.sigma_sampler, p.selection,
                     p.selection.pairwise, p.selection.elitism, p.restart_strategy, p.bounds,
                     p.repelling, p.center_placement, p.criteria, p.coordinate_mapping,
                     p.stats.global_best, *p.criteria.items]
            if len(p.stats.centers):
                parts += [p.stats.centers[-1], p.stats.centers[-1].metric]
            if len(p.repelling.archive):
                parts.append(p.repelling.archive[-1])
            for obj in parts:
                keep_object(obj)
    finally:
        for name, value in golden.DEFAULT_CONSTANTS.items():
            setattr(c_maes.constants, name, value)

    # Samplers on their own, after a few samples
    c_maes.set_seed(golden.SEED)
    s = c_maes.sampling
    for sampler in (s.Gaussian(3), s.Uniform(3), s.Sobol(3), s.Halton(3), s.FastGaussian(3),
                    s.Mirrored(s.GaussianTransformer(s.Sobol(3))),
                    s.Orthogonal(s.GaussianTransformer(s.Halton(3)), 5),
                    s.CachedSampler(s.LaplaceTransformer(s.Uniform(3))),
                    s.CachedSampler([np.array([1.0, 2.0]), np.array([3.0, 4.0])])):
        for _ in range(7):
            sampler()
        keep_object(sampler)

    for name, states in sorted(objects.items()):
        for state in states:
            records.append((f"object/{name}", name, state))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = {
        "format_version": 1,
        "generated_from_commit": golden.git_commit(),
        "states": [[name, cls, len(state)] for name, cls, state in records],
    }
    with lzma.open(path, "wb", preset=9 | lzma.PRESET_EXTREME) as f:
        f.write(json.dumps(header, separators=(",", ":")).encode() + b"\n")
        for _, _, state in records:
            f.write(state)
    print(f"{len(records)} states, {os.path.getsize(path)} bytes: {path}")


def read(path: str = FIXTURE):
    """The fixture: (format_version, [(name, class, state), ...])"""
    with lzma.open(path, "rb") as f:
        header = json.loads(f.readline())
        states = [(name, cls, f.read(size)) for name, cls, size in header["states"]]
        assert f.read() == b"", "data after the last state"
    return header["format_version"], states


class TestFormat1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.version, cls.states = read()

    def test_the_states_cover_every_class(self):
        names = {name for _, name, _ in self.states}
        for name in ("ModularCMAES", "Parameters", "parameters.Settings", "sampling.Sobol",
                     "sampling.Halton", "sampling.Orthogonal", "sampling.CachedSampler",
                     "sampling.Mirrored", "sampling.FastGaussian", "repelling.TabooPoint",
                     "repelling.AdaptiveRepelling", "restart.Stagnation", "restart.BIPOP",
                     "mutation.TPAInPlace", "bounds.COTN",
                     "matrix_adaptation.NaturalGradientAdaptation"):
            self.assertIn(name, names)
        self.assertTrue(any(n.endswith("/hill-valley") for n, _, _ in self.states))

    def test_a_loaded_state_saves_to_the_same_bytes(self):
        self.assertEqual(self.version, 1)
        for name, cls, state in self.states:
            with self.subTest(state=name):
                self.assertEqual(load(cls, state).__getstate__(), state)

    def test_a_loaded_optimizer_runs(self):
        for name, cls, state in self.states:
            if cls == "ModularCMAES" and name.endswith("/ask3"):
                with self.subTest(state=name):
                    es = load(cls, state)
                    for _ in range(3):
                        X = es.ask()
                        es.tell([float(np.sum(x * x)) for x in X])


if __name__ == "__main__":
    if sys.argv[1:2] == ["write"]:
        write(*sys.argv[2:3])
    else:
        unittest.main()
