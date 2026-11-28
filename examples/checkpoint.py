"""Checkpointing a long run: the optimizer is saved with pickle every EVERY generations, and a run
that is interrupted continues from its last checkpoint.

The objective is not part of the saved state; it is given again. The resumed run is the run that
was not interrupted, bit for bit: it ends with the same best point, after the same number of
evaluations.
"""

import os
import pickle
import tempfile

import numpy as np

import modcma

QUICK = os.environ.get("MODCMA_EXAMPLE_QUICK") == "1"
EVERY = 100  # generations between two checkpoints
BUDGET = 20_000 if QUICK else 100_000


def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def new_optimizer():
    modules = modcma.Modules()
    modules.restart_strategy = modcma.options.RestartStrategy.IPOP
    return modcma.ModularCMAES(modcma.Settings(10, modules, budget=BUDGET, seed=42))


def save(es, path):
    """Write the checkpoint to a new file, then replace the old one: an interruption while the
    file is written leaves the previous checkpoint intact"""
    with open(path + ".new", "wb") as f:
        pickle.dump(es, f)
    os.replace(path + ".new", path)


def run(path, stop_after=None):
    """Run until a break condition holds, from the checkpoint at path when there is one.
    stop_after simulates an interruption (a crash, the end of a cluster job) after that many
    generations of this call."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            es = pickle.load(f)
        print(f"resumed from the checkpoint of generation {es.p.stats.t}")
    else:
        es = new_optimizer()
    generations = 0
    while es.step(rastrigin):
        generations += 1
        if es.p.stats.t % EVERY == 0:
            save(es, path)
        if generations == stop_after:
            print(f"interrupted after generation {es.p.stats.t}")
            return None
    return es


with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "run.pkl")
    run(path, stop_after=250)
    resumed = run(path)
    print(f"checkpoint: {os.path.getsize(path)} bytes")

uninterrupted = new_optimizer()
uninterrupted.run(rastrigin)

for name, es in (("resumed", resumed), ("uninterrupted", uninterrupted)):
    stats = es.p.stats
    print(
        f"{name:13s} best f = {stats.global_best.y:.6g} after {stats.evaluations} evaluations, "
        f"{stats.t} generations, {len(stats.centers)} runs"
    )
assert resumed.p.stats.evaluations == uninterrupted.p.stats.evaluations
assert np.array_equal(resumed.p.stats.global_best.x, uninterrupted.p.stats.global_best.x)
