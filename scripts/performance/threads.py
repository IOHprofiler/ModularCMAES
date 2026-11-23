"""Throughput of independent optimizers in parallel threads, on free-threaded CPython (3.14t).

Every optimizer has its own random generator and the module does not re-enable the GIL, so N
threads can each run an optimizer at the same time. Each thread runs `runs` optimizers (d = 10,
ellipsoid, vectorized objective, a fixed number of generations); the throughput is the number of
generations per second of all threads together.

    python3.14t scripts/performance/threads.py --output scripts/performance/results/threads.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from modcma import c_maes


def work(d: int, runs: int, generations: int, seed: int) -> None:
    w = 10.0 ** (6.0 * np.arange(d) / (d - 1))
    for r in range(runs):
        settings = c_maes.Settings(
            d, x0=np.zeros(d), sigma0=2.0, max_generations=generations, budget=10**15,
            seed=seed * 1000 + r,
        )
        c_maes.ModularCMAES(settings).run(lambda X: (X * X) @ w, vectorized=True)


def throughput(n_threads: int, d: int, runs: int, generations: int) -> float:
    threads = [threading.Thread(target=work, args=(d, runs, generations, k))
               for k in range(n_threads)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return n_threads * runs * generations / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threads", default="1,2,4,6,8,10")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--output", default="scripts/performance/results/threads.csv")
    args = parser.parse_args()

    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, GIL enabled: {gil}")
    rows = []
    for n in (int(x) for x in args.threads.split(",")):
        rate = max(throughput(n, args.dim, args.runs, args.generations) for _ in range(3))
        rows.append((n, rate))
        print(f"{n:3d} threads: {rate:10.0f} generations/s", flush=True)
    import csv

    with open(args.output, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["threads", "generations_per_second", "speedup", "gil_enabled"])
        for n, rate in rows:
            out.writerow([n, round(rate), round(rate / rows[0][1], 2), gil])


if __name__ == "__main__":
    main()
