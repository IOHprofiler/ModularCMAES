"""Benchmark two configurations on the BBOB functions with IOHexperimenter.

Requires `pip install ioh`. The runs are logged in the IOHanalyzer format (folder `bbob-data`), so
they can be compared at https://iohanalyzer.liacs.nl, and summarized here: the fraction of
(function, instance) pairs that reached the target and the median evaluations they needed.
"""

import os
import tempfile

import ioh
import numpy as np

import modcma

QUICK = os.environ.get("MODCMA_EXAMPLE_QUICK") == "1"
DIM = 5
FUNCTIONS = [1, 8, 15] if QUICK else range(1, 25)
INSTANCES = [1] if QUICK else [1, 2, 3, 4, 5]
BUDGET = 1_000 * DIM if QUICK else 10_000 * DIM
PRECISION = 1e-8  # target: f - f_opt <= 1e-8

CONFIGURATIONS = {
    "CMA-ES": {},
    "BIPOP active CMA-ES": {"restart_strategy": "BIPOP", "active": True},
}


def run(problem, seed, **modules):
    settings = modcma.settings_from_dict(
        DIM,
        lb=problem.bounds.lb,
        ub=problem.bounds.ub,
        budget=BUDGET,
        target=problem.optimum.y + PRECISION,
        seed=seed,
        **modules,
    )
    es = modcma.ModularCMAES(settings)
    es.run(problem)  # an ioh problem is a callable: x -> f(x)
    return problem.state.current_best.y - problem.optimum.y, problem.state.evaluations


def main(folder):
    for name, modules in CONFIGURATIONS.items():
        logger = ioh.logger.Analyzer(root=folder, folder_name=name, algorithm_name=name)
        hits = []
        for fid in FUNCTIONS:
            for instance in INSTANCES:
                problem = ioh.get_problem(fid, instance, DIM, ioh.ProblemClass.BBOB)
                problem.attach_logger(logger)
                precision, evaluations = run(problem, seed=instance, **modules)
                hits.append(evaluations if precision <= PRECISION else None)
                problem.reset()
        logger.close()
        solved = [e for e in hits if e is not None]
        print(
            f"{name:22s} solved {len(solved)}/{len(hits)} (f - f_opt <= {PRECISION:g}), "
            f"median {np.median(solved):.0f} evaluations when solved"
        )


if __name__ == "__main__":
    if QUICK:
        with tempfile.TemporaryDirectory() as folder:
            main(folder)
    else:
        main("bbob-data")
