"""Statistical comparison of two builds of the C++ ModularCMAES (dev/overhaul-2.0.md, T0.1 tier 2; D2).

Changes that alter the floating-point results (a different summation order, a different random
stream) make the golden trajectories diverge, so they are validated statistically instead: both
builds run the same grid (BBOB functions from ioh x module configurations x dimensions x seeds) and
for every cell the distributions are compared with a two-sided Mann-Whitney U test, on the number of
evaluations used (to the target 1e-8, or the budget) and on the final precision.

Pre-registered criterion: no cell differs after a Bonferroni correction over all tests
(p < 0.05 / number of tests), and at most 10 % of the tests have a raw p < 0.05.

    python -m tests.statistical run --output new.json        # in the environment of each build
    python -m tests.statistical compare old.json new.json

Needs ioh and scipy (not dependencies of modcma).
"""

import argparse
import json
import math
import multiprocessing
import sys

import numpy as np

FUNCTIONS = (1, 2, 8, 10, 12, 15, 21)  # sphere, ellipsoid, Rosenbrock, rotated ellipsoid,
# bent cigar, rotated Rastrigin, Gallagher 101 peaks
DIMS = (5, 20)
SEEDS = range(25)
BUDGET_PER_DIM = 3000
TARGET = 1e-8

CONFIGS = {
    "default": {},
    "active": {"active": True},
    "elitist": {"elitist": True},
    "matrix": {"matrix_adaptation": "MATRIX"},
    "cholesky": {"matrix_adaptation": "CHOLESKY"},
    "separable": {"matrix_adaptation": "SEPARABLE"},
    "cmsa": {"matrix_adaptation": "CMSA"},
    "natural_gradient": {"matrix_adaptation": "NATURAL_GRADIENT"},
    "no_eigv_active": {"matrix_adaptation": "COVARIANCE_NO_EIGV", "active": True},
    "matrix_elitist_active": {"matrix_adaptation": "MATRIX", "elitist": True, "active": True},
    "ipop_uniform_center": {"restart_strategy": "IPOP", "center_placement": "UNIFORM"},
    "bipop": {"restart_strategy": "BIPOP"},
    "tpa_mirrored_orthogonal": {"ssa": "TPA", "mirrored": "MIRRORED", "orthogonal": True},
    "sequential_resample": {
        "sequential_selection": True,
        "bound_correction": "RESAMPLE",
    },
    # WP8: the fixed modules on their own
    "exponential_weights_active": {"weights": "EXPONENTIAL", "active": True},
    "mirrored_pairwise": {"mirrored": "PAIRWISE"},
    "orthogonal": {"orthogonal": True},
    "ipop": {"restart_strategy": "IPOP"},
    "sequential_active": {"sequential_selection": True, "active": True},
    # The design decisions of #83
    "msr": {"ssa": "MSR"},
    "psr": {"ssa": "PSR"},
    "sr": {"ssa": "SR"},
    "one_plus_one": {"lambda0": 1},
    "pairwise_active": {"mirrored": "PAIRWISE", "active": True},
    "elitist_sa_cmsa": {"elitist": True, "ssa": "SA", "matrix_adaptation": "CMSA"},
    # #77: sequential selection's reference, h_sigma per run
    "sequential": {"sequential_selection": True},
    "sequential_mirrored": {"sequential_selection": True, "mirrored": "MIRRORED"},
    "sequential_pairwise": {"sequential_selection": True, "mirrored": "PAIRWISE"},
    "sequential_mirrored_ipop": {
        "sequential_selection": True,
        "mirrored": "MIRRORED",
        "restart_strategy": "IPOP",
    },
    "tpa": {"ssa": "TPA"},
    # #78: TPA's ties, threshold convergence per generation
    "tpa_elitist": {"ssa": "TPA", "elitist": True},
    "tpa_ipop": {"ssa": "TPA", "restart_strategy": "IPOP"},
    "threshold": {"threshold_convergence": True},
    "threshold_bipop": {"threshold_convergence": True, "restart_strategy": "BIPOP"},
    # #80: the restart criteria for every matrix adaptation, BIPOP's regimes
    "restart": {"restart_strategy": "RESTART"},
    "ipop_msr": {"restart_strategy": "IPOP", "ssa": "MSR"},  # COVARIANCE_NO_EIGV
    "ipop_matrix": {"restart_strategy": "IPOP", "matrix_adaptation": "MATRIX"},
    "ipop_separable": {"restart_strategy": "IPOP", "matrix_adaptation": "SEPARABLE"},
    "ipop_cholesky": {"restart_strategy": "IPOP", "matrix_adaptation": "CHOLESKY"},
    "ipop_cmsa": {"restart_strategy": "IPOP", "matrix_adaptation": "CMSA"},
    "ipop_natural_gradient": {"restart_strategy": "IPOP", "matrix_adaptation": "NATURAL_GRADIENT"},
    "ipop_none": {"restart_strategy": "IPOP", "matrix_adaptation": "NONE"},
    # #72: the learning rates of NATURAL_GRADIENT and of the xNES rules, the rate of ps
    "natural_gradient_xnes": {"matrix_adaptation": "NATURAL_GRADIENT", "ssa": "XNES"},
    "natural_gradient_lpxnes": {"matrix_adaptation": "NATURAL_GRADIENT", "ssa": "LPXNES"},
    "natural_gradient_mxnes": {"matrix_adaptation": "NATURAL_GRADIENT", "ssa": "MXNES"},
    "xnes": {"ssa": "XNES"},
    "xnes_utility": {"ssa": "XNES", "weights": "UTILITY"},
    "lpxnes": {"ssa": "LPXNES"},
    "lpxnes_ipop": {"ssa": "LPXNES", "restart_strategy": "IPOP"},
    "mxnes_matrix": {"ssa": "MXNES", "matrix_adaptation": "MATRIX"},
    # #73: the active update outside COVARIANCE, the weights for mu != lambda / 2, SEPARABLE's cap
    "separable_active": {"matrix_adaptation": "SEPARABLE", "active": True},
    "ipop_separable_active": {
        "restart_strategy": "IPOP",
        "matrix_adaptation": "SEPARABLE",
        "active": True,
    },
    "matrix_active": {"matrix_adaptation": "MATRIX", "active": True},
    "cholesky_active": {"matrix_adaptation": "CHOLESKY", "active": True},
    "cholesky_utility": {"matrix_adaptation": "CHOLESKY", "weights": "UTILITY"},
    "cmsa_active": {"matrix_adaptation": "CMSA", "active": True},
    "mu_low_active": {"lambda0": 10, "mu0": 3, "active": True},
    "mu_high_active": {"lambda0": 10, "mu0": 7, "active": True},
    # #74: the inverse of M with elitism
    "matrix_elitist": {"matrix_adaptation": "MATRIX", "elitist": True},
    "bipop_pairwise": {"restart_strategy": "BIPOP", "mirrored": "PAIRWISE"},
    # #75, #76, #116: mirrored pairs per generation, pairwise values, orthogonal blocks per
    # generation and without the sign bias, the expected lengths, portable HALTON and SOBOL (run
    # the pairwise and mirrored ones at d = 3 too: an odd default lambda)
    "mirrored": {"mirrored": "MIRRORED"},
    "mirrored_resample": {"mirrored": "MIRRORED", "bound_correction": "RESAMPLE"},
    "pairwise_msr": {"mirrored": "PAIRWISE", "ssa": "MSR"},
    "pairwise_psr": {"mirrored": "PAIRWISE", "ssa": "PSR"},
    "orthogonal_mirrored": {"orthogonal": True, "mirrored": "MIRRORED"},
    "orthogonal_pairwise": {"orthogonal": True, "mirrored": "PAIRWISE"},
    "halton": {"sampler": "HALTON"},
    "sobol": {"sampler": "SOBOL"},
    "scaled_uniform": {"sample_transformation": "SCALED_UNIFORM"},
    "laplace": {"sample_transformation": "LAPLACE"},
    "logistic": {"sample_transformation": "LOGISTIC"},
    "double_weibull": {"sample_transformation": "DOUBLE_WEIBULL"},
    # #121, #139, #150: adaptive repelling's run lengths, the criteria per run, mu after a restart
    # (an odd lambda0 at d = 3, 6, 7; a given mu0; a quarter with SA)
    "restart_adaptive": {"restart_strategy": "RESTART", "repelling_type": "ADAPTIVE"},
    "ipop_adaptive": {"restart_strategy": "IPOP", "repelling_type": "ADAPTIVE"},
    "bipop_adaptive": {"restart_strategy": "BIPOP", "repelling_type": "ADAPTIVE"},
    "ipop_mu0": {"restart_strategy": "IPOP", "mu0": 3},
    "bipop_mu0": {"restart_strategy": "BIPOP", "mu0": 3},
    "ipop_sa": {"restart_strategy": "IPOP", "ssa": "SA"},
    "restart_max_iter": {"restart_strategy": "RESTART", "max_iter": 30},
    "ipop_max_iter": {"restart_strategy": "IPOP", "max_iter": 30},
    # #71: one sampled step size per individual (SA, LPXNES, sample_sigma); dm divided by the
    # recombined step size
    "sa": {"ssa": "SA"},
    "sa_cmsa_equal": {"ssa": "SA", "matrix_adaptation": "CMSA", "weights": "EQUAL"},
    "sa_elitist": {"ssa": "SA", "elitist": True},
    "lpxnes": {"ssa": "LPXNES"},
    "sample_sigma": {"sample_sigma": True},
    "sample_sigma_tpa": {"sample_sigma": True, "ssa": "TPA"},
    # #98: TPA's points corrected into the box (a box of +-4: BBOB's optima lie in [-4, 4]^d)
    "tpa_saturate": {"ssa": "TPA", "bound_correction": "SATURATE", "box": 4.0},
    "tpa_mirror": {"ssa": "TPA", "bound_correction": "MIRROR", "box": 4.0},
    "tpa_cotn": {"ssa": "TPA", "bound_correction": "COTN", "box": 4.0},
    "tpa_toroidal": {"ssa": "TPA", "bound_correction": "TOROIDAL", "box": 4.0},
    "tpa_uniform_resample": {"ssa": "TPA", "bound_correction": "UNIFORM_RESAMPLE", "box": 4.0},
    "tpa_resample": {"ssa": "TPA", "bound_correction": "RESAMPLE", "box": 4.0},
    # #120: PSR's ties (certain with elitism)
    "psr_elitist": {"ssa": "PSR", "elitist": True},
    # #131: lambda0 = 1 runs SR whatever ssa is (one_plus_one above is the default, CSA)
    "one_plus_one_tpa": {"lambda0": 1, "ssa": "TPA"},
    "one_plus_one_sa": {"lambda0": 1, "ssa": "SA"},
}

#: Settings arguments next to the module options
SETTINGS = ("lambda0", "mu0")


def cells(dims=DIMS):
    for config in CONFIGS:
        for fid in FUNCTIONS:
            for dim in dims:
                yield config, fid, dim


def run_one(task):
    config, fid, dim, seed = task
    import ioh

    from modcma import c_maes

    problem = ioh.get_problem(fid, 1, dim, ioh.ProblemClass.BBOB)
    modules = c_maes.Modules()
    extra = {}  # Settings arguments (lambda0, mu0) next to the module options
    max_iter = None  # the most generations of a run (restart.Strategy.max_iter)
    box = 5.0  # the box [-box, box]^d
    for name, value in CONFIGS[config].items():
        if name == "box":
            box = value
        elif name in SETTINGS:
            extra[name] = value
        elif name == "max_iter":
            max_iter = value
        elif not c_maes.set_module(modules, name, value):
            raise ValueError(f"unknown module option {name}={value}")
    c_maes.set_seed(seed)
    settings = c_maes.Settings(
        dim,
        modules,
        budget=BUDGET_PER_DIM * dim,
        lb=np.full(dim, -box),
        ub=np.full(dim, box),
        verbose=False,
        **extra,
    )
    settings.target = problem.optimum.y + TARGET
    cma = c_maes.ModularCMAES(settings)
    if max_iter is not None:
        # ExceededMaxIter takes the smaller of its own limit and the strategy's at every reset;
        # the first run was reset when the optimizer was created
        cma.p.restart_strategy.max_iter = max_iter
        for criterion in cma.p.criteria.items:
            if criterion.name == "ExceededMaxIter":
                criterion.max_iter = min(criterion.max_iter, max_iter)
    cma.run(lambda X: problem(X), vectorized=True)
    precision = max(float(cma.p.stats.global_best.y - problem.optimum.y), TARGET)
    return config, fid, dim, seed, int(cma.p.stats.evaluations), precision


def run(output: str, processes: int, configs=None, dims=DIMS):
    from modcma import c_maes

    tasks = [(c, f, d, s) for c, f, d in cells(dims) for s in SEEDS if not configs or c in configs]
    with multiprocessing.Pool(processes) as pool:
        results = pool.map(run_one, tasks, chunksize=4)
    data = {"build": c_maes.build_info(), "runs": {}}
    for config, fid, dim, seed, evaluations, precision in results:
        data["runs"].setdefault(f"{config}/f{fid}/d{dim}", []).append([seed, evaluations, precision])
    with open(output, "w") as f:
        json.dump(data, f)
    print(f"wrote {len(results)} runs to {output}")


def compare(old_path: str, new_path: str) -> int:
    from scipy.stats import mannwhitneyu

    old = json.load(open(old_path))["runs"]
    new = json.load(open(new_path))["runs"]
    tests = []
    old = {cell: runs for cell, runs in old.items() if cell in new}
    for cell in old:
        a, b = np.array(old[cell]), np.array(new[cell])
        for column, name in ((1, "evaluations"), (2, "precision")):
            x, y = a[:, column], b[:, column]
            p = 1.0 if np.all(x == x[0]) and np.all(y == x[0]) else mannwhitneyu(x, y).pvalue
            tests.append((p, cell, name, np.median(x), np.median(y)))

    n = len(tests)
    bonferroni = 0.05 / n
    raw = sum(p < 0.05 for p, *_ in tests)
    print(f"{len(old)} cells, {n} tests; Bonferroni threshold {bonferroni:.2e}")
    print(f"raw p < 0.05: {raw} of {n} ({100 * raw / n:.1f} %, criterion <= 10 %)")
    print("smallest p-values (median old -> new):")
    for p, cell, name, mo, mn in sorted(tests)[:12]:
        flag = "  DIFFERS" if p < bonferroni else ""
        print(f"  {p:9.2e}  {cell:42s} {name:11s} {mo:12.4g} -> {mn:12.4g}{flag}")

    # Overall: the ratio of median evaluations of the cells where most runs reach the target
    ratios = [
        np.median(np.array(new[c])[:, 1]) / np.median(np.array(old[c])[:, 1])
        for c in old
        if np.median(np.array(old[c])[:, 2]) <= TARGET
    ]
    if ratios:
        print(f"median evaluations new/old over {len(ratios)} solved cells: "
              f"geometric mean {math.exp(np.mean(np.log(ratios))):.4f}, "
              f"range {min(ratios):.3f}..{max(ratios):.3f}")
    failed = any(p < bonferroni for p, *_ in tests) or raw > 0.10 * n
    print("criterion:", "FAILED" if failed else "met")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    r.add_argument("--processes", type=int, default=multiprocessing.cpu_count())
    r.add_argument("--configs", default="", help="comma-separated subset of the configurations")
    r.add_argument("--dims", default=",".join(map(str, DIMS)), help="comma-separated dimensions")
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.processes, [c for c in args.configs.split(",") if c],
            tuple(int(d) for d in args.dims.split(",")))
    else:
        sys.exit(compare(args.old, args.new))


if __name__ == "__main__":
    main()
