import argparse
import time
import os
import sys

import ioh
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import scipy as sp
import math
import pandas as pd
import modcma.c_maes as c_cmaes
from modcma.c_maes.cmaescpp.parameters import Solution


base_dir = os.path.realpath(os.path.dirname(__file__))

def belongs_to_same_basin(x, y, problem, samples=10):
    return c_cmaes.repelling.hill_valley_test(x, y, problem, samples)

def make_solution(x, y, t, e):
    sol = Solution()
    sol.x = x
    sol.y = y
    sol.t = t
    sol.e = e
    return sol

def calculate_potential(centers, problem):
    opt = Solution()
    opt.x = problem.optimum.x
    opt.y = problem.optimum.y

    potential = 0
    n_duplicate_runs = 0
    last_restart = centers[0].e
    basins = [centers[0]]

    for center in centers[1:]:
        if not belongs_to_same_basin(center, opt, problem):
            for basin in basins:
                if belongs_to_same_basin(center, basin, problem):
                    potential += center.e - last_restart
                    n_duplicate_runs += 1
                    break
            else:
                basins.append(center)

        last_restart = center.e

    return potential, n_duplicate_runs


def plot_contour(X, Y, Z, colorbar=True):
    plt.contourf(
        X, Y, np.log10(Z), levels=200, cmap="Spectral", zorder=-1, vmin=-1, vmax=2.5
    )
    plt.xlabel(R"$x_1$")
    plt.ylabel(R"$x_2$")
    if colorbar:
        plt.colorbar()
    plt.tight_layout()


def get_meshgrid(objective_function, lb, ub, delta: float = 0.025):
    x = np.arange(lb, ub + delta, delta)
    y = np.arange(lb, ub + delta, delta)

    if hasattr(objective_function, "optimum"):
        xo, yo = objective_function.optimum.x
        x = np.sort(np.r_[x, xo])
        y = np.sort(np.r_[y, yo])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(X.shape)
    for idx1 in range(X.shape[0]):
        for idx2 in range(X.shape[1]):
            Z[idx1, idx2] = objective_function([X[idx1, idx2], Y[idx1, idx2]])
    return X, Y, Z


def rastrigin_function(x: np.ndarray) -> float:
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x * x - (10 * np.cos(2 * np.pi * x)))


def plot(
    cma: c_cmaes.ModularCMAES,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    lb: float,
    ub: float,
    problem
):
    plt.clf()
    plt.title(
        f"Generation: {cma.p.stats.t} evals: {cma.p.stats.evaluations} Lambda: {cma.p.lamb} n_resamples: {cma.p.repelling.attempts}"
    )
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plot_contour(X, Y, Z)

    m = cma.p.adaptation.m.copy()
    C = cma.p.adaptation.C.copy()
    X = cma.p.pop.X.copy()
    current_best = cma.p.stats.current_best.x.copy()
    global_best = cma.p.stats.global_best.x.copy()
    current_best_y = cma.p.stats.current_best.y - problem.optimum.y
    global_best_y = cma.p.stats.global_best.y - problem.optimum.y

    sigma = cma.p.mutation.sigma
    theta = np.degrees(np.arctan2(C[1, 0], C[0, 0]))

    main_color = "m"
    for scale in (0.5, 1.0, 2.0, 3.0, 1000):
        current = Ellipse(
            m,
            *(scale * sigma * np.diag(C)),
            angle=theta,
            facecolor="none",
            edgecolor=main_color,
            linewidth=2,
            linestyle="dashed",
            zorder=0,
        )
        ax.add_patch(current)

    ax.scatter(*m, color=main_color, label="current dist")
    if len(current_best):
        ax.scatter(
            *current_best,
            color=main_color,
            label=f"current best {current_best_y: .2f}",
            marker="x",
        )
    if len(global_best):
        ax.scatter(
            *global_best,
            color='blue',
            label=f"global best {global_best_y: .2f}",
            marker="*",
        )

    ax.scatter(X[0, :], X[1, :], color=main_color, alpha=0.5)

    for t, tabu_point in enumerate(cma.p.repelling.archive, 1):
        if c_cmaes.constants.repelling_current_cov:
            Ct = C
        else:
            Ct = tabu_point.C

        theta_t = np.degrees(np.arctan2(Ct[1, 0], Ct[0, 0]))

        # print(theta_t, np.degrees(np.arctan2(tabu_point.C[1, 0],  tabu_point.C[0, 0])))
        current = Ellipse(
            tabu_point.solution.x,
            *(2 * sigma * tabu_point.radius * np.diag(Ct)),
            angle=theta_t,
            facecolor="grey",
            alpha=.5,
            edgecolor="black",
            label=f"taboo point {t} radius: {tabu_point.radius: .2f} n_rep: {tabu_point.n_rep} y: {tabu_point.solution.y: .2f}",
            linewidth=2,
            linestyle="dashed",
            zorder=0,
        )
        c = pow(tabu_point.shrinkage, cma.p.repelling.attempts)
        effective_c = Ellipse(
            tabu_point.solution.x,
            *(c * 2 * sigma * tabu_point.radius * np.diag(Ct)),
            angle=theta_t,
            facecolor="none",
            edgecolor="gray",
            linewidth=1.5,
            linestyle="dashed",
            zorder=0,
        )
        
        if t != len(cma.p.repelling.archive):
            ax.scatter(
            *tabu_point.solution.x, 
                marker='p',
                color='black',            
            )


        if np.isnan(tabu_point.radius):
            m = len(cma.p.repelling.archive)
            i = len(cma.p.stats.solutions)
            the = 0.5 * (i / m)
            tau = 1 / np.sqrt(2)
            breakpoint()
        ax.add_patch(current)
        ax.add_patch(effective_c)

    # print(cma.p.stats)
    plt.grid()
    # plt.legend(loc="upper left")
    plt.xlim(lb, ub)
    plt.ylim(lb, ub)
    plt.draw()
    plt.pause(0.01)
    
    if len(cma.p.repelling.archive) > 2:
        input()

def hill_valley(u: Solution, v: Solution, f, n_evals: int):
    max_f = max(u.y, v.y)
    for k in range(1, n_evals + 1):
        a = k / (n_evals + 1)
        x = v.x + a * (u.x - v.x)
        y = f(x)
        if max_f < y:
            return False
    return True


def is_same_basin(u: Solution, v: Solution, eps=1e-2):
    return np.linalg.norm(u.x - v.x) < eps


class ModifiedHimmelblau(ioh.problem.RealSingleObjective):
    optima = np.array(
        [
            [3, 2],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        ]
    )

    def __init__(self, dim, instance):
        self.shift = self.optima[instance % 4]
        super().__init__(
            "ModifiedHimmelblau",
            2,
            instance,
            True,
            ioh.RealBounds(dim, -6, 6),
            [],
            ioh.RealSolution(self.shift, 0.0),
        )

    def evaluate(self, x):
        value = pow(pow(x[0], 2) + x[1] - 11, 2) + pow(x[0] + pow(x[1], 2) - 7, 2)
        sphere = np.linalg.norm(x - self.shift)
        return value + sphere


def himmelblau_exp():
    # c_cmaes.constants.tol_min_sigma = .05
    dim = 2

    problem = ModifiedHimmelblau(dim, 3)
    X, Y, Z = get_meshgrid(problem, -6, 6)
    Z = (Z - problem.optimum.y).clip(1e-8)
    mng = plt.get_current_fig_manager()
    mng.resize(1000, 800)
    problem.reset()

    c_cmaes.utils.set_seed(42)
    modules = c_cmaes.parameters.Modules()
    modules.restart_strategy = c_cmaes.options.RESTART
    modules.bound_correction = c_cmaes.options.SATURATE

    ## Using center placement uniform for now
    modules.center_placement = c_cmaes.options.UNIFORM

    # modules.elitist = True
    # modules.repelling_restart = True
    # modules.center_placement = c_cmaes.options.REPELLING

    settings = c_cmaes.parameters.Settings(
        dim,
        modules,
        sigma0=2.0,
        budget=10_000 * dim,
        target=problem.optimum.y + 1e-8,
    )
    parameters = c_cmaes.Parameters(settings)
    cma = c_cmaes.ModularCMAES(parameters)

    for run in range(1):
        cma = c_cmaes.ModularCMAES(parameters)
        traces = []
        n_sol = 0
        trace = []
        while cma.step(problem):
            if len(cma.p.stats.solutions) != n_sol:
                n_sol += 1
                traces.append(np.array(trace))
                trace = []
            trace.append(cma.p.adaptation.m.copy())

        # Ensure the last restart is included
        solutions = cma.p.stats.solutions + [cma.p.stats.current_best]
        traces.append(np.array(trace))

        final_target = problem.state.current_best.y - problem.optimum.y
        print(
            "final target: ", final_target, "used budget: ", problem.state.evaluations
        )
        plt.figure(figsize=(4, 5))
        plot_contour(X, Y, Z, colorbar=False)
        basins = []

        from collections import Counter

        c = Counter()
        for i, (trace, sol) in enumerate(zip(traces, solutions)):
            new_basin = True

            if i == 0:
                basins.append(sol)
            else:
                for basin in basins:
                    # I needed to reduce the number of points here in
                    if c_cmaes.repelling.hill_valley_test(basin, sol, problem, 5):
                        new_basin = False
                        break
                if new_basin:
                    basins.append(sol)
            if i == len(traces) - 1:
                color = "tab:green"
            elif new_basin:
                color = "yellow"
            else:
                color = "tab:red"
            c[color] += 1
            every = len(trace) // 25
            p = plt.plot(
                trace[::every, 0], trace[::every, 1], linestyle="dashed", color=color
            )
            plt.scatter(
                [trace[0, 0]], [trace[0, 1]], marker="o", color=p[0].get_color()
            )
            plt.scatter(
                [sol.x[0]], [sol.x[1]], s=50, marker="*", color=p[0].get_color()
            )
        print(c)
        plt.title("Modified Himmelblau function")
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(
            handles=[
                matplotlib.patches.Patch(
                    color="yellow", label="Restart converging to new local optimum"
                ),
                matplotlib.patches.Patch(
                    color="tab:red",
                    label="Restart converging to previously found local optimum",
                ),
                matplotlib.patches.Patch(
                    color="tab:green", label="Restart converging to global optimum"
                ),
            ]
        )
        # plt.legend()

        plt.tight_layout()
        plt.savefig("data/himmelblau_rep.pdf", dpi=500)
        plt.show()


def calc_taboo_potential(fid=3, instance=6, dim=2, n_trials=1):

    ## Setting this greatly increases the number of restarts
    c_cmaes.constants.tol_min_sigma = 0.05
    c_cmaes.utils.set_seed(42)

    problem = ioh.get_problem(fid, instance, dim)
    modules = c_cmaes.parameters.Modules()
    modules.restart_strategy = c_cmaes.options.BIPOP
    modules.bound_correction = c_cmaes.options.SATURATE

    ## Using center placement uniform for now
    modules.center_placement = c_cmaes.options.UNIFORM
    settings = c_cmaes.parameters.Settings(
        dim,
        modules,
        sigma0=2.0,
        budget=10_000 * dim,
        target=problem.optimum.y + 1e-8,
    )
    parameters = c_cmaes.Parameters(settings)
    cma = c_cmaes.ModularCMAES(parameters)

    # if dim == 2:
    #     X, Y, Z = get_meshgrid(problem, -5, 5)
    #     Z = Z - problem.optimum.y  # .clip(1e-8, 1e-)
    #     mng = plt.get_current_fig_manager()
    #     mng.resize(1000, 800)
    #     problem.reset()

    for _ in range(n_trials):
        cma.run(problem)
        potential = 0
        n_duplicate_runs = 0
        last_restart = cma.p.stats.solutions[0].e
        basins = [cma.p.stats.solutions[0]]
        for solution in cma.p.stats.solutions[1:]:
            for basin in basins:
                if c_cmaes.repelling.hill_valley_test(basin, solution, problem, 10):
                    potential += solution.e - last_restart
                    n_duplicate_runs += 1
                    # hill_valley(basin, solution, problem, 10)

                    # plt.scatter([basin.x[0]], [basin.x[1]], label="basin")
                    # plt.scatter([solution.x[0]], [solution.x[1]], label="solution")

                    # plot_contour(X, Y, Z)
                    # plt.legend()
                    # plt.show()
                    # breakpoint()
                    break
            else:
                basins.append(solution)
            last_restart = solution.e

        print("number of runs", len(cma.p.stats.solutions))
        print("number of redundant runs", n_duplicate_runs)
        print("number of redundant function evaluations", potential)

        problem.reset()


def interactive(fid=21, instance=6, dim=2, rep=True, coverage=5, save_frames = True):
    lb = -5
    ub = 5

    c_cmaes.utils.set_seed(42)

    problem = ioh.get_problem(fid, instance, dim)

    if dim == 2:
        X, Y, Z = get_meshgrid(problem, lb, ub)
        Z = (Z - problem.optimum.y).clip(1e-8, None)
        mng = plt.get_current_fig_manager()
        mng.resize(1000, 800)
        problem.reset()

    # c_cmaes.constants.sigma_threshold = 0.25
    # c_cmaes.constants.tol_min_sigma = 0.01

    c_cmaes.constants.repelling_current_cov = True
    modules = c_cmaes.parameters.Modules()
    modules.restart_strategy = c_cmaes.options.RESTART
    modules.bound_correction = c_cmaes.options.SATURATE
    modules.elitist = True
    modules.repelling_restart = rep
    modules.center_placement = c_cmaes.options.UNIFORM
    settings = c_cmaes.parameters.Settings(
        dim,
        modules,
        sigma0=2.0,
        budget=10_000 * dim,
        target=problem.optimum.y + 1e-8,
    )
    parameters = c_cmaes.Parameters(settings)
    parameters.repelling.coverage = coverage
    cma = c_cmaes.ModularCMAES(parameters)

    archive_size = 0
    while not cma.break_conditions():
        cma.mutate(problem)
        if dim == 2:
            plot(cma, X, Y, Z, lb, ub, problem)
            if save_frames:
                plt.savefig(os.path.join(
                    base_dir,
                    f"figures/interactive/f{fid}i{instance}r{rep}{cma.p.stats.t:03d}.png"
                ))

        if len(cma.p.repelling.archive) != archive_size:
            archive_size = len(cma.p.repelling.archive)
            for p in cma.p.repelling.archive:
                print(f"({p.radius:.2e}, {p.criticality: .2e})", end=", ")
            print()

            # breakpoint()
            # time.sleep(1)

        cma.select()
        cma.recombine()
        cma.adapt(problem)
    print(problem.optimum)
    print(cma.p.stats.solutions)
    # breakpoint()
    final_target = problem.state.current_best.y - problem.optimum.y
    print("final target: ", final_target, "used budget: ", problem.state.evaluations)

    plt.show()

def collect(
    fid=21,
    dim=2,
    rep=True,
    n_instances=10,
    logged=True,
    coverage=5,
    budget_f=10_000,
    verbose=True,
    n_runs=10,
    elitist=True,
):
    if dim == 0:
        dims = range(2, 10)
    else:
        dims = (dim,)

    if logged:
        algorithm_name = "CMA-ES" + (f"-rep-{coverage}" if rep else "")
        logger = ioh.logger.Analyzer(
            root="data", algorithm_name=algorithm_name, folder_name=algorithm_name
        )

    modules = c_cmaes.parameters.Modules()
    modules.restart_strategy = c_cmaes.options.RESTART
    modules.repelling_restart = rep
    modules.bound_correction = c_cmaes.options.SATURATE
    # c_cmaes.constants.sigma_threshold = .25
    # c_cmaes.constants.tol_min_sigma = .01

    # if rep:
    modules.elitist = elitist
    modules.center_placement = c_cmaes.options.UNIFORM

    if fid == 0:
        fids = list(range(1, 25))
    else:
        fids = (fid,)

    erts = []
    for fid in fids:
        for dim in dims:
            hitting_times = []
            for instance in range(1, n_instances + 1):
                problem = ioh.get_problem(fid, instance, dim)
                c_cmaes.utils.set_seed(42)

                # if instance == 1:
                #     print(problem.meta_data)

                if logged:
                    problem.attach_logger(logger)

                settings = c_cmaes.parameters.Settings(
                    dim,
                    modules,
                    sigma0=2.0,
                    budget=dim * budget_f,
                    target=problem.optimum.y + 1e-8,
                )

                parameters = c_cmaes.Parameters(settings)
                parameters.repelling.coverage = coverage
                cma = c_cmaes.ModularCMAES(parameters)

                cma.run(problem)

                final_target = problem.state.current_best.y - problem.optimum.y
                if verbose:
                    print(
                        "final target:",
                        final_target,
                        "used budget:",
                        problem.state.evaluations,
                        "size of archive:",
                        len(cma.p.repelling.archive),
                        "number of restarts:",
                        len(cma.p.stats.centers),
                    )
                # if problem.state.evaluations > 10_000:
                #     arc = cma.p.repelling.archive[0].solution
                #     print(arc)

                #     for sol in cma.p.stats.centers:
                #         hv = hill_valley(arc, sol, problem, 10)
                #         print(sol, hv, np.linalg.norm(arc.x - sol.x))

                #     breakpoint()
                if final_target > 1e-8:
                    hitting_times.append(dim * budget_f)
                else:
                    hitting_times.append(problem.state.evaluations)
                problem.reset()
            ert = np.sum(hitting_times) / np.sum(np.isfinite(hitting_times))
            # print("ERT", ert)
            erts.append(ert)
            # print()
    return erts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=int, default=21)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n_instances", type=int, default=100)
    parser.add_argument("--instance", type=int, default=6)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--elitist", action="store_true")
    parser.add_argument("--repelling", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--logged", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--coverage", type=float, default=10.0)
    parser.add_argument("--budget", type=int, default=10_000)
    args = parser.parse_args()

    # himmelblau_exp()
    # calc_taboo_potential()
    # exit(0)
    if args.interactive:
        interactive(args.fid, args.instance, args.dim, args.repelling, args.coverage)
    else:
        if args.both:
            ert_no, *_ = collect(
                args.fid,
                args.dim,
                False,
                args.n_instances,
                args.logged,
                args.coverage,
                args.budget,
                args.verbose,
                elitist=args.elitist
            )
            print("no repelling", ert_no)

            if args.coverage == 0:
                for c in (2, 5, 10, 20, 50, 100, 200, 500, 5000):
                    ert_r, *_ = collect(
                        args.fid,
                        args.dim,
                        True,
                        args.n_instances,
                        args.logged,
                        c,
                        args.budget,
                        args.verbose,
                        elitist=args.elitist
                    )
                    print("repelling cov:", c, ert_r, ert_r / ert_no)
            else:
                ert_r, *_ = collect(
                    args.fid,
                    args.dim,
                    True,
                    args.n_instances,
                    args.logged,
                    args.coverage,
                    args.budget,
                    args.verbose,
                    elitist=args.elitist
                )
                print("repelling cov:", args.coverage, ert_r, ert_r / ert_no)
        else:
            collect(
                args.fid,
                args.dim,
                args.repelling,
                args.n_instances,
                args.logged,
                args.coverage,
                args.budget,
                args.verbose,
            )


    # dims = range(2, 10)
    # n_instances = 10
    # n_runs = 10
    # fids = (args.fid,)
    # budget_f = 10_000
    # time.sleep(np.random.uniform(0, 3))
    # algorithm_name = "CMA-ES"
    # if args.repelling:
    #     algorithm_name += f"-rep-c{args.coverage}"
    # if args.elitist:
    #     algorithm_name += "-elitist"

    # if args.logged:
    #     logger = ioh.logger.Analyzer(
    #         root="data/rep", algorithm_name=algorithm_name, folder_name=algorithm_name
    #     )

    # modules = c_cmaes.parameters.Modules()
    # modules.restart_strategy = c_cmaes.options.RESTART
    # modules.center_placement = c_cmaes.options.UNIFORM
    # modules.bound_correction = c_cmaes.options.SATURATE
    # modules.repelling_restart = args.repelling
    # modules.elitist = args.elitist
    # c_cmaes.constants.repelling_current_cov = True
    # # c_cmaes.constants.sigma_threshold = .25
    # # c_cmaes.constants.tol_min_sigma = .01

    # data = []
    # for fid in fids:
    #     print(fid, end=": ")
    #     for dim in dims:
    #         print(dim, end=", ")
    #         sys.stdout.flush()
    #         for instance in range(1, n_instances + 1):
    #             problem = ioh.get_problem(fid, instance, dim)
    #             if args.logged:
    #                 problem.attach_logger(logger)

    #             for run in range(n_runs):
    #                 c_cmaes.utils.set_seed(42 * run)

    #                 settings = c_cmaes.parameters.Settings(
    #                     dim,
    #                     modules,
    #                     sigma0=2.0,
    #                     budget=dim * budget_f,
    #                     target=problem.optimum.y + 1e-8,
    #                 )

    #                 parameters = c_cmaes.Parameters(settings)
    #                 parameters.repelling.coverage = args.coverage
    #                 cma = c_cmaes.ModularCMAES(parameters)

    #                 cma.run(problem)

    #                 final_target = problem.state.current_best.y - problem.optimum.y

    #                 potential, n_duplicate_runs = calculate_potential(
    #                     cma.p.stats.centers,
    #                     ioh.get_problem(fid, instance, dim),
    #                 )
    #                 data.append(
    #                     (
    #                         fid,
    #                         instance,
    #                         dim,
    #                         42 * run,
    #                         cma.p.stats.evaluations,
    #                         final_target,
    #                         final_target < 1e-8,
    #                         len(cma.p.stats.centers),
    #                         n_duplicate_runs,
    #                         potential,
    #                     )
    #                 )
    #                 problem.reset()
    #     print()
            

    # df = pd.DataFrame(data, columns="fid, instance, dim, seed, evaluations, final_target, target_reached, n_runs, n_duplicate_runs, potential".split(", "))
    # df.to_pickle(f"data/rep/{algorithm_name}_fid{args.fid}.pkl")