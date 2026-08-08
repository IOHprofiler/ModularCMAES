from itertools import product
import csv
from multiprocessing import Pool
import numpy as np
import matplotlib
from pathlib import Path

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from modcma import c_maes
from problem.ProblemMM import ProblemMM

from plotter import ModCMABlitPlotter


class ReachedPrecision(c_maes.restart.Criterion):

    def __init__(self, precision):
        super().__init__("ReachedPrecision")
        self.precision = precision

    def update(self, par: c_maes.Parameters):
        self.met = par.stats.current_best.y <= self.precision


def count_unique_archive_optima(
    archive,
    target_y: float,
    *,
    distance_tol: float = 1e-3,
):
    good_solutions = []

    for tabu_point in archive:
        sol = tabu_point.solution

        if float(sol.y) <= target_y:
            good_solutions.append(sol)

    unique_solutions = []

    for sol in good_solutions:
        x = np.asarray(sol.x, dtype=float)

        is_new = True

        for existing in unique_solutions:
            x_existing = np.asarray(existing.x, dtype=float)

            if np.linalg.norm(x - x_existing) <= distance_tol:
                is_new = False
                break

        if is_new:
            unique_solutions.append(sol)

    return unique_solutions, good_solutions


def archive_found_all_optima(
    archive,
    n_unique_optima: int,
    target_y: float,
    *,
    distance_tol: float = 1e-3,
    verbose: bool = True,
):

    unique_solutions, good_solutions = count_unique_archive_optima(
        archive,
        target_y,
        distance_tol=distance_tol,
    )

    found = len(unique_solutions)
    success = found >= n_unique_optima

    if verbose:
        print(
            f"Found {found}/{n_unique_optima} unique archive optima "
            f"with y <= {target_y:.3e}"
        )
        print(f"Good archive solutions: {len(good_solutions)}")
        print(f"Archive size: {len(archive)}")

        for i, sol in enumerate(unique_solutions, 1):
            x = np.asarray(sol.x, dtype=float)
            print(f"{i:>2}. y={float(sol.y):.6e}, x={x}")

    return success, unique_solutions


def write_found_solutions_csv(
    solutions,
    *,
    pid: int,
    pin: int,
    dim: int,
    out_dir: str | Path = ".",
    overwrite: bool = True,
):
    """
    Write found solutions as CSV rows:

        x_1, x_2, ..., x_dim, y

    No header.

    Filename:
        pid01_pin_01_dim_02.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"pid{pid:02d}_pin_{pin:02d}_dim_{dim:02d}.csv"
    path = out_dir / filename

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        for sol in solutions:
            x = np.asarray(sol.x, dtype=float).reshape(-1)
            y = float(sol.y)

            if x.size != dim:
                raise ValueError(
                    f"Solution has dimension {x.size}, expected {dim}: {x}"
                )

            writer.writerow([*x, y])

    return path


def run_modcma(
    fid,
    iid,
    dim,
    *,
    interactive=False,
    plot_every=5,
    lambda0=None,
    repelling_type: str = "COVERAGE",  # ADAPTIVE, COVERAGE
    restart_strategy: str = "RESTART",
    center_placement: str = "NOVELTY_WEIGHTED",  # UNIFORM, MAXIMIN_TABOO, NOVELTY_WEIGHTED
    elitist: bool = False,
    check_per_iteration: bool = False,
):

    problem = ProblemMM(fid, iid, dim)
    problem.form()

    settings = c_maes.settings_from_dict(
        dim,
        **dict(
            repelling_type=repelling_type,
            restart_strategy=restart_strategy,
            center_placement=center_placement,
            lambda0=lambda0,
            elitist=elitist,
            bound_correction="RESAMPLE",
            budget=20_000 * dim,
            sigma0=1.0,
            lb=np.ones(dim) * problem.low_bound,
            ub=np.ones(dim) * problem.up_bound,
            target=problem.minima.f,
        ),
    )

    target_y = problem.minima.f + 1e-5
    es = c_maes.ModularCMAES(settings)

    c1 = ReachedPrecision(target_y)
    es.p.criteria.items.append(c1)

    es.p.repelling.coverage = 2

    plotter = None
    if interactive and dim == 2:
        check_per_iteration = True
        plt.ion()
        plotter = ModCMABlitPlotter.from_problem_mm(
            problem,
            delta=0.05,
            colorbar=True,
            title=f"ProblemMM fid={fid}, iid={iid}, dim={dim}",
        )

    n_solutions = 0
    iteration = 0

    while not es.break_conditions():
        es.step(problem._func_eval_single)
        iteration += 1

        if check_per_iteration:
            if (archive_size := len(es.p.repelling.archive)) != n_solutions:
                n_solutions = archive_size

                success, unique_archive_optima = archive_found_all_optima(
                    es.p.repelling.archive,
                    n_unique_optima=problem.minima.X.shape[0],
                    target_y=target_y,
                    verbose=interactive,
                )

                if success:
                    break

            if es.p.criteria.any() and interactive:
                print("reason", es.p.criteria.reason())

            if plotter is not None:
                if lambda0 != 1:
                    if iteration % plot_every != 0:
                        continue
                else:
                    if not es.p.stats.has_improved:
                        continue

                plotter.update(es)

    success, unique_archive_optima = archive_found_all_optima(
        es.p.repelling.archive,
        n_unique_optima=problem.minima.X.shape[0],
        target_y=target_y,
        verbose=True,
    )
    csv_path = write_found_solutions_csv(
        unique_archive_optima,
        pid=fid,
        pin=iid,
        dim=dim,
        out_dir=f"solutions_{repelling_type}_{center_placement}_{restart_strategy}_{lambda0}_elitist{elitist}",
    )
    print(fid, iid, dim, es.p.stats.evaluations, csv_path)

    if plotter is not None:
        plotter.update(es, force_draw=True)
        plt.ioff()
        plt.show()


def main():
    functions = tuple(range(1, 17))
    instances = tuple(range(1, 16))
    dimensions = (2,)  # 5, 10, 20)

    c_maes.utils.set_seed(69)
    settings = tuple(product(functions, instances, dimensions))

    with Pool(30) as p:
        p.starmap(run_modcma, settings)


if __name__ == "__main__":
    # main()

    run_modcma(1, 1, 2, interactive=True)
