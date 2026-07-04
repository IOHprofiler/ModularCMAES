# evaluate_solution_csvs.py

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from problem.ProblemMM import ProblemMM


FILENAME_RE = re.compile(
    r"pid(?P<pid>\d+)_pin_(?P<pin>\d+)_dim_(?P<dim>\d+)\.csv$"
)


@dataclass
class RunResult:
    file: Path
    pid: int
    pin: int
    dim: int
    ngm: int
    nsol: int
    rpr: float
    precision: float
    f1: float
    overall: float
    missing_optima: int


def parse_solution_filename(path: Path):
    match = FILENAME_RE.match(path.name)

    if match is None:
        return None

    return {
        "pid": int(match.group("pid")),
        "pin": int(match.group("pin")),
        "dim": int(match.group("dim")),
    }


def read_solutions_csv(path: Path, dim: int):
    """
    Reads rows formatted as:

        x_1, x_2, ..., x_dim, y

    No header.
    """
    X = []
    y = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader, 1):
            if not row:
                continue

            if len(row) != dim + 1:
                raise ValueError(
                    f"{path}: row {row_idx} has {len(row)} columns, "
                    f"expected {dim + 1}"
                )

            values = [float(v) for v in row]
            X.append(values[:dim])
            y.append(values[dim])

    if not X:
        return np.empty((0, dim), dtype=float), np.empty((0,), dtype=float)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def niche_radii(optima_X: np.ndarray):
    """
    R_nich,k = 0.5 * min_{j != k} ||x*_k - x*_j||
    """
    optima_X = np.asarray(optima_X, dtype=float)

    if optima_X.ndim == 1:
        optima_X = optima_X.reshape(1, -1)

    n = optima_X.shape[0]

    if n == 1:
        return np.array([np.inf], dtype=float)

    distances = np.linalg.norm(
        optima_X[:, None, :] - optima_X[None, :, :],
        axis=2,
    )

    np.fill_diagonal(distances, np.inf)

    return 0.5 * np.min(distances, axis=1)


def pda_from_error_ln(
    error: float,
    *,
    eps_tight: float = 1e-5,
    eps_loose: float = 1.0,
):
    """
    Official PDA interpolation:

        PDA = min(1, max(0,
            (ln(eps_loose) - ln(error))
            /
            (ln(eps_loose) - ln(eps_tight))
        ))

    with:
        error <= eps_tight -> PDA = 1
        error >= eps_loose -> PDA = 0
    """
    error = float(error)

    if error <= eps_tight:
        return 1.0

    if error >= eps_loose:
        return 0.0

    value = (
        (np.log(eps_loose) - np.log(error))
        /
        (np.log(eps_loose) - np.log(eps_tight))
    )

    return float(min(1.0, max(0.0, value)))


def calculate_rpr_official(
    X_reported: np.ndarray,
    y_eval: np.ndarray,
    optima_X: np.ndarray,
    f_min: float,
    *,
    eps_tight: float = 1e-5,
    eps_loose: float = 1.0,
):
    """
    Official RPR calculation.

    For each known optimum x*_k:
        1. Compute R_nich,k.
        2. Find all reported solutions within R_nich,k.
        3. Use the best function value among those solutions.
        4. Convert its error to PDA.
        5. RPR is the average PDA over all known optima.
    """
    X_reported = np.asarray(X_reported, dtype=float)
    y_eval = np.asarray(y_eval, dtype=float)
    optima_X = np.asarray(optima_X, dtype=float)

    if optima_X.ndim == 1:
        optima_X = optima_X.reshape(1, -1)

    ngm = optima_X.shape[0]

    if X_reported.size == 0:
        return 0.0, np.zeros(ngm), np.full(ngm, np.inf)

    radii = niche_radii(optima_X)

    pda_per_optimum = np.zeros(ngm, dtype=float)
    best_error_per_optimum = np.full(ngm, np.inf, dtype=float)

    for k in range(ngm):
        distances = np.linalg.norm(X_reported - optima_X[k], axis=1)
        in_niche = distances < radii[k]

        if not np.any(in_niche):
            continue

        best_y = float(np.min(y_eval[in_niche]))
        error = max(0.0, best_y - float(f_min))

        best_error_per_optimum[k] = error

        pda_per_optimum[k] = pda_from_error_ln(
            error,
            eps_tight=eps_tight,
            eps_loose=eps_loose,
        )

    rpr = float(np.mean(pda_per_optimum))

    return rpr, pda_per_optimum, best_error_per_optimum


def evaluate_file(
    path: Path,
    *,
    eps_tight: float = 1e-5,
    eps_loose: float = 1.0,
    trust_csv_y: bool = False,
):
    info = parse_solution_filename(path)
    if info is None:
        raise ValueError(f"Invalid filename format: {path.name}")

    pid = info["pid"]
    pin = info["pin"]
    dim = info["dim"]

    problem = ProblemMM(pid, pin, dim)
    problem.form()

    optima_X = np.asarray(problem.minima.X, dtype=float)
    f_min = float(problem.minima.f)

    if optima_X.ndim == 1:
        optima_X = optima_X.reshape(1, -1)

    ngm = int(optima_X.shape[0])

    X_reported, y_reported = read_solutions_csv(path, dim)
    nsol = int(X_reported.shape[0])

    if nsol == 0:
        return RunResult(
            file=path,
            pid=pid,
            pin=pin,
            dim=dim,
            ngm=ngm,
            nsol=nsol,
            rpr=0.0,
            precision=0.0,
            f1=0.0,
            overall=0.0,
            missing_optima=ngm,
        )

    if trust_csv_y:
        y_eval = y_reported
    else:
        used_eval_before = getattr(problem, "used_eval", None)

        y_eval = np.array(
            [problem._func_eval_single(x) for x in X_reported],
            dtype=float,
        )

        if used_eval_before is not None:
            problem.used_eval = used_eval_before

    rpr, pda_per_optimum, best_error_per_optimum = calculate_rpr_official(
        X_reported,
        y_eval,
        optima_X,
        f_min,
        eps_tight=eps_tight,
        eps_loose=eps_loose,
    )

    if nsol > 0:
        precision = float(rpr * ngm / nsol)
        precision = min(1.0, max(0.0, precision))
    else:
        precision = 0.0

    if precision + rpr > 0.0:
        f1 = float(2.0 * precision * rpr / (precision + rpr))
    else:
        f1 = 0.0

    overall = 0.5 * (rpr + f1)
    missing_optima = int(np.sum(pda_per_optimum <= 0.0))

    return RunResult(
        file=path,
        pid=pid,
        pin=pin,
        dim=dim,
        ngm=ngm,
        nsol=nsol,
        rpr=rpr,
        precision=precision,
        f1=f1,
        overall=overall,
        missing_optima=missing_optima,
    )


def find_solution_files(folder: Path):
    files = []

    for path in sorted(folder.glob("pid*_pin_*_dim_*.csv")):
        if parse_solution_filename(path) is not None:
            files.append(path)

    return files


def evaluate_folder(
    folder: Path,
    *,
    eps_tight: float = 1e-5,
    eps_loose: float = 1.0,
    trust_csv_y: bool = False,
):
    files = find_solution_files(folder)

    results = []

    for path in files:
        result = evaluate_file(
            path,
            eps_tight=eps_tight,
            eps_loose=eps_loose,
            trust_csv_y=trust_csv_y,
        )
        results.append(result)

    return results


def print_aggregates_by_function_dim(results: list[RunResult]):
    if not results:
        return

    groups = {}

    for r in results:
        key = (r.pid, r.dim)
        groups.setdefault(key, []).append(r)

    print()
    print("Aggregated over instances: function/dim")
    print("-" * 105)
    print(
        f"{'pid':>3s} {'dim':>3s} {'n':>4s} "
        f"{'Mean RPR':>10s} {'Mean Prec':>10s} {'Mean F1':>10s} "
        f"{'Mean Overall':>12s} {'Final Score':>12s} "
        f"{'Mean Nsol':>10s} {'Mean Missing':>12s}"
    )
    print("-" * 105)

    for (pid, dim), group in sorted(groups.items()):
        rprs = np.array([r.rpr for r in group], dtype=float)
        precisions = np.array([r.precision for r in group], dtype=float)
        f1s = np.array([r.f1 for r in group], dtype=float)
        overalls = np.array([r.overall for r in group], dtype=float)
        nsols = np.array([r.nsol for r in group], dtype=float)
        missing = np.array([r.missing_optima for r in group], dtype=float)

        # Same scoring convention: average over all RPR and F1 indicators.
        final_score = float(np.mean(np.r_[rprs, f1s]))

        print(
            f"{pid:3d} {dim:3d} {len(group):4d} "
            f"{float(np.mean(rprs)):10.6f} "
            f"{float(np.mean(precisions)):10.6f} "
            f"{float(np.mean(f1s)):10.6f} "
            f"{float(np.mean(overalls)):12.6f} "
            f"{final_score:12.6f} "
            f"{float(np.mean(nsols)):10.2f} "
            f"{float(np.mean(missing)):12.2f}"
        )

    print("-" * 105)
    print()

def print_aggregates_by_function_dim(results: list[RunResult]):
    if not results:
        return

    groups = {}

    for r in results:
        key = (r.pid, r.dim)
        groups.setdefault(key, []).append(r)

    print()
    print("Aggregated over instances: function/dim")
    print("-" * 105)
    print(
        f"{'pid':>3s} {'dim':>3s} {'n':>4s} "
        f"{'Mean RPR':>10s} {'Mean Prec':>10s} {'Mean F1':>10s} "
        f"{'Mean Overall':>12s} {'Final Score':>12s} "
        f"{'Mean Nsol':>10s} {'Mean Missing':>12s}"
    )
    print("-" * 105)

    for (pid, dim), group in sorted(groups.items()):
        rprs = np.array([r.rpr for r in group], dtype=float)
        precisions = np.array([r.precision for r in group], dtype=float)
        f1s = np.array([r.f1 for r in group], dtype=float)
        overalls = np.array([r.overall for r in group], dtype=float)
        nsols = np.array([r.nsol for r in group], dtype=float)
        missing = np.array([r.missing_optima for r in group], dtype=float)

        # Same scoring convention: average over all RPR and F1 indicators.
        final_score = float(np.mean(np.r_[rprs, f1s]))

        print(
            f"{pid:3d} {dim:3d} {len(group):4d} "
            f"{float(np.mean(rprs)):10.6f} "
            f"{float(np.mean(precisions)):10.6f} "
            f"{float(np.mean(f1s)):10.6f} "
            f"{float(np.mean(overalls)):12.6f} "
            f"{final_score:12.6f} "
            f"{float(np.mean(nsols)):10.2f} "
            f"{float(np.mean(missing)):12.2f}"
        )

    print("-" * 105)
    print()

def print_results(results: list[RunResult], per_file: bool = False):
    if not results:
        print("No valid solution CSV files found.")
        return
    
    if per_file:
        print()
        print("Per-file results")
        print("-" * 120)
        print(
            f"{'file':35s} "
            f"{'pid':>3s} {'pin':>3s} {'dim':>3s} "
            f"{'NGM':>4s} {'Nsol':>5s} "
            f"{'RPR':>10s} {'Prec':>10s} {'F1':>10s} {'Overall':>10s} "
            f"{'Missing':>7s}"
        )
        print("-" * 120)

        for r in results:
            print(
                f"{r.file.name:35s} "
                f"{r.pid:3d} {r.pin:3d} {r.dim:3d} "
                f"{r.ngm:4d} {r.nsol:5d} "
                f"{r.rpr:10.6f} {r.precision:10.6f} "
                f"{r.f1:10.6f} {r.overall:10.6f} "
                f"{r.missing_optima:7d}"
            )

    rprs = np.array([r.rpr for r in results], dtype=float)
    f1s = np.array([r.f1 for r in results], dtype=float)
    overalls = np.array([r.overall for r in results], dtype=float)

    # Spec: final score is the average of all RPR and F1 indicators.
    final_score = float(np.mean(np.r_[rprs, f1s]))

    print("-" * 120)
    print(f"Files evaluated: {len(results)}")
    print(f"Mean RPR:        {float(np.mean(rprs)):.8f}")
    print(f"Mean F1:         {float(np.mean(f1s)):.8f}")
    print(f"Mean overall:    {float(np.mean(overalls)):.8f}")
    print(f"Final score:     {final_score:.8f}")
    print()


def write_summary_csv(results: list[RunResult], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "file",
                "pid",
                "pin",
                "dim",
                "NGM",
                "Nsol",
                "RPR",
                "precision",
                "F1",
                "overall",
                "missing_optima",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r.file.name,
                    r.pid,
                    r.pin,
                    r.dim,
                    r.ngm,
                    r.nsol,
                    r.rpr,
                    r.precision,
                    r.f1,
                    r.overall,
                    r.missing_optima,
                ]
            )


def write_function_dim_summary_csv(results: list[RunResult], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = {}

    for r in results:
        key = (r.pid, r.dim)
        groups.setdefault(key, []).append(r)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "pid",
                "dim",
                "n_instances_present",
                "mean_RPR",
                "mean_precision",
                "mean_F1",
                "mean_overall",
                "final_score",
                "mean_Nsol",
                "mean_missing_optima",
            ]
        )

        for (pid, dim), group in sorted(groups.items()):
            rprs = np.array([r.rpr for r in group], dtype=float)
            precisions = np.array([r.precision for r in group], dtype=float)
            f1s = np.array([r.f1 for r in group], dtype=float)
            overalls = np.array([r.overall for r in group], dtype=float)
            nsols = np.array([r.nsol for r in group], dtype=float)
            missing = np.array([r.missing_optima for r in group], dtype=float)

            final_score = float(np.mean(np.r_[rprs, f1s]))

            writer.writerow(
                [
                    pid,
                    dim,
                    len(group),
                    float(np.mean(rprs)),
                    float(np.mean(precisions)),
                    float(np.mean(f1s)),
                    float(np.mean(overalls)),
                    final_score,
                    float(np.mean(nsols)),
                    float(np.mean(missing)),
                ]
            )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing pidXX_pin_XX_dim_XX.csv files.",
    )

    parser.add_argument(
        "--eps-tight",
        type=float,
        default=1e-5,
        help="Tight function-value error threshold.",
    )

    parser.add_argument(
        "--eps-loose",
        type=float,
        default=1.0,
        help="Loose function-value error threshold.",
    )

    parser.add_argument(
        "--trust-csv-y",
        action="store_true",
        help="Use y values from the CSV instead of recomputing f(x).",
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to write a per-file summary CSV.",
    )
    parser.add_argument(
        "--function-dim-summary-csv",
        type=Path,
        default=None,
        help="Optional path to write function/dim aggregate summary CSV.",
    )

    args = parser.parse_args()

    results = evaluate_folder(
        args.folder,
        eps_tight=args.eps_tight,
        eps_loose=args.eps_loose,
        trust_csv_y=args.trust_csv_y,
    )

    print_results(results)
    print_aggregates_by_function_dim(results)

    if args.summary_csv is not None:
        write_summary_csv(results, args.summary_csv)
        print(f"Wrote summary CSV: {args.summary_csv}")

    if args.function_dim_summary_csv is not None:
        write_function_dim_summary_csv(results, args.function_dim_summary_csv)
        print(f"Wrote function/dim summary CSV: {args.function_dim_summary_csv}")


if __name__ == "__main__":
    main()