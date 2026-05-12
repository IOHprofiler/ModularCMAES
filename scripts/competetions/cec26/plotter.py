# optimizer_plotter.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def get_meshgrid_from_problem_mm(problem, delta: float = 0.05):
    """
    Build a 2D meshgrid for ProblemMM.

    Uses problem._func_eval_single(...).

    Note:
    If evaluating the mesh increments problem.used_eval, this restores used_eval
    afterwards.
    """
    if problem.dim != 2:
        raise ValueError("Contour plotting only works for dim == 2")

    lb = float(np.min(problem.low_bound))
    ub = float(np.max(problem.up_bound))

    used_eval_before = getattr(problem, "used_eval", None)

    x = np.arange(lb, ub + delta, delta)
    y = np.arange(lb, ub + delta, delta)

    if hasattr(problem, "minima") and hasattr(problem.minima, "X"):
        optima = np.asarray(problem.minima.X)

        if optima.ndim == 2 and optima.shape[1] == 2:
            x = np.sort(np.unique(np.r_[x, optima[:, 0]]))
            y = np.sort(np.unique(np.r_[y, optima[:, 1]]))

    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problem._func_eval_single(
                np.array([X[i, j], Y[i, j]])
            )

    if used_eval_before is not None:
        problem.used_eval = used_eval_before

    return X, Y, Z, lb, ub


def _ellipse_from_cov(
    center,
    C,
    sigma: float,
    scale: float,
    *,
    edgecolor,
    linewidth=2.0,
    linestyle="dashed",
    zorder=5,
    animated=False,
):
    """
    Create an ellipse for:

        distance_C(x, center) / sigma = scale

    Matplotlib Ellipse expects full width/height, hence the factor 2.
    """
    center = np.asarray(center, dtype=float)
    C = np.asarray(C, dtype=float)

    if center.size != 2:
        raise ValueError("center must be 2-dimensional")

    if C.shape != (2, 2):
        raise ValueError("C must be a 2x2 covariance matrix")

    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-16)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    width, height = 2.0 * scale * sigma * np.sqrt(vals)

    return Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
        animated=animated,
    )


def prepare_contour_values(
    Z,
    *,
    optimum_y: float | None = None,
    transform: str = "log1p_shift",
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
    eps: float = 1e-12,
):
    """
    Convert raw objective values into stable values for contour plotting.

    transform options:
        "raw"
            Plot raw Z.

        "shift"
            Plot Z - optimum_y if provided, otherwise Z - min(Z).

        "log10_shift"
            Plot log10(max(Z - optimum_y, eps)) if optimum_y provided,
            otherwise log10(max(Z - min(Z), eps)).

        "log1p_shift"
            Plot log1p(max(Z - optimum_y, 0)) if optimum_y provided,
            otherwise log1p(max(Z - min(Z), 0)).

        "rank"
            Plot rank-normalized values in [0, 1].
            Very robust, but loses magnitude information.

    Returns
    -------
    Z_plot, vmin, vmax
    """
    Z = np.asarray(Z, dtype=float)
    finite = np.isfinite(Z)

    if not np.any(finite):
        raise ValueError("Z contains no finite values")

    Z_safe = Z.copy()

    # Replace non-finite values with the worst finite value.
    worst = np.nanmax(Z_safe[finite])
    Z_safe[~finite] = worst

    if optimum_y is None or not np.isfinite(optimum_y):
        shift = np.nanmin(Z_safe)
    else:
        shift = float(optimum_y)

    if transform == "raw":
        Z_plot = Z_safe

    elif transform == "shift":
        Z_plot = Z_safe - shift

    elif transform == "log10_shift":
        Z_plot = np.log10(np.maximum(Z_safe - shift, eps))

    elif transform == "log1p_shift":
        Z_plot = np.log1p(np.maximum(Z_safe - shift, 0.0))

    elif transform == "rank":
        flat = Z_safe.ravel()
        order = np.argsort(flat)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(flat))
        Z_plot = ranks.reshape(Z_safe.shape)

    else:
        raise ValueError(
            f"Unknown contour transform {transform!r}. "
            "Use 'raw', 'shift', 'log10_shift', 'log1p_shift', or 'rank'."
        )

    finite_plot = np.isfinite(Z_plot)

    if robust_percentiles is None:
        vmin = float(np.nanmin(Z_plot[finite_plot]))
        vmax = float(np.nanmax(Z_plot[finite_plot]))
    else:
        lo, hi = robust_percentiles
        vmin, vmax = np.nanpercentile(Z_plot[finite_plot], [lo, hi])
        vmin = float(vmin)
        vmax = float(vmax)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(Z_plot[finite_plot]))
        vmax = float(np.nanmax(Z_plot[finite_plot]))

    if vmin == vmax:
        vmax = vmin + 1.0

    return Z_plot, vmin, vmax

class ModCMABlitPlotter:
    """
    Interactive 2D plotter for ModularCMAES / RR-CMA-ES.

    Supports:
        - NoRepelling
        - CoverageRepelling
        - AdaptiveRepelling
    """

    def __init__(
        self,
        X,
        Y,
        Z,
        *,
        lb: float,
        ub: float,
        optimum_y: float = 0.0,
        title: str = "Modular CMA-ES",
        colorbar: bool = True,
        vmin: float | None = -2,
        vmax: float | None = 2,
        use_blit: bool = False,
        contour_transform: str = "log1p_shift",
        robust_percentiles: tuple[float, float] | None = (1.0, 99.0),
    ):
        self.X = X
        self.Y = Y
        self.lb = float(lb)
        self.ub = float(ub)
        self.optimum_y = float(optimum_y)

        self.contour_transform = contour_transform
        self.robust_percentiles = robust_percentiles

        self.Z_plot, self.vmin, self.vmax = prepare_contour_values(
            Z,
            optimum_y=self.optimum_y,
            transform=contour_transform,
            robust_percentiles=robust_percentiles,
        )

        self.fig, (self.ax, self.info_ax) = plt.subplots(
            1,
            2,
            figsize=(14, 8),
            gridspec_kw={"width_ratios": [4, 1.35]},
        )

        self.info_ax.axis("off")
        self.use_blit = bool(use_blit and self.fig.canvas.supports_blit)

        self.ax.set_title(title)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(self.lb, self.ub)
        self.ax.set_ylim(self.lb, self.ub)
        self.ax.set_xlabel(r"$x_1$")
        self.ax.set_ylabel(r"$x_2$")
        self.ax.grid(True)

        self.contour = self.ax.contourf(
            self.X,
            self.Y,
            self.Z_plot,
            levels=200,
            cmap="Spectral",
            zorder=-10,
            vmin=self.vmin,
            vmax=self.vmax,
            extend="both",
        )

        if colorbar:
            self.fig.colorbar(self.contour, ax=self.ax, fraction=0.046, pad=0.04)

        animated = self.use_blit

        self.mean_artist = self.ax.scatter(
            [],
            [],
            color="m",
            s=80,
            label="current mean",
            animated=animated,
            zorder=30,
        )

        self.population_artist = self.ax.scatter(
            [],
            [],
            color="m",
            alpha=0.45,
            s=35,
            label="population",
            animated=animated,
            zorder=25,
        )

        self.current_best_artist = self.ax.scatter(
            [],
            [],
            color="black",
            marker="x",
            s=100,
            label="current best",
            animated=animated,
            zorder=35,
        )

        self.global_best_artist = self.ax.scatter(
            [],
            [],
            color="black",
            marker="D",
            s=75,
            label="global best",
            animated=animated,
            zorder=35,
        )

        self.archive_centers_artist = self.ax.scatter(
            [],
            [],
            color="black",
            marker="o",
            s=35,
            label="tabu archive centers",
            animated=animated,
            zorder=36,
        )

        self.stats_text = self.info_ax.text(
            0.0,
            1.0,
            "plotter initialized",
            transform=self.info_ax.transAxes,
            va="top",
            ha="left",
            animated=animated,
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
            zorder=100,
        )

        self.dynamic_patches: list[Ellipse] = []
        self.background = None
        self._needs_background = True

        if self.use_blit:
            self.fig.canvas.mpl_connect("draw_event", self._on_draw)

        self.fig.tight_layout()
        plt.ion()
        plt.show(block=False)
        plt.pause(0.05)

        if self.use_blit:
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self._needs_background = False
        else:
            self.fig.canvas.draw_idle()
            plt.pause(0.05)

        print("Matplotlib backend:", plt.get_backend())
        print("Canvas supports blit:", self.fig.canvas.supports_blit)
        print("Blit enabled:", self.use_blit)

    @classmethod
    def from_problem_mm(
        cls,
        problem,
        *,
        delta: float = 0.05,
        colorbar: bool = True,
        title: str | None = None,
        use_blit: bool = False,
        vmin: float | None = -2,
        vmax: float | None = 2,
        contour_transform: str = "log1p_shift",
        robust_percentiles: tuple[float, float] | None = (1.0, 99.0),
    ):
        X, Y, Z, lb, ub = get_meshgrid_from_problem_mm(problem, delta=delta)

        if title is None:
            title = "ProblemMM contour"

        return cls(
            X,
            Y,
            Z,
            lb=lb,
            ub=ub,
            optimum_y=float(problem.minima.f),
            title=title,
            colorbar=colorbar,
            use_blit=use_blit,
            vmin=vmin,
            vmax=vmax,
            contour_transform=contour_transform,
            robust_percentiles=robust_percentiles,
        )

    def _on_draw(self, event):
        self._needs_background = True

    def _clear_dynamic_patches(self):
        for patch in self.dynamic_patches:
            try:
                patch.remove()
            except ValueError:
                pass

        self.dynamic_patches.clear()

    @staticmethod
    def _empty_offsets():
        return np.empty((0, 2))

    def _set_scatter_point(self, artist, x):
        if x is None:
            artist.set_offsets(self._empty_offsets())
            return

        x = np.asarray(x, dtype=float)

        if x.size != 2:
            artist.set_offsets(self._empty_offsets())
            return

        artist.set_offsets(x.reshape(1, 2))

    def _get_population_points(self, par):
        pop_X = np.asarray(par.pop.X, dtype=float)

        if pop_X.ndim != 2:
            return self._empty_offsets()

        if pop_X.shape[0] == 2:
            return pop_X.T

        if pop_X.shape[1] == 2:
            return pop_X

        return self._empty_offsets()

    def _get_archive(self, par):
        repelling = getattr(par, "repelling", None)
        return list(getattr(repelling, "archive", [])) if repelling is not None else []

    def _repelling_kind(self, repelling):
        if repelling is None:
            return "none"

        cls_name = type(repelling).__name__.lower()

        if "adaptive" in cls_name:
            return "adaptive"

        if "coverage" in cls_name:
            return "coverage"

        if "norepelling" in cls_name or "no" in cls_name:
            return "none"

        # Attribute-based fallback.
        if hasattr(repelling, "local_max_rejection_rate") or hasattr(repelling, "grow_eta"):
            return "adaptive"

        if hasattr(repelling, "coverage"):
            return "coverage"

        return "unknown"

    def _update_archive_centers(self, par):
        archive = self._get_archive(par)

        archive_centers = []

        for tabu_point in archive:
            center = np.asarray(tabu_point.solution.x, dtype=float)

            if center.size == 2:
                archive_centers.append(center)

        if archive_centers:
            self.archive_centers_artist.set_offsets(np.vstack(archive_centers))
        else:
            self.archive_centers_artist.set_offsets(self._empty_offsets())

    def _add_current_distribution_ellipses(self, m, C, sigma, animated):
        for scale in (0.5, 1.0, 2.0, 3.0):
            patch = _ellipse_from_cov(
                m,
                C,
                sigma,
                scale,
                edgecolor="m",
                linewidth=2,
                linestyle="dashed",
                zorder=20,
                animated=animated,
            )
            self.ax.add_patch(patch)
            self.dynamic_patches.append(patch)

    def _add_archive_ellipses(self, par, C, sigma, animated):
        """
        Add tabu archive regions.

        The plotted ellipse scale is radius, not 2 * radius.
        _ellipse_from_cov already converts radius to full ellipse width.
        """
        repelling = getattr(par, "repelling", None)
        archive = self._get_archive(par)
        kind = self._repelling_kind(repelling)

        attempts = int(getattr(repelling, "attempts", 0)) if repelling is not None else 0

        criticality_threshold = 0.01

        local_max_rejection_rate = (
            float(getattr(repelling, "local_max_rejection_rate", np.inf))
            if repelling is not None
            else np.inf
        )

        for tabu_point in archive:
            radius = float(getattr(tabu_point, "radius", np.nan))

            if not np.isfinite(radius):
                continue

            center = np.asarray(tabu_point.solution.x, dtype=float)

            if center.size != 2:
                continue

            criticality = float(getattr(tabu_point, "criticality", 1.0))
            active = criticality >= criticality_threshold

            last_rejection_rate = float(getattr(tabu_point, "last_rejection_rate", 0.0))
            overactive = (
                kind == "adaptive"
                and np.isfinite(local_max_rejection_rate)
                and last_rejection_rate > local_max_rejection_rate
            )

            if kind == "coverage":
                if active:
                    base_color = "black"
                    effective_color = "gray"
                    base_linestyle = "dashed"
                    effective_linestyle = "solid"
                    base_linewidth = 2.0
                    effective_linewidth = 1.5
                else:
                    base_color = "gray"
                    effective_color = "lightgray"
                    base_linestyle = "dotted"
                    effective_linestyle = "dotted"
                    base_linewidth = 1.0
                    effective_linewidth = 1.0

            elif kind == "adaptive":
                if overactive:
                    base_color = "red"
                    effective_color = "red"
                    base_linestyle = "dashed"
                    effective_linestyle = "solid"
                    base_linewidth = 2.5
                    effective_linewidth = 2.0
                elif active:
                    base_color = "black"
                    effective_color = "gray"
                    base_linestyle = "dashed"
                    effective_linestyle = "solid"
                    base_linewidth = 2.0
                    effective_linewidth = 1.5
                else:
                    base_color = "gray"
                    effective_color = "lightgray"
                    base_linestyle = "dotted"
                    effective_linestyle = "dotted"
                    base_linewidth = 1.0
                    effective_linewidth = 1.0

            else:
                base_color = "black" if active else "gray"
                effective_color = "gray" if active else "lightgray"
                base_linestyle = "dashed" if active else "dotted"
                effective_linestyle = "solid" if active else "dotted"
                base_linewidth = 2.0 if active else 1.0
                effective_linewidth = 1.5 if active else 1.0

            base_patch = _ellipse_from_cov(
                center,
                C,
                sigma,
                radius,
                edgecolor=base_color,
                linewidth=base_linewidth,
                linestyle=base_linestyle,
                zorder=22,
                animated=animated,
            )
            self.ax.add_patch(base_patch)
            self.dynamic_patches.append(base_patch)

            shrinkage = float(getattr(tabu_point, "shrinkage", 1.0))
            effective_radius = shrinkage**attempts * radius

            effective_patch = _ellipse_from_cov(
                center,
                C,
                sigma,
                effective_radius,
                edgecolor=effective_color,
                linewidth=effective_linewidth,
                linestyle=effective_linestyle,
                zorder=23,
                animated=animated,
            )
            self.ax.add_patch(effective_patch)
            self.dynamic_patches.append(effective_patch)

    def _archive_summary(self, par):
        repelling = getattr(par, "repelling", None)
        archive = self._get_archive(par)
        kind = self._repelling_kind(repelling)

        active_archive = 0
        total_n_rep = 0
        total_duplicate_evals = 0.0
        total_rejected = 0
        total_checked = 0

        radii = []
        min_radii = []
        max_radii = []
        last_rejection_rates = []

        local_max_rejection_rate = (
            float(getattr(repelling, "local_max_rejection_rate", np.inf))
            if repelling is not None
            else np.inf
        )

        overactive_archive = 0

        for point in archive:
            criticality = float(getattr(point, "criticality", 1.0))

            if criticality >= 0.01:
                active_archive += 1

            total_n_rep += int(getattr(point, "n_rep", 1))
            total_duplicate_evals += float(getattr(point, "duplicate_evaluations", 0.0))
            total_rejected += int(getattr(point, "rejected_count", 0))
            total_checked += int(getattr(point, "checked_count", 0))

            radius = float(getattr(point, "radius", np.nan))
            if np.isfinite(radius):
                radii.append(radius)

            min_radius = float(getattr(point, "min_radius", np.nan))
            if np.isfinite(min_radius):
                min_radii.append(min_radius)

            max_radius = float(getattr(point, "max_radius", np.nan))
            if np.isfinite(max_radius):
                max_radii.append(max_radius)

            last_rejection_rate = float(getattr(point, "last_rejection_rate", np.nan))
            if np.isfinite(last_rejection_rate):
                last_rejection_rates.append(last_rejection_rate)

                if (
                    kind == "adaptive"
                    and np.isfinite(local_max_rejection_rate)
                    and last_rejection_rate > local_max_rejection_rate
                ):
                    overactive_archive += 1

        return {
            "kind": kind,
            "active_archive": active_archive,
            "overactive_archive": overactive_archive,
            "total_n_rep": total_n_rep,
            "total_duplicate_evals": total_duplicate_evals,
            "total_rejected": total_rejected,
            "total_checked": total_checked,
            "mean_radius": float(np.mean(radii)) if radii else np.nan,
            "max_radius": float(np.max(radii)) if radii else np.nan,
            "mean_min_radius": float(np.mean(min_radii)) if min_radii else np.nan,
            "mean_max_radius": float(np.mean(max_radii)) if max_radii else np.nan,
            "mean_last_rejection_rate": (
                float(np.mean(last_rejection_rates)) if last_rejection_rates else np.nan
            ),
            "max_last_rejection_rate": (
                float(np.max(last_rejection_rates)) if last_rejection_rates else np.nan
            ),
        }

    def _update_stats_text(self, par, current_best_y, global_best_y, m, sigma, pop_points):
        stats = getattr(par, "stats", None)
        repelling = getattr(par, "repelling", None)

        generation = getattr(stats, "t", "?")
        evaluations = getattr(stats, "evaluations", "?")
        lamb = getattr(par, "lamb", getattr(par, "lambda", "?"))

        archive = self._get_archive(par)
        attempts = int(getattr(repelling, "attempts", 0)) if repelling is not None else 0

        coverage = (
            float(getattr(repelling, "coverage", np.nan))
            if repelling is not None
            else np.nan
        )

        summary = self._archive_summary(par)
        kind = summary["kind"]

        lines = [
            f"repelling: {kind}",
            f"generation: {generation}",
            f"evals: {evaluations}",
            f"lambda: {lamb}",
            f"sigma: {sigma:.3e}",
            f"archive size: {len(archive)}",
            f"active archive: {summary['active_archive']}",
            f"total n_rep: {summary['total_n_rep']}",
            f"rejections this gen: {attempts}",
            f"population points: {len(pop_points)}",
        ]

        if np.isfinite(summary["mean_radius"]):
            lines.append(f"mean radius: {summary['mean_radius']:.3e}")
        else:
            lines.append("mean radius: n/a")

        if np.isfinite(summary["max_radius"]):
            lines.append(f"max radius: {summary['max_radius']:.3e}")
        else:
            lines.append("max radius: n/a")

        if kind == "coverage":
            if np.isfinite(coverage):
                lines.append(f"coverage c: {coverage:.3g}")
            else:
                lines.append("coverage c: n/a")

        elif kind == "adaptive":
            local_max_rejection_rate = float(
                getattr(repelling, "local_max_rejection_rate", np.nan)
            )
            max_rejection_rate = float(
                getattr(repelling, "max_rejection_rate", np.nan)
            )
            grow_eta = float(getattr(repelling, "grow_eta", np.nan))
            shrink_factor = float(getattr(repelling, "shrink_factor", np.nan))
            avg_restart_evals = float(
                getattr(repelling, "average_restart_evaluations", np.nan)
            )

            lines.extend(
                [
                    f"overactive archive: {summary['overactive_archive']}",
                    f"duplicate evals: {summary['total_duplicate_evals']:.0f}",
                    f"local rejects/checks: {summary['total_rejected']}/{summary['total_checked']}",
                ]
            )

            if np.isfinite(summary["mean_last_rejection_rate"]):
                lines.append(
                    f"mean local rej-rate: {summary['mean_last_rejection_rate']:.2f}"
                )

            if np.isfinite(summary["max_last_rejection_rate"]):
                lines.append(
                    f"max local rej-rate: {summary['max_last_rejection_rate']:.2f}"
                )

            if np.isfinite(local_max_rejection_rate):
                lines.append(f"local max rej-rate: {local_max_rejection_rate:.2f}")

            if np.isfinite(max_rejection_rate):
                lines.append(f"global max rej-rate: {max_rejection_rate:.2f}")

            if np.isfinite(grow_eta):
                lines.append(f"grow eta: {grow_eta:.3g}")

            if np.isfinite(shrink_factor):
                lines.append(f"shrink factor: {shrink_factor:.3g}")

            if np.isfinite(avg_restart_evals):
                lines.append(f"avg restart evals: {avg_restart_evals:.1f}")

            if np.isfinite(summary["mean_min_radius"]):
                lines.append(f"mean min radius: {summary['mean_min_radius']:.3e}")

            if np.isfinite(summary["mean_max_radius"]):
                lines.append(f"mean max radius: {summary['mean_max_radius']:.3e}")

        else:
            if np.isfinite(coverage):
                lines.append(f"coverage c: {coverage:.3g}")

        if np.asarray(m).size == 2:
            lines.append(f"mean: [{m[0]:.3f}, {m[1]:.3f}]")
        else:
            lines.append("mean: invalid")

        if current_best_y is not None:
            lines.append(f"current best - f*: {current_best_y:.3e}")

        if global_best_y is not None:
            lines.append(f"global best - f*: {global_best_y:.3e}")

        self.stats_text.set_text("\n".join(lines))

    def update(self, es, *, force_draw: bool = False):
        if not plt.fignum_exists(self.fig.number):
            return

        par = es.p
        animated = self.use_blit

        self._clear_dynamic_patches()

        m = np.asarray(par.adaptation.m, dtype=float)
        C = np.asarray(par.adaptation.C, dtype=float)
        sigma = float(par.mutation.sigma)

        if m.size == 2:
            self.mean_artist.set_offsets(m.reshape(1, 2))
        else:
            self.mean_artist.set_offsets(self._empty_offsets())

        pop_points = self._get_population_points(par)
        self.population_artist.set_offsets(pop_points)

        current_best = getattr(par.stats, "current_best", None)
        global_best = getattr(par.stats, "global_best", None)

        current_best_y = None
        global_best_y = None

        if current_best is not None and hasattr(current_best, "x"):
            self._set_scatter_point(self.current_best_artist, current_best.x)

            if hasattr(current_best, "y"):
                current_best_y = float(current_best.y) - self.optimum_y
        else:
            self._set_scatter_point(self.current_best_artist, None)

        if global_best is not None and hasattr(global_best, "x"):
            self._set_scatter_point(self.global_best_artist, global_best.x)

            if hasattr(global_best, "y"):
                global_best_y = float(global_best.y) - self.optimum_y
        else:
            self._set_scatter_point(self.global_best_artist, None)

        self._update_archive_centers(par)

        if m.size == 2 and C.shape == (2, 2):
            self._add_current_distribution_ellipses(m, C, sigma, animated)
            self._add_archive_ellipses(par, C, sigma, animated)

        self._update_stats_text(
            par,
            current_best_y,
            global_best_y,
            m,
            sigma,
            pop_points,
        )

        if self.use_blit and not force_draw:
            if self._needs_background or self.background is None:
                self.fig.canvas.draw()
                self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
                self._needs_background = False

            self.fig.canvas.restore_region(self.background)

            artists = [
                self.mean_artist,
                self.population_artist,
                self.current_best_artist,
                self.global_best_artist,
                self.archive_centers_artist,
                *self.dynamic_patches,
            ]

            for artist in artists:
                self.ax.draw_artist(artist)

            self.info_ax.draw_artist(self.stats_text)

            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()

        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

    def finalize(self):
        artists = [
            self.mean_artist,
            self.population_artist,
            self.current_best_artist,
            self.global_best_artist,
            self.archive_centers_artist,
            self.stats_text,
            *self.dynamic_patches,
        ]

        for artist in artists:
            artist.set_animated(False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)