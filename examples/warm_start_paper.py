"""The synthetic experiment of the WS-CMA-ES paper, for a comparison with its Table 1.

Nomura, Watanabe, Akimoto, Ozaki and Onishi, "Warm starting CMA-ES for hyperparameter
optimization", AAAI 2021, Section 4.1 and Appendices A and B: d = 2, lambda = 8, the best value of
the first 50 evaluations, 20 runs, alpha = gamma = 0.1, the cold start N(0.5, 0.2^2) in every
coordinate. The target task has b = 0.6, the source tasks b = 0.4, ..., 0.8:

- sphere: f(x) = (x1 - b)^2 + (x2 - b)^2;
- rotated ellipsoid: f(x) = f_ell(R x), f_ell(y) = (y1 - b)^2 + 25 (y2 - b)^2, R the rotation by
  pi/6.

The source solutions are a random search of 100 points, uniform in [0, 1]^2: an assumption, since
the paper gives the number of source points only for its hyperparameter experiments (100). Seeds:
run i (0..19) uses seed 1000 + i for the source points and seed i + 1 for the optimizer.

Every run, cold and warm, uses the active update: the paper's CMA-ES is pycma with its default
options (Appendix A), which include the active update (CMA_active=True); modcma's default does not.
"""

import numpy as np

import modcma

D, LAMBDA, EVALUATIONS, RUNS = 2, 8, 50, 20
SOURCE_POINTS = 100
ANGLE = np.pi / 6
ROTATION = np.array([[np.cos(ANGLE), -np.sin(ANGLE)], [np.sin(ANGLE), np.cos(ANGLE)]])

# Table 1 of the paper: mean and standard error, scaled, for the cold start and b = 0.4, ..., 0.8
PAPER = {
    "sphere": ["0.43 ± 0.1", "1.3 ± 0.3", "0.26 ± 0.06", "0.073 ± 0.02", "0.27 ± 0.07",
               "0.82 ± 0.2"],
    "rotated ellipsoid": ["0.25 ± 0.1", "0.26 ± 0.06", "0.21 ± 0.06", "0.14 ± 0.05",
                          "0.15 ± 0.06", "0.38 ± 0.2"],
}


def sphere(b):
    return lambda x: float((x[0] - b) ** 2 + (x[1] - b) ** 2)


def rotated_ellipsoid(b):
    def f(x):
        y = ROTATION @ x
        return float((y[0] - b) ** 2 + 25.0 * (y[1] - b) ** 2)

    return f


def best_of_first_evaluations(f, seed, x0, sigma0, C0=None):
    """The best value among the first 50 evaluations. The run gets a larger budget: every run ends
    with an evaluation of its center of mass, which must not be one of the 50."""
    values = []

    def recorded(x):
        values.append(f(x))
        return values[-1]

    settings = modcma.settings_from_dict(
        D, active=True, x0=x0, sigma0=sigma0, C0=C0, lambda0=LAMBDA, budget=2 * EVALUATIONS,
        seed=seed,
    )
    modcma.ModularCMAES(settings).run(recorded)
    return min(values[:EVALUATIONS])


def mean_and_error(values, scale):
    values = np.asarray(values) * scale
    return f"{values.mean():.2g} ± {values.std(ddof=1) / np.sqrt(len(values)):.2g}"


def main():
    print(f"Best f of the first {EVALUATIONS} evaluations, mean ± standard error of {RUNS} runs")
    sources = (0.4, 0.5, 0.6, 0.7, 0.8)
    print(f"{'':22s}{'cold':>14s}" + "".join(f"{f'b = {b}':>14s}" for b in sources))
    problems = [("sphere", sphere, 3), ("rotated ellipsoid", rotated_ellipsoid, 2)]
    for name, problem, exponent in problems:
        scale = 10.0**exponent
        target = problem(0.6)
        cold = [best_of_first_evaluations(target, i + 1, np.full(D, 0.5), 0.2) for i in range(RUNS)]
        row = [mean_and_error(cold, scale)]
        for b in sources:
            source = problem(b)
            warm = []
            for i in range(RUNS):
                X = np.random.default_rng(1000 + i).uniform(0, 1, size=(SOURCE_POINTS, D))
                ws = modcma.warm_start([(x, source(x)) for x in X], gamma=0.1, alpha=0.1)
                warm.append(best_of_first_evaluations(target, i + 1, **ws))
            row.append(mean_and_error(warm, scale))
        print(f"{name} (x 1e{exponent})")
        print(f"  {'modcma':20s}" + "".join(f"{v:>14s}" for v in row))
        print(f"  {'paper, Table 1':20s}" + "".join(f"{v:>14s}" for v in PAPER[name]))


if __name__ == "__main__":
    main()
