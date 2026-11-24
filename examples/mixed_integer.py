"""A mixed-integer problem: choose integer counts and continuous sizes.

A toy design problem: x[0], x[1] are the integer numbers of two kinds of components (0 to 20),
x[2], x[3] continuous sizes. The cost couples the first count with the first size, so the best size
depends on the count; the optimum is at the counts (7, 12) with the sizes (3 / 7.4, -1.25). The
integer coordinates are rounded before every evaluation, and their step size has a lower bound so
the search does not stall on one integer value.
"""

import numpy as np

import modcma


def cost(x):
    count_a, count_b, size_a, size_b = x
    assert count_a == int(count_a) and count_b == int(count_b)  # the objective sees integers
    return float(
        (count_a - 7) ** 2
        + 0.5 * (count_b - 12) ** 2
        + 3 * (size_a - 0.5) ** 2
        + (size_b + 1.25) ** 2
        + 0.1 * count_a * size_a**2
    )


if __name__ == "__main__":
    d = 4
    modules = modcma.Modules()
    modules.bound_correction = modcma.options.CorrectionMethod.SATURATE
    settings = modcma.Settings(
        d,
        integer_variables=[0, 1],
        lb=np.array([0.0, 0.0, -5.0, -5.0]),
        ub=np.array([20.0, 20.0, 5.0, 5.0]),
        x0=np.array([10.0, 10.0, 0.0, 0.0]),
        sigma0=3.0,
        budget=5_000,
        seed=1,
        modules=modules,
    )
    es = modcma.ModularCMAES(settings)
    es.run(cost)

    best = es.p.stats.global_best
    print(f"best x = {best.x}, f = {best.y:.6f}, after {es.p.stats.evaluations} evaluations")
