"""Module containing tests for ModularCMA-ES C++ Bounds."""

import unittest
import numpy as np
import modcma
from modcma import c_maes
from modcma.c_maes import bounds, center, Population, Parameters, parameters, options


class TestBounds(unittest.TestCase):
    """Test case for Bounds of Modular C++ CMA-ES."""

    __bound_fixers = (
        bounds.COTN,
        bounds.UniformResample,
        bounds.Mirror,
        bounds.Saturate,
        bounds.Toroidal,
    )

    __bound_fixers_options = (
        options.CorrectionMethod.COTN,
        options.CorrectionMethod.UNIFORM_RESAMPLE,
        options.CorrectionMethod.MIRROR,
        options.CorrectionMethod.SATURATE,
        options.CorrectionMethod.TOROIDAL,
    )

    def setUp(self):
        self.lb, self.ub = np.zeros(2), np.ones(2) * 2
        self.par = Parameters(parameters.Settings(2, lambda0=2, lb=self.lb, ub=self.ub))
        self.par.pop.S = np.ones((2, 2)) * 2
        self.par.adaptation.m = np.ones(2) * 0.1
        Z = np.ones((2, 2)) * 0.9
        Z[0, 0] *= 2
        self.par.pop.Z = Z
        self.par.pop.Y = self.par.pop.Z.copy()
        self.par.pop.X_transformed = self.par.pop.X = self.par.adaptation.m + (
            self.par.pop.S * self.par.pop.Y
        )

    def test_bound_fixers(self):
        for i, (boundcntrl, option) in enumerate(
            zip(self.__bound_fixers, self.__bound_fixers_options)
        ):
            if i < 2:
                method = boundcntrl(self.lb.size)
            else:
                method = boundcntrl()

            self.par.settings.modules.bound_correction = option
            method.correct(1, self.par)
            self.assertEqual(method.n_out_of_bounds, 0)
            method.correct(0, self.par)
            self.assertEqual(method.n_out_of_bounds, 1)
            self.assertTrue(np.all(self.par.pop.X <= 2))
            self.assertTrue(np.all(np.isclose(self.par.pop.Y.ravel()[1:], 0.9)))
            self.assertTrue(np.all(np.isclose(self.par.pop.X.ravel()[1:], 1.9)))
            self.setUp()

    def test_do_nothing(self):
        method = bounds.NoCorrection()
        method.correct(0, self.par)
        self.assertEqual(method.n_out_of_bounds, 1)
        self.assertFalse(np.all(self.par.pop.X <= 2))
        self.assertTrue(np.all(np.isclose(self.par.pop.Y.ravel()[1:], 0.9)))
        self.assertTrue(np.all(np.isclose(self.par.pop.X.ravel()[1:], 1.9)))

    def test_center(self):
        self.assertTrue(np.all(self.par.settings.center == 1))
        self.assertTrue(np.all(self.par.settings.lb == 0))
        self.assertTrue(np.all(self.par.settings.ub == 2))
        self.assertTrue(np.all(self.par.settings.db == 2))


# Infinite and half-infinite boxes, and integer variables in the box (#79: F19, F20, F21)

INF = np.inf
CORRECTIONS = ("SATURATE", "MIRROR", "COTN", "UNIFORM_RESAMPLE", "TOROIDAL")
DRIVERS = ("run", "ask_tell")


def make(d, seed=5, budget=600, **kwargs):
    return modcma.ModularCMAES(modcma.settings_from_dict(d, seed=seed, budget=budget, **kwargs))


def evaluated(driver, objective, d, criterion=None, **kwargs):
    """The optimizer after a whole optimization with run() or ask-tell, and every point it
    evaluated, in order"""
    points = []

    def recorded(x):
        points.append(np.array(x, dtype=float))
        return objective(points[-1])

    es = make(d, **kwargs)
    if criterion is not None:
        es.p.criteria.items.append(criterion)
    if driver == "run":
        es.run(recorded)
    else:
        while True:
            X = es.ask()
            if es.tell([recorded(x) for x in X]) and es.break_conditions():
                break
    return es, np.array(points)


def evaluated_both(test, objective, d, **kwargs):
    """The points of run() and of ask-tell, which are the same, bit for bit"""
    es, X = evaluated("run", objective, d, **kwargs)
    _, X_ask_tell = evaluated("ask_tell", objective, d, **kwargs)
    np.testing.assert_array_equal(X, X_ask_tell)
    return es, X


def inside(X, lb, ub):
    return np.all((X >= lb) & (X <= ub), axis=1)


class TestHalfInfiniteBox(unittest.TestCase):
    """F20: each finite side of a box with infinite bounds is corrected"""

    lb, ub = np.array([0.0, -INF]), np.array([INF, 1.0])

    @staticmethod
    def corner(x):
        # The minimum, (-3, 3), is outside the box: the best point in it is the corner (0, 1)
        return float(np.sum((x - np.array([-3.0, 3.0])) ** 2))

    def test_correct_x_per_side(self):
        settings = parameters.Settings(
            3, lb=np.array([0.0, -INF, 0.0]), ub=np.array([INF, 1.0, 2.0])
        )
        x = np.array([-0.5, 1.7, 2.5])
        expected = {"SATURATE": 2.0, "MIRROR": 1.5, "TOROIDAL": 0.5}
        for name in CORRECTIONS:
            with self.subTest(method=name):
                method = {
                    "SATURATE": bounds.Saturate(),
                    "MIRROR": bounds.Mirror(),
                    "COTN": bounds.COTN(3),
                    "UNIFORM_RESAMPLE": bounds.UniformResample(3),
                    "TOROIDAL": bounds.Toroidal(),
                }[name]
                oob = method.is_out_of_bounds(x, settings)
                np.testing.assert_array_equal(oob, [True, True, True])
                y = method.correct_x(x, oob, 0.5, settings)
                # The half-infinite coordinates: the finite bound they crossed (SATURATE)
                np.testing.assert_array_equal(y[:2], [0.0, 1.0])
                # The finite coordinate: the method
                self.assertTrue(0.0 <= y[2] <= 2.0, y)
                if name in expected:
                    self.assertEqual(y[2], expected[name])
                # The infinite sides are never crossed
                far = np.array([1e300, -1e300, 1.0])
                np.testing.assert_array_equal(method.is_out_of_bounds(far, settings), [False] * 3)

    def test_runs_stay_in_the_box(self):
        for name in CORRECTIONS:
            with self.subTest(method=name):
                es, X = evaluated_both(
                    self, self.corner, 2, bound_correction=name, lb=self.lb, ub=self.ub
                )
                self.assertTrue(np.all(inside(X, self.lb, self.ub)))
                np.testing.assert_allclose(es.p.stats.global_best.x, [0.0, 1.0], atol=1e-6)

    def test_without_correction_points_leave_the_box(self):
        es, X = evaluated_both(self, self.corner, 2, lb=self.lb, ub=self.ub)
        self.assertFalse(np.all(inside(X, self.lb, self.ub)))
        np.testing.assert_allclose(es.p.stats.global_best.x, [-3.0, 3.0], atol=1e-4)

    def test_resample(self):
        # At most 5 d samples per point: a few points stay outside the box
        _, X = evaluated_both(
            self, self.corner, 2, bound_correction="RESAMPLE", lb=self.lb, ub=self.ub
        )
        self.assertGreater(np.mean(inside(X, self.lb, self.ub)), 0.9)

    def test_out_of_bounds_are_counted(self):
        es = make(2, bound_correction="SATURATE", lb=self.lb, ub=self.ub)
        outside = 0
        while es.step(self.corner):
            outside += es.p.bounds.n_out_of_bounds
        self.assertGreater(outside, 0)


class PlacedCenters(c_maes.restart.Criterion):
    """Restarts every 5 generations and records the center of each new run"""

    def __init__(self):
        super().__init__("PlacedCenters")
        self.centers = []

    def update(self, parameters):
        self.met = parameters.stats.t_run >= 5

    def on_reset(self, parameters):
        self.centers.append(parameters.adaptation.m.copy())


class TestCenterPlacementInInfiniteBoxes(unittest.TestCase):
    """F19: UNIFORM, NOVELTY_WEIGHTED and MAXIMIN_TABOO in boxes with infinite bounds place the
    centers in x0 +- 2 sigma0, moved inside a finite bound"""

    PLACEMENTS = {
        "UNIFORM": center.Uniform,
        "NOVELTY_WEIGHTED": center.NoveltyWeighted,
        "MAXIMIN_TABOO": center.MaximinTaboo,
    }
    # (lb, ub, the interval of the centers, with x0 = Settings.center = 0 and sigma0 = 2)
    BOXES = {
        "infinite": ([-INF, -INF], [INF, INF], [[-4, -4], [4, 4]]),
        "half": ([0.0, -INF], [INF, 1.0], [[0, -7], [8, 1]]),
        "x0 outside": ([10.0, -INF], [INF, -10.0], [[10, -18], [18, -10]]),
    }

    @staticmethod
    def rastrigin(x):
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

    def test_region(self):
        # Per coordinate: lb only, ub only, neither, both; x0 inside the finite bound or not
        settings = parameters.Settings(
            4,
            lb=np.array([0.0, -INF, -INF, -3.0]),
            ub=np.array([INF, 1.0, INF, 3.0]),
            x0=np.array([5.0, 5.0, 1.0, 0.0]),
            sigma0=1.0,
        )
        low, high = np.array([3.0, -3.0, -1.0, -3.0]), np.array([7.0, 1.0, 3.0, 3.0])
        for name, placement in self.PLACEMENTS.items():
            with self.subTest(placement=name):
                p = Parameters(settings)
                M = []
                for _ in range(300):
                    placement()(p)
                    M.append(p.adaptation.m.copy())
                M = np.array(M)
                self.assertTrue(np.all(inside(M, low, high)), (M.min(0), M.max(0)))
                # It covers the interval
                np.testing.assert_allclose(M.min(0), low, atol=0.2)
                np.testing.assert_allclose(M.max(0), high, atol=0.2)

    def test_restart_centers(self):
        for name in self.PLACEMENTS:
            repelling = "NONE" if name == "UNIFORM" else "COVERAGE"
            for box, (lb, ub, interval) in self.BOXES.items():
                for driver in DRIVERS:
                    with self.subTest(placement=name, box=box, driver=driver):
                        criterion = PlacedCenters()
                        es, X = evaluated(
                            driver,
                            self.rastrigin,
                            2,
                            criterion=criterion,
                            budget=1500,
                            restart_strategy="RESTART",
                            center_placement=name,
                            repelling_type=repelling,
                            bound_correction="SATURATE",
                            lb=np.array(lb),
                            ub=np.array(ub),
                        )
                        self.assertTrue(np.all(np.isfinite(X)))
                        self.assertTrue(np.all(inside(X, lb, ub)))
                        # The first entry is the start of the optimizer, at x0
                        centers = np.array(criterion.centers[1:])
                        self.assertGreater(len(centers), 3)
                        self.assertTrue(np.all(inside(centers, *interval)), centers)
                        if driver == "run":
                            X_run = X
                        else:
                            np.testing.assert_array_equal(X, X_run)


class TestIntegerVariablesInTheBox(unittest.TestCase):
    """F21: an integer variable is rounded to the nearest integer in [ceil(lb), floor(ub)]"""

    def test_no_integer_in_the_box_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer variable 0 has no integer value"):
            parameters.Settings(2, lb=np.array([0.6, 0.0]), ub=np.array([0.9, 1.0]),
                                integer_variables=[0])
        settings = parameters.Settings(
            2, lb=np.array([0.6, 0.0]), ub=np.array([1.3, 1.0]), integer_variables=[0]
        )
        # The setters check their own field; the box is checked when the optimizer is created
        settings.ub = np.array([0.9, 1.0])
        with self.assertRaisesRegex(ValueError, "integer variable 0"):
            modcma.ModularCMAES(settings)
        settings.ub = np.array([1.3, 1.0])
        settings.lb = np.array([1.1, 0.0])
        with self.assertRaisesRegex(ValueError, "integer variable 0"):
            modcma.ModularCMAES(settings)
        # A continuous variable may have such bounds; infinite bounds have integers
        parameters.Settings(2, lb=np.array([0.6, -INF]), ub=np.array([0.9, INF]),
                            integer_variables=[1])

    def test_mapping(self):
        mapping = parameters.CoordinateMapping([0, 1], 3)
        lb, ub = np.array([0.3, -2.6, 0.0]), np.array([2.7, INF, 1.0])
        cases = [
            # (x, rounded without bounds, rounded with bounds)
            ([0.4, -2.55, 0.4], [0.0, -3.0, 0.4], [1.0, -2.0, 0.4]),
            ([2.6, 7.5, 0.5], [3.0, 8.0, 0.5], [2.0, 8.0, 0.5]),
            ([1.5, -2.4, 0.5], [2.0, -2.0, 0.5], [2.0, -2.0, 0.5]),
            # Outside the box (bound correction NONE): the nearest integer, outside
            ([0.2, -2.9, 0.5], [0.0, -3.0, 0.5], [0.0, -3.0, 0.5]),
        ]
        for x, plain, bounded in cases:
            np.testing.assert_array_equal(mapping.transform(np.array(x)), plain)
            np.testing.assert_array_equal(mapping.transform(np.array(x), lb, ub), bounded)

    def test_runs_stay_in_the_box(self):
        lb = np.array([0.3, -2.6, -INF, -1.0])
        ub = np.array([2.7, INF, 0.4, 1.0])
        integer = [0, 1, 2]
        target = np.array([-1.0, -5.0, 3.0, 0.5])

        def objective(x):
            return float(np.sum((x - target) ** 2))

        for name in CORRECTIONS:
            with self.subTest(method=name):
                es, X = evaluated_both(
                    self, objective, 4, bound_correction=name, lb=lb, ub=ub,
                    integer_variables=integer,
                )
                self.assertTrue(np.all(inside(X, lb, ub)))
                np.testing.assert_array_equal(X[:, integer], np.round(X[:, integer]))
                np.testing.assert_allclose(es.p.stats.global_best.x, [1.0, -2.0, 0.0, 0.5],
                                           atol=1e-3)


class TestDegenerateBox(unittest.TestCase):
    """#133: a fixed coordinate (lb_i == ub_i) stays at lb_i; an empty coordinate is rejected"""

    lb, ub = np.array([0.0, 1.0, -2.0]), np.array([1.0, 1.0, 2.0])

    @staticmethod
    def sphere(x):
        return float(np.sum(np.asarray(x) ** 2))

    def test_fixed_coordinate(self):
        for name in CORRECTIONS:
            with self.subTest(method=name):
                es, X = evaluated_both(
                    self, self.sphere, 3, bound_correction=name, lb=self.lb, ub=self.ub,
                    budget=200, sigma0=2.0,
                )
                self.assertFalse(np.isnan(X).any())
                self.assertTrue(np.all(X[:, 1] == 1.0))
                self.assertTrue(np.all(inside(X, self.lb, self.ub)))
                self.assertEqual(es.p.stats.global_best.x[1], 1.0)

    def test_fixed_coordinate_infinite_neighbours(self):
        lb, ub = np.array([-INF, 1.0, 0.0]), np.array([1.0, 1.0, INF])
        for name in CORRECTIONS:
            with self.subTest(method=name):
                _, X = evaluated_both(
                    self, self.sphere, 3, bound_correction=name, lb=lb, ub=ub, budget=200,
                )
                self.assertFalse(np.isnan(X).any())
                self.assertTrue(np.all(X[:, 1] == 1.0))

    def test_empty_coordinate_in_settings(self):
        cases = {
            "lb = ub = inf": ([INF, 0.0], [INF, 1.0]),
            "lb = ub = -inf": ([-INF, 0.0], [-INF, 1.0]),
            "lb = inf": ([INF, 0.0], [INF, 1.0]),
            "ub = -inf": ([0.0, 0.0], [-INF, 1.0]),
            "lb > ub": ([2.0, 0.0], [1.0, 1.0]),
            "nan": ([np.nan, 0.0], [1.0, 1.0]),
        }
        for name, (lb, ub) in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                parameters.Settings(2, lb=np.array(lb), ub=np.array(ub))

    def test_setters_check_their_own_field(self):
        settings = parameters.Settings(2, lb=np.zeros(2), ub=np.ones(2))
        for bad in ([INF, 0.0], [np.nan, 0.0]):
            with self.subTest(lb=bad), self.assertRaises(ValueError):
                settings.lb = np.array(bad)
        for bad in ([-INF, 1.0], [np.nan, 1.0]):
            with self.subTest(ub=bad), self.assertRaises(ValueError):
                settings.ub = np.array(bad)
        np.testing.assert_array_equal(settings.lb, [0.0, 0.0])
        np.testing.assert_array_equal(settings.ub, [1.0, 1.0])

    def test_either_order(self):
        for first in ("lb", "ub"):
            with self.subTest(first=first):
                settings = parameters.Settings(2)
                new = {"lb": np.array([6.0, 6.0]), "ub": np.array([10.0, 10.0])}
                if first == "ub":
                    new = dict(reversed(list(new.items())))
                for name, value in new.items():  # the first assignment crosses the old box
                    setattr(settings, name, value)
                np.testing.assert_array_equal(settings.db, [4.0, 4.0])
                modcma.ModularCMAES(settings)

    def test_empty_box_rejected_when_the_optimizer_is_created(self):
        cases = {
            "lb > ub": ([2.0, 0.0], [1.0, 1.0]),
        }
        for name, (lb, ub) in cases.items():
            with self.subTest(case=name):
                settings = parameters.Settings(2)
                settings.lb = np.array(lb)
                settings.ub = np.array(ub)
                with self.assertRaises(ValueError):
                    modcma.ModularCMAES(settings)
                with self.assertRaises(ValueError):
                    Parameters(settings)

    def test_integer_variables_checked_when_the_optimizer_is_created(self):
        settings = parameters.Settings(2, integer_variables=[0])
        settings.lb = np.array([0.6, 0.0])
        settings.ub = np.array([0.9, 1.0])
        with self.assertRaisesRegex(ValueError, "integer variable 0"):
            modcma.ModularCMAES(settings)

    def test_fixed_coordinate_set_later(self):
        settings = parameters.Settings(2, lb=np.zeros(2), ub=np.ones(2))
        settings.lb = np.array([1.0, 0.0])  # lb_0 == ub_0: a fixed variable
        np.testing.assert_array_equal(settings.db, [0.0, 1.0])
        modcma.ModularCMAES(settings)


if __name__ == "__main__":
    unittest.main()
