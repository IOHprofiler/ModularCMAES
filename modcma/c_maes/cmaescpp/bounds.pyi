"""
Bound correction: what happens to samples outside the box [lb, ub].

The optimizer holds one as `es.p.bounds`, selected by `Modules.bound_correction` (`options.CorrectionMethod`). Each coordinate and each side is corrected separately: an infinite bound is never crossed, and a coordinate with one infinite bound is set to its finite bound when it crosses it (the methods other than SATURATE need the width ub - lb). A fixed coordinate, lb_i = ub_i, is set to lb_i (#144).

The samples are corrected by `BoundCorrection.correct`; TPA's two points and the point of gradient injection by `BoundCorrection.correct_point`. The methods follow Caraffini et al. (2019); they are described in de Nobel et al. (2021) and in the framework paper (de Nobel et al. 2026, Sec. "Boundary Correction").
"""

from typing import Annotated

import numpy
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.parameters
import modcma.c_maes.cmaescpp.sampling


class BoundCorrection:
    """
    Base class of the bound corrections.

    After a sample x = m + sigma y is formed, `correct` counts it when it lies outside [lb, ub] and, unless the method is NONE, moves the coordinates outside into the box (`correct_x`) and recomputes y and z of the individual from the corrected x, so the updates learn from the point that is evaluated. RESAMPLE moves nothing: the optimizer samples the individual again (`Resample`).
    """

    @property
    def n_out_of_bounds(self) -> int:
        """
        The number of samples of the current population that were outside the bounds (reset at the start of a population; with RESAMPLE every rejected sample counts).
        """

    def correct(self, index: int, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Correct individual `index` of the population in place.

        Counts it in `n_out_of_bounds` when it is outside the bounds and, unless the method is NONE, applies `correct_x`; X, X_transformed, Y and Z of the individual are then updated, y = (x - m) / s with the individual's step sizes s and z = A^-1 y.

        Parameters
        ----------
        index : int
            The column of the individual in `parameters.pop`.
        parameters : Parameters
            The state of the optimizer (`es.p`).
        """

    def correct_x(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], oob: Annotated[NDArray[numpy.bool_], dict(shape=(None,), order='C')], sigma: float, settings: modcma.c_maes.cmaescpp.parameters.Settings) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The corrected point: the coordinates of xi where oob is True moved into [lb, ub], the others unchanged.

        Coordinates with two finite bounds are corrected by the method; a coordinate with one infinite bound is set to the finite bound it crossed, as SATURATE does, and a fixed coordinate (lb_i = ub_i) to lb_i (#144). NONE and RESAMPLE return xi unchanged.

        Parameters
        ----------
        xi : numpy.ndarray
            The point, shape (d,).
        oob : numpy.ndarray
            The mask (bool) of the coordinates outside the bounds, shape (d,) (see `is_out_of_bounds`).
        sigma : float
            The step size (used by COTN).
        settings : Settings
            The settings with the bounds lb and ub.

        Returns
        -------
        numpy.ndarray
            The corrected point, shape (d,).
        """

    def correct_point(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], sigma: float, settings: modcma.c_maes.cmaescpp.parameters.Settings) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        A point that is not sampled from the search distribution, moved into [lb, ub]: TPA's two points (#98) and the point of gradient injection (#90).

        As `correct_x` for the coordinates outside the bounds, without counting it in `n_out_of_bounds`. RESAMPLE cannot sample such a point again: it sets the coordinates outside on the bound they crossed, as SATURATE does. With NONE the point is returned as it is.

        TPA first moves its two points toward the mean, both by the same factor, until both are in the box, so that they stay symmetric about the mean; this method then changes them only when the mean itself is outside.

        Parameters
        ----------
        xi : numpy.ndarray
            The point, shape (d,).
        sigma : float
            The step size (used by COTN).
        settings : Settings
            The settings with the bounds lb and ub and the method (`modules.bound_correction`).

        Returns
        -------
        numpy.ndarray
            The point inside the box (unless the method is NONE), shape (d,).
        """

    def delta_out_of_bounds(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], oob: Annotated[NDArray[numpy.bool_], dict(shape=(None,), order='C')], settings: modcma.c_maes.cmaescpp.parameters.Settings) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The coordinates of xi where oob is True normalized to the box, (x - lb) / (ub - lb) (in [0, 1] inside it); the others unchanged. Meaningful only for a finite, positive width ub - lb: `correct_x` gives the methods no other coordinates.

        Parameters
        ----------
        xi : numpy.ndarray
            The point, shape (d,).
        oob : numpy.ndarray
            The mask (bool) of the coordinates to normalize, shape (d,).
        settings : Settings
            The settings with the bounds lb and ub.

        Returns
        -------
        numpy.ndarray
            Shape (d,).
        """

    def is_out_of_bounds(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], settings: modcma.c_maes.cmaescpp.parameters.Settings) -> Annotated[NDArray[numpy.bool_], dict(shape=(None,), order='C')]:
        """
        The mask of the coordinates of xi outside [lb, ub].

        Parameters
        ----------
        xi : numpy.ndarray
            The point, shape (d,).
        settings : Settings
            The settings with the bounds lb and ub.

        Returns
        -------
        numpy.ndarray
            A bool array, shape (d,).
        """

class Resample(BoundCorrection):
    """
    Sample again (`CorrectionMethod.RESAMPLE`): an individual outside the bounds is drawn again, up to 5 d times (a mirrored pair as one), and kept as it is after that. Every rejected sample counts in `n_out_of_bounds`.

    Keeps the shape of the distribution inside the box; costs samples, not evaluations. A point still outside after the last try is evaluated, and can be selected, but never becomes `es.p.stats.global_best`, the solution that `run()` and `fmin` return (#168).
    """

    def __init__(self) -> None:
        """Create the correction."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class NoCorrection(BoundCorrection):
    """
    No bound correction (`CorrectionMethod.NONE`, the default): points outside the bounds are evaluated as they are (and counted in `n_out_of_bounds`).
    """

    def __init__(self) -> None:
        """Create the correction."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class COTN(BoundCorrection):
    """
    Complete one-tailed normal correction (`CorrectionMethod.COTN`; Caraffini et al. 2019): a coordinate outside the bounds is moved inside, at a random distance from the violated bound.

    In coordinates normalized to [0, 1], the new value is |b - r| with r = sigma |n|, b the violated bound (0 or 1), sigma the step size and n normal with mean 0 and standard deviation 1/3. Where r > 1, r is drawn again (up to 10 times), then capped at 1, so the new value stays in the box.
    """

    def __init__(self, dim: int) -> None:
        """
        Create the correction. It draws from the global random generator (`modcma.set_seed`); the optimizer creates its own from its generator.

        Parameters
        ----------
        dim : int
            The dimension d.
        """

    @property
    def sampler(self) -> modcma.c_maes.cmaescpp.sampling.Gaussian:
        """The Gaussian sampler of the offsets n (standard deviation 1/3)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Mirror(BoundCorrection):
    """
    Reflect a coordinate outside the bounds back into the box at the violated bound (`CorrectionMethod.MIRROR`); a point more than one box width outside is reflected repeatedly.
    """

    def __init__(self) -> None:
        """Create the correction."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class UniformResample(BoundCorrection):
    """
    Replace each coordinate outside the bounds by a uniform random value in [lb, ub] (`CorrectionMethod.UNIFORM_RESAMPLE`).
    """

    def __init__(self, dim: int) -> None:
        """
        Create the correction. It draws from the global random generator (`modcma.set_seed`); the optimizer creates its own from its generator.

        Parameters
        ----------
        dim : int
            The dimension d.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Saturate(BoundCorrection):
    """
    Set a coordinate outside the bounds to the violated bound, i.e. clip x to [lb, ub] (`CorrectionMethod.SATURATE`).
    """

    def __init__(self) -> None:
        """Create the correction."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Toroidal(BoundCorrection):
    """
    Wrap a coordinate outside the bounds around to the other side of the box (`CorrectionMethod.TOROIDAL`): periodic boundaries.
    """

    def __init__(self) -> None:
        """Create the correction."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """
