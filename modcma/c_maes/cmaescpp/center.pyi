"""
Center placement: where the mean of the search distribution is placed at a restart.

The optimizer holds one as `es.p.center_placement`, selected by `Modules.center_placement` (`options.CenterPlacement`). It is applied at every restart; the first run starts at `Settings.x0`, or at the center of the bounds when x0 is not given.

UNIFORM, NOVELTY_WEIGHTED and MAXIMIN_TABOO sample in the box [lb, ub]. In a coordinate with an infinite bound they sample uniformly in [x0 - 2 sigma0, x0 + 2 sigma0], with x0 `Settings.x0` or `Settings.center`; the interval is shifted to start at a finite lb, or to end at a finite ub, when it would cross it. NOVELTY_WEIGHTED and MAXIMIN_TABOO then divide the distances in that coordinate by the width of the interval, 4 sigma0.
"""

import modcma.c_maes.cmaescpp


class Placement:
    """
    Base class of the center placements: sets the mean `es.p.adaptation.m`.
    """

    def __call__(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Set the mean of the search distribution.

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer (`es.p`); `adaptation.m` is set.
        """

class X0(Placement):
    """
    The mean is `Settings.x0`, `Settings.center` (the center of the box; see it for infinite bounds) when x0 is not given (`CenterPlacement.X0`, the default).
    """

    def __init__(self) -> None:
        """Create the placement."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy: the bytes of the object's C++ serialization (format described in the guide's page on saving and resuming a run), with the attributes of a Python subclass.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated or corrupted.
        """

class Uniform(Placement):
    """
    The mean is a uniform random point in the box [lb, ub] (`CenterPlacement.UNIFORM`); in a coordinate with an infinite bound, in x0 +- 2 sigma0 (see the module).
    """

    def __init__(self) -> None:
        """Create the placement."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy: the bytes of the object's C++ serialization (format described in the guide's page on saving and resuming a run), with the attributes of a Python subclass.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated or corrupted.
        """

class Zero(Placement):
    """The mean is the origin (`CenterPlacement.ZERO`)."""

    def __init__(self) -> None:
        """Create the placement."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy: the bytes of the object's C++ serialization (format described in the guide's page on saving and resuming a run), with the attributes of a Python subclass.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated or corrupted.
        """

class Center(Placement):
    """
    The mean is `Settings.center`: the midpoint (lb + ub) / 2 of a finite coordinate, with the rule for infinite bounds there (`CenterPlacement.CENTER`).
    """

    def __init__(self) -> None:
        """Create the placement."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy: the bytes of the object's C++ serialization (format described in the guide's page on saving and resuming a run), with the attributes of a Python subclass.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated or corrupted.
        """

class MaximinTaboo(Placement):
    """
    The mean is placed as far as possible from the taboo points of repelling (`CenterPlacement.MAXIMIN_TABOO`).

    Of 512 uniform random points in the box, the one whose nearest taboo point of `es.p.repelling.archive` is farthest away (in coordinates divided by the widths of the box); a uniform random point while the archive is empty. For use with repelling restarts. In a coordinate with an infinite bound, the candidates are drawn in x0 +- 2 sigma0 and the distances divided by 4 sigma0 (see the module).
    """

    def __init__(self) -> None:
        """Create the placement."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy: the bytes of the object's C++ serialization (format described in the guide's page on saving and resuming a run), with the attributes of a Python subclass.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated or corrupted.
        """

class NoveltyWeighted(Placement):
    """
    The mean is a random point that favours regions away from the taboo points of repelling (`CenterPlacement.NOVELTY_WEIGHTED`).

    Of 512 uniform random points in the box, one is drawn with probability proportional to the square of the distance to its nearest taboo point (in coordinates divided by the widths of the box); a uniform random point while the archive is empty. For use with repelling restarts; needs finite bounds.
    """

    def __init__(self) -> None:
        """Create the placement."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy: the bytes of the object's C++ serialization (format described in the guide's page on saving and resuming a run), with the attributes of a Python subclass.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated or corrupted.
        """
