"""
Center placement: where the mean of the search distribution is placed at a restart.

The optimizer holds one as `es.p.center_placement`, selected by `Modules.center_placement` (`options.CenterPlacement`). It is applied at every restart; the first run starts at `Settings.x0`, or at the center of the bounds when x0 is not given.
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
    The mean is `Settings.x0`, or the center of the box [lb, ub] (the origin without finite bounds) when x0 is not given (`CenterPlacement.X0`, the default).
    """

    def __init__(self) -> None:
        """Create the placement."""

class Uniform(Placement):
    """
    The mean is a uniform random point in the box [lb, ub] (`CenterPlacement.UNIFORM`); needs finite bounds.
    """

    def __init__(self) -> None:
        """Create the placement."""

class Zero(Placement):
    """The mean is the origin (`CenterPlacement.ZERO`)."""

    def __init__(self) -> None:
        """Create the placement."""

class Center(Placement):
    """
    The mean is the center of the box, (lb + ub) / 2, or the origin without finite bounds (`CenterPlacement.CENTER`).
    """

    def __init__(self) -> None:
        """Create the placement."""

class MaximinTaboo(Placement):
    """
    The mean is placed as far as possible from the taboo points of repelling (`CenterPlacement.MAXIMIN_TABOO`).

    Of 512 uniform random points in the box, the one whose nearest taboo point of `es.p.repelling.archive` is farthest away (in coordinates divided by the widths of the box); a uniform random point while the archive is empty. For use with repelling restarts; needs finite bounds.
    """

    def __init__(self) -> None:
        """Create the placement."""

class NoveltyWeighted(Placement):
    """
    The mean is a random point that favours regions away from the taboo points of repelling (`CenterPlacement.NOVELTY_WEIGHTED`).

    Of 512 uniform random points in the box, one is drawn with probability proportional to the square of the distance to its nearest taboo point (in coordinates divided by the widths of the box); a uniform random point while the archive is empty. For use with repelling restarts; needs finite bounds.
    """

    def __init__(self) -> None:
        """Create the placement."""
