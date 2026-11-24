"""
Selection of the evaluated population: pairwise selection, elitism and sorting.

The optimizer holds a `Strategy` as `es.p.selection`; it is built from `Modules.mirrored` and `Modules.elitist`.
"""

from typing import TypeAlias

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.parameters


class Elitism:
    """
    Elitist (mu + lambda) selection: the parents compete with the offspring.

    Selected by `Modules.elitist = True` (always on for a (1+1)-ES); `NoElitism` otherwise. From the second generation on, the previous population (`es.p.old_pop`, the lambda best of the previous selection) is appended to the offspring before sorting, so the best points found so far survive; their y and z are recomputed in the current distribution. Can speed up convergence on unimodal functions, at the cost of diversity.
    """

    def __init__(self) -> None:
        """Create the elitist selection."""

    def __call__(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Append the previous population to the current one (from the second generation on).

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer (`es.p`); its `pop` is changed in place.
        """

class NoElitism(Elitism):
    """
    No elitism, (mu, lambda) selection: only the offspring are selected (the default).
    """

    def __init__(self) -> None:
        """Create the non-elitist selection."""

    def __call__(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Do nothing.

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer (`es.p`), unchanged.
        """

class Pairwise:
    """
    Pairwise selection for mirrored sampling: of each mirrored pair only the better point can be selected.

    Selected by `Modules.mirrored = Mirror.PAIRWISE`; `NoPairwise` otherwise. The worse point of each pair (individuals 2i and 2i + 1) gets the objective value inf, so the two mirrored steps do not cancel each other in the recombination. Needs an even population size (Settings makes lambda even with PAIRWISE).
    """

    def __init__(self) -> None:
        """Create the pairwise selection."""

    def __call__(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Set the objective value of the worse point of every pair to inf.

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer (`es.p`); `pop.f` is changed in place.
        """

class NoPairwise(Pairwise):
    """No pairwise selection (the default unless `Mirror.PAIRWISE`)."""

    def __init__(self) -> None:
        """Create the no-op pairwise selection."""

    def __call__(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Do nothing.

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer (`es.p`), unchanged.
        """

class Strategy:
    """
    The selection of a generation: pairwise selection, elitism, then sorting.

    `select` applies `pairwise` and `elitsm`, sorts the population by objective value, keeps the lambda best and updates `es.p.stats` (the best solution and the mean objective value).
    """

    def __init__(self, modules: modcma.c_maes.cmaescpp.parameters.Modules) -> None:
        """
        Create the selection that the modules ask for.

        Parameters
        ----------
        modules : Modules
            The modules; `mirrored` and `elitist` choose `pairwise` and `elitsm`.
        """

    def select(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Select the evaluated population in place.

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer (`es.p`); `pop` is sorted (best first) and cut to lambda individuals, `stats` is updated.
        """

    @property
    def pairwise(self) -> Pairwise:
        """
        The pairwise selection: `Pairwise` with `Mirror.PAIRWISE`, `NoPairwise` otherwise.
        """

    @pairwise.setter
    def pairwise(self, arg: Pairwise, /) -> None: ...

    @property
    def elitism(self) -> Elitism:
        """The elitism: `Elitism` with `Modules.elitist`, `NoElitism` otherwise."""

    @elitism.setter
    def elitism(self, arg: Elitism, /) -> None: ...

    @property
    def elitsm(self) -> Elitism:
        """The old (misspelled) name of `elitism`."""

    @elitsm.setter
    def elitsm(self, arg: Elitism, /) -> None: ...

Elitsm: TypeAlias = Elitism

NoElitsm: TypeAlias = NoElitism
