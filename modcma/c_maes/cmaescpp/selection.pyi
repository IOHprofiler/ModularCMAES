"""
Selection of the evaluated population: pairwise selection, elitism and sorting.

The optimizer holds a `Strategy` as `es.p.selection`; it is built from `Modules.mirrored` and `Modules.elitist`.
"""

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.parameters


class Elitism:
    """
    Elitist (mu + lambda) selection: the parents compete with the offspring.

    Selected by `Modules.elitist = True` (always on for a (1+1)-ES); `NoElitism` otherwise. From the second generation on, the mu parents (the mu best of the previous selection, the first columns of `es.p.old_pop`) are appended to the offspring before sorting, so the best points found so far survive. Their y and z are recomputed in the current distribution with the current step size, y = (x - m) / sigma and z = A^-1 y, as for a repaired point (the framework paper, Sec. 4.1). Can speed up convergence on unimodal functions, at the cost of diversity.
    """

    def __init__(self) -> None:
        """Create the elitist selection."""

    def __call__(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Append the mu parents to the current population (from the second generation on).

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
    Pairwise selection for mirrored sampling (Auger, Brockhoff and Hansen 2011): of each mirrored pair only the better point is ranked.

    Selected by `Modules.mirrored = Mirror.PAIRWISE`; `NoPairwise` otherwise. The better point of each pair (offspring 2i and 2i + 1) is ranked with the unpaired points and the parents of elitism; the losers follow them, sorted by f, with their real values (`Population.n_ranked`). So the two mirrored steps do not cancel each other in the recombination, and the losers get no negative weights in the active update. Settings makes lambda even with PAIRWISE. Before 2.0 the losers got f = +inf (#75).
    """

    def __init__(self) -> None:
        """Create the pairwise selection."""

    def sort(self, population: modcma.c_maes.cmaescpp.Population, n_offspring: int) -> None:
        """
        Sort the population: the ranked individuals by f, then the losers by f.

        Sets population.n_ranked.

        Parameters
        ----------
        population : Population
            Changed in place.
        n_offspring : int
            The first n_offspring individuals are the offspring, in their pairs (2i, 2i + 1); an odd last one is unpaired. The individuals after them (the parents of elitism) are ranked.
        """

    def __call__(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        sort(parameters.pop, parameters.pop.n): every individual is an offspring.

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer (`es.p`); `pop` is changed in place.
        """

class NoPairwise(Pairwise):
    """
    No pairwise selection (the default unless `Mirror.PAIRWISE`): `sort` sorts the population by f.
    """

    def __init__(self) -> None:
        """Create the selection without pairs."""

    def sort(self, population: modcma.c_maes.cmaescpp.Population, n_offspring: int) -> None:
        """
        Sort the population by f (`Population.sort`); every individual is ranked.

        Parameters
        ----------
        population : Population
            Changed in place.
        n_offspring : int
            Not used.
        """

    def __call__(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None:
        """
        Sort parameters.pop by f. Before 2.0 this did nothing: the selection sorted afterwards.

        Parameters
        ----------
        parameters : Parameters
            The state of the optimizer (`es.p`); `pop` is changed in place.
        """

class Strategy:
    """
    The selection of a generation: elitism, then sorting (with pairwise selection).

    `select` applies `elitism` (the parents join the offspring), sorts the population with `pairwise` (by objective value; with `Mirror.PAIRWISE` the losers of the mirrored pairs last), keeps the first lambda and updates `es.p.stats` (the best solution and the mean objective value).
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
            The state of the optimizer (`es.p`); `pop` is sorted (best first; with PAIRWISE the losers of the pairs last) and cut to lambda individuals, `stats` is updated.
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
        """Deprecated: the old (misspelled) name of `elitism`."""

    @elitsm.setter
    def elitsm(self, arg: Elitism, /) -> None: ...

def __getattr__(name: str) -> object:
    """
    The deprecated names of 1.x: `Elitsm` and `NoElitsm` (now `Elitism`, `NoElitism`).
    """
