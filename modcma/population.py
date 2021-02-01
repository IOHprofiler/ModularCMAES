"""TImplemention for the Population object used in the ModularCMA-ES."""
from typing import Any
import numpy as np


class Population:
    """Object for holding a Population of individuals."""

    def __init__(self, x, y, f, s=None):
        """Reshape x and y."""
        self.x = x
        self.y = y
        self.f = f
        self.s = np.empty(self.f.shape) if s is None else s
        if len(self.x.shape) == 1:
            self.x = self.x.reshape(-1, 1)
            self.y = self.y.reshape(-1, 1)

    def sort(self) -> "Population":
        """Sort the population according to their fitness values."""
        rank = np.argsort(self.f)
        self.x = self.x[:, rank]
        self.y = self.y[:, rank]
        self.f = self.f[rank]
        self.s = self.s[rank]
        return self

    def copy(self) -> "Population":
        """Return a new population object, with it's variables copied.

        Returns
        -------
        Population

        """
        return Population(self.x.copy(), self.y.copy(), self.f.copy(), self.s.copy())

    def __add__(self, other: "Population") -> "Population":
        """Add two population objects with each other.

        Parameters
        ----------
        other: Population
            another population which is to be used to perform the addition

        Returns
        -------
        Population

        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Other should be {self.__class__}" f"got {other.__class__}"
            )
        return Population(
            np.hstack([self.x, other.x]),
            np.hstack([self.y, other.y]),
            np.append(self.f, other.f),
            np.append(self.s, other.s),
        )

    def __getitem__(self, key: Any) -> "Population":
        """Method allowing for indexing the population object as if it were an np.ndarray.

        Parameters
        ----------
        key: int, [int], itertools.slice
            value by with to index the population

        Returns
        -------
        Population

        """
        if isinstance(key, int):
            return Population(
                self.x[:, key].reshape(-1, 1),
                self.y[:, key].reshape(-1, 1),
                np.array([self.f[key]]),
                np.array([self.s[key]]),
            )
        if isinstance(key, slice):
            return Population(
                self.x[:, key.start: key.stop: key.step],
                self.y[:, key.start: key.stop: key.step],
                self.f[key.start: key.stop: key.step],
                self.s[key.start: key.stop: key.step],
            )
        if isinstance(key, list) and all(
            map(lambda x: isinstance(x, int) and x >= 0, key)
        ):
            return Population(self.x[:, key], self.y[:, key], self.f[key], self.s[key])

        raise KeyError(
            "Key must be (list of non-negative) integer(s) or slice, not {}".format(
                type(key)
            )
        )

    @property
    def n(self) -> int:
        """The number of individuals in the population."""
        return len(self.f)

    @property
    def d(self) -> int:
        """The dimension of the individuals in the population."""
        shape_ = list(self.x.shape)
        shape_.remove(self.n)
        return shape_[0]

    def __repr__(self) -> str:
        """Representation of Population object."""
        return f"<Population d: {self.d}, n: {self.n}>"

    def __str__(self) -> str:
        """String representation of Population object."""
        return repr(self)
