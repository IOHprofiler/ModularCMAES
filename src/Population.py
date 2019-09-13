import itertools
import numpy as np
from Utils import NpArray


class Population:
    x = NpArray('x')
    y = NpArray('y')
    f = NpArray('f')

    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f

    def sort(self):
        fidx = np.argsort(self.f)
        self.x = self.x[:, fidx]
        self.y = self.y[:, fidx]
        self.f = self.f[fidx]

    def copy(self):
        return Population(**self.__dict__)

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        return Population(
            np.hstack([self.x, other.x]),
            np.hstack([self.y, other.y]),
            np.append(self.f, other.f)
        )

    def __getitem__(self, key):
        if isinstance(key, int):
            return Population(
                self.x[:, key].reshape(-1, 1),
                self.y[:, key].reshape(-1, 1),
                np.array([self.f[key]])
            )
        elif isinstance(key, slice):
            return Population(
                self.x[:, key.start: key.stop: key.step].copy(),
                self.y[:, key.start: key.stop: key.step].copy(),
                self.f[key.start: key.stop: key.step].copy()
            )
        elif isinstance(key, list) and all(
                map(lambda x: isinstance(x, int) and x >= 0, key)):
            return Population(
                self.x[:, key].copy(),
                self.y[:, key].copy(),
                self.f[key].copy()
            )
        else:
            raise KeyError("Key must be (list of) non-negative integer(s) or slice, not {}"
                           .format(type(key)))

    def __repr__(self):
        return "d: {}, n: {}".format(*str(self.x.shape))
