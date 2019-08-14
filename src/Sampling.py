import numpy as np


class GaussianSampling:
    def __init__(self, d: int):
        self.d = d

    def next(self) -> np.ndarray:
        return np.random.randn(*(self.d, 1))
