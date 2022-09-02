"""Module implementing various samplers."""
from typing import Generator
from collections.abc import Iterator

import numpy as np
from scipy import stats


class QmcSampler(Iterator):
    """Wrapper around scipy.stats.qmc quasi random samplers."""

    def __init__(self, qmc: stats.qmc.QMCEngine):
        """Intialize qmc wrapper.

        Parameters
        ----------
        qmc :stats.qmc.QMCEngine
            Any of the samplers defined in scipy.stats.qmc module

        """
        self.qmc = qmc

    def __next__(self) -> np.ndarray:
        """Get next sample and advance qmc sampler.

        Returns
        -------
        np.ndarray

        """
        sample = self.qmc.random(1)
        self.qmc.fast_forward(1)
        return stats.norm.ppf(sample.ravel()).reshape(-1, 1)


class Sobol(QmcSampler):
    """Wrapper around scipy.stats.qmc.Sobol sampler."""

    def __init__(self, d: int):
        """Call super init.

        Parameters
        ----------
        d: int
            dimensionality of the generated samples

        """
        super().__init__(stats.qmc.Sobol(d, seed=np.random.randint(1e9)))


class Halton(QmcSampler):
    """Wrapper around scipy.stats.qmc.Halton sampler."""

    def __init__(self, d: int):
        """Call super init.

        Parameters
        ----------
        d: int
            dimensionality of the generated samples

        """
        super().__init__(stats.qmc.Halton(d, seed=np.random.randint(1e9)))


def gaussian_sampling(d: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding random normal (gaussian) samples.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """
    while True:
        yield np.random.randn(d, 1)


def sobol_sampling(sobol: Sobol) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from a Sobol sequence.

    Parameters
    ----------
    sobol: Sobol
        QmcSampler which wraps a scipy.stats.qmc.Sobol sampler        

    Yields
    ------
    numpy.ndarray

    """
    while True:
        yield next(sobol)


def halton_sampling(halton: Halton) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from a Halton sequence.

    Parameters
    ----------
    halton: Halton
        QmcSampler which wraps a scipy.stats.qmc.Halton sampler

    Yields
    ------
    numpy.ndarray

    """
    while True:
        yield next(halton)


def mirrored_sampling(sampler: Generator) -> Generator[np.ndarray, None, None]:
    """Generator yielding mirrored samples.

    For every sample from the input sampler (generator), both its
    original and complemented form are yielded.

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray

    Yields
    ------
    numpy.ndarray

    """
    for sample in sampler:
        yield sample
        yield sample * -1


def orthogonal_sampling(
    sampler: Generator, n_samples: int
) -> Generator[np.ndarray, None, None]:
    """Generator yielding orthogonal samples.

    This function orthogonalizes <n_samples>, and succesively yields each
    of them. It uses the linalg.qr decomposition function of the numpy library.

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray
    n_samples: int
        An integer indicating the number of sample to be orthogonalized.

    Yields
    ------
    numpy.ndarray

    """
    samples = []
    for sample in sampler:
        samples.append(sample)
        if len(samples) == max(max(sample.shape), n_samples):
            samples = np.hstack(samples)
            length = np.linalg.norm(samples, axis=0)
            q, *_ = np.linalg.qr(samples.T)
            samples = [s.reshape(-1, 1) for s in (q.T * length).T][::-1]
            for _ in range(n_samples):
                yield samples.pop()
