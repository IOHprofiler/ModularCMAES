from typing import Generator
import numpy as np
from scipy import stats, linalg
from sobol_seq import i4_sobol
from ghalton import Halton


def gaussian_sampling(d: int) -> Generator[np.ndarray, None, None]:
    '''Generator yielding random normal (gaussian) samples.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray
    '''
    while True:
        yield np.random.randn(d, 1)


def sobol_sampling(d: int) -> Generator[np.ndarray, None, None]:
    '''Generator yielding samples from a Sobol sequence

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray
    '''
    seed = np.random.randint(2, max(3, d**2))
    while True:
        sample, seed = i4_sobol(d, max(seed, 2))
        yield stats.norm.ppf(sample).reshape(-1, 1)


def halton_sampling(d: int) -> Generator[np.ndarray, None, None]:
    '''Generator yielding samples from a Halton sequence

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray
    '''
    halton = Halton(d)
    while True:
        yield stats.norm.ppf(halton.get(1)[0]).reshape(-1, 1)


def mirrored_sampling(sampler: Generator) -> Generator[np.ndarray, None, None]:
    '''Generator yielding mirrored samples.
    For every sample from the input sampler (generator), both its
    original and complemented form are yielded.

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray

    Yields
    ------
    numpy.ndarray
    '''
    for sample in sampler:
        yield sample
        yield sample * -1


def orthogonal_sampling(sampler: Generator, n_samples: int) -> Generator[np.ndarray, None, None]:
    '''Generator yielding orthogonal samples.
    This function orthogonalizes <n_samples>, and succesively yields each
    of them. It uses the linalg.orth decomposition function of the scipy library.

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray
    n_samples: int
        An integer indicating the number of sample to be orthogonalized.

    Yields
    ------
    numpy.ndarray
    '''
    samples = []
    for sample in sampler:
        samples.append(sample)
        if len(samples) == max(max(sample.shape), n_samples):
            samples = np.hstack(samples)
            L = np.linalg.norm(samples, axis=0)
            Q, *_ = np.linalg.qr(samples.T)
            samples = [s.reshape(-1, 1) for s in (Q.T * L).T]
            for _ in range(n_samples):
                yield samples.pop()
                
