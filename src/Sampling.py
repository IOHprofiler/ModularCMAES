from typing import Generator
import numpy as np
from scipy.stats import norm
from sobol_seq import i4_sobol
from ghalton import Halton


def gaussian_sampling(d: int) -> Generator[np.ndarray]:
    '''Generator yielding random normal (gaussian) samples.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    sample: numpy.ndarray
    '''
    while True:
        yield np.random.randn(d, 1)


def sobol_sampling(d: int) -> Generator[np.ndarray]:
    '''Generator yielding samples from a Sobol sequence

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    sample: numpy.ndarray
    '''
    seed = np.random.randint(2, d**2)
    while True:
        sample, seed = i4_sobol(d, max(seed, 2))
        yield norm.ppf(sample).reshape(-1, 1)


def halton_sampling(d: int) -> Generator[np.ndarray]:
    '''Generator yielding samples from a Halton sequence

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    sample: numpy.ndarray
    '''
    halton = Halton(d)
    while True:
        yield norm.ppf(halton.get(1)[0]).reshape(-1, 1)


def mirrored_sampling(sampler: Generator) -> Generator[np.ndarray]:
    '''Generator yielding mirrored samples.
    For every sample from the input sampler (generator), both its 
    original and complemented form are yielded.     

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray

    Yields
    ------
    sample: numpy.ndarray
    '''
    for sample in sampler:
        yield sample
        yield sample * -1


def orthogonal_sampling(sampler: Generator, n_samples: int) -> Generator[np.ndarray]:
    '''Generator yielding orthogonal samples.
    This function orthogonalizes <n_samples>, and succesively yields each 
    of them. It uses the QR decomposition function of the numpy library. 

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray
    n_samples: int
        An integer indicating the number of sample to be orthogonalized. 

    Yields
    ------
    sample: numpy.ndarray
    '''
    samples = []
    for sample in sampler:
        d = max(sample.shape)
        samples.append(sample)
        if len(samples) == max(d, n_samples):
            samples = np.hstack(samples)
            L = np.linalg.norm(samples, axis=0)
            Q, _ = np.linalg.qr(samples)
            samples = [s.reshape(-1, 1) for s in (Q * L).T]
            for _ in range(n_samples):
                yield samples.pop()


if __name__ == "__main__":
    s = gaussian_sampling(4)
    o = orthogonal_sampling(s, 3)
    next(o)
    next(o)
    next(o)
    next(o)
