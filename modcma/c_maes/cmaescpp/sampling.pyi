"""
The samplers that generate the vectors z of the mutations y = A z.

The optimizer builds its sampler from the modules: a base sampler of uniform numbers in (0, 1)^d (`Uniform`, `Sobol` or `Halton`; `options.BaseSampler`), a transformer to the distribution of the mutations (`GaussianTransformer` and the others; `options.SampleTranformerType`), then optionally `Orthogonal`, `Mirrored` and `CachedSampler` around it. Calling a sampler returns one sample, a vector of length d. The samplers created here draw from the global generator (`set_seed`); the ones of an optimizer draw from its own.
"""

from collections.abc import Callable, Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp.parameters


class Sampler:
    """The base class of the samplers: generates vectors of d random numbers."""

    @property
    def d(self) -> int:
        """The dimension of the samples."""

    def reset(self, modules: modcma.c_maes.cmaescpp.parameters.Modules, lamb: int) -> None:
        """
        Prepare for populations of lamb individuals with these modules; called at the start and at every restart. `Orthogonal` adapts its block size and `Mirrored` starts a new pair; most samplers do nothing.
        """

    def expected_length(self) -> float:
        """
        The expected length of a sample, used to normalize the step-size adaptation; sqrt(d) unless a sampler knows better.
        """

class PySampler(Sampler):
    """A sampler that calls a Python function for every coordinate."""

    def __init__(self, d: int, function: Callable[[], float]) -> None:
        """Samples of dimension d, each coordinate a call function() -> float."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: d calls of the function."""

class Gaussian(Sampler):
    """
    Standard normal samples with the Box-Muller transform of mt19937 numbers.

    The optimizer builds its normal samples from a base sampler and a `GaussianTransformer` instead.
    """

    def __init__(self, d: int) -> None:
        """Samples of dimension d."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample."""

class FastGaussian(Sampler):
    """
    Standard normal samples from xoshiro256++ and a ziggurat (Settings.fast_sampling).

    Faster than the default UNIFORM + GAUSSIAN sampler, but a different random stream.
    """

    def __init__(self, d: int) -> None:
        """Samples of dimension d, seeded from the global generator (`set_seed`)."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample."""

class Uniform(Sampler):
    """
    Pseudo-random uniform samples in (0, 1)^d (mt19937); BaseSampler.UNIFORM.
    """

    def __init__(self, d: int) -> None:
        """Samples of dimension d."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample."""

class Sobol(Sampler):
    """
    Points of the Sobol sequence in (0, 1)^d, in a random order; BaseSampler.SOBOL.

    The points (from the second point of the sequence on) are computed when the sampler is created, as many as `constants.cache_max_doubles` allows, and drawn in a random order that repeats after all of them.
    """

    def __init__(self, d: int) -> None:
        """Samples of dimension d."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: the next point in the random order."""

class Halton(Sampler):
    """
    Points of the Halton sequence in (0, 1)^d; BaseSampler.HALTON.

    Coordinate j uses the j-th prime as its base; the sequence starts at index 6.
    """

    def __init__(self, d: int, scrambled: bool = True) -> None:
        """
        Samples of dimension d; with scrambled, the digits are permuted with random permutations (drawn once, from the global generator).
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: the next point of the sequence."""

class Mirrored(Sampler):
    """
    Mirrored samples (Modules.mirrored): a sample of the wrapped sampler, then its negation, alternately.
    """

    def __init__(self, sampler: Sampler) -> None:
        """Mirror the samples of sampler."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: a new one, or the last negated."""

class CachedSampler(Sampler):
    """
    A sampler that cycles through a fixed set of points (constants.cache_samples).
    """

    @overload
    def __init__(self, sampler: Sampler) -> None:
        """
        Cache the first n_samples samples of sampler and then repeat them; n_samples = max(constants.cache_min_samples, the largest power of 2 <= constants.cache_max_doubles / d).
        """

    @overload
    def __init__(self, cache: Sequence[Sequence[float]], transform_ppf: bool = False) -> None:
        """
        Cycle through the given points (a non-empty list of equally long vectors).

        With transform_ppf, every coordinate is first mapped through the inverse normal CDF, turning uniform numbers in (0, 1) into standard normal ones.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: the next point of the cycle."""

    @property
    def index(self) -> int:
        """
        A counter of the samples: the next sample is point index mod n_samples.
        """

    @property
    def n_samples(self) -> int:
        """The number of points in the cycle."""

    @property
    def cache(self) -> list[Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]]:
        """The points (with a wrapped sampler: filled as they are drawn)."""

class Orthogonal(Sampler):
    """
    Orthogonalized samples (Modules.orthogonal).

    Draws n = max(d, n_samples) samples of the wrapped sampler, orthonormalizes them in groups of (at most) d with a Householder QR decomposition (more than d vectors cannot be orthogonal), gives each the length of the sample it came from, and returns them one by one before drawing the next n (Wang et al. 2014).
    """

    def __init__(self, sampler: Sampler, n_samples: int) -> None:
        """
        Orthogonalize the samples of sampler, max(d, n_samples) at a time. The optimizer uses n_samples = lambda (lambda / 2 with mirroring), minus 2 with TPA.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: the next of the block."""

class SampleTransformer(Sampler):
    """
    The base class of the transformers: map the uniform numbers in (0, 1) of a base sampler, coordinate by coordinate, to another distribution (options.SampleTranformerType).
    """

    def raw(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample of the wrapped sampler, without the transformation."""

class IdentityTransformer(SampleTransformer):
    """
    No transformation: the uniform numbers of the base sampler as they are (SampleTranformerType.NONE).
    """

    def __init__(self, sampler: Sampler) -> None:
        """
        Transform the samples of sampler, which should give uniform numbers in (0, 1).
        """

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The transformation of x, a vector of uniform numbers in (0, 1), as a new vector.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One transformed sample of the sampler."""

    def expected_length(self) -> float:
        """
        The expected length of a sample, used to normalize the step-size adaptation.
        """

class GaussianTransformer(SampleTransformer):
    """
    Standard normal numbers: the inverse normal CDF of each uniform number, or the Box-Muller transform of pairs with `constants.use_box_muller` (SampleTranformerType.GAUSSIAN, the default).

    Its expected length is that of a d-dimensional standard normal vector, sqrt(d) (1 - 1/(4 d) + 1/(21 d^2)).
    """

    def __init__(self, sampler: Sampler) -> None:
        """
        Transform the samples of sampler, which should give uniform numbers in (0, 1).
        """

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The transformation of x, a vector of uniform numbers in (0, 1), as a new vector.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One transformed sample of the sampler."""

    def expected_length(self) -> float:
        """
        The expected length of a sample, used to normalize the step-size adaptation.
        """

class UniformScaler(SampleTransformer):
    """
    Uniform numbers in (-sqrt(3), sqrt(3)), unit variance: -sqrt(3) + 2 sqrt(3) u (SampleTranformerType.SCALED_UNIFORM).
    """

    def __init__(self, sampler: Sampler) -> None:
        """
        Transform the samples of sampler, which should give uniform numbers in (0, 1).
        """

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The transformation of x, a vector of uniform numbers in (0, 1), as a new vector.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One transformed sample of the sampler."""

    def expected_length(self) -> float:
        """
        The expected length of a sample, used to normalize the step-size adaptation.
        """

class LaplaceTransformer(SampleTransformer):
    """
    Laplace numbers with scale b = 1/sqrt(2), unit variance: b ln(2u) for u < 0.5, -b ln(2 (1 - u)) otherwise (SampleTranformerType.LAPLACE).
    """

    def __init__(self, sampler: Sampler) -> None:
        """
        Transform the samples of sampler, which should give uniform numbers in (0, 1).
        """

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The transformation of x, a vector of uniform numbers in (0, 1), as a new vector.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One transformed sample of the sampler."""

    def expected_length(self) -> float:
        """
        The expected length of a sample, used to normalize the step-size adaptation.
        """

class LogisticTransformer(SampleTransformer):
    """
    Logistic numbers with scale s = sqrt(3)/pi, unit variance: s ln(u / (1 - u)) (SampleTranformerType.LOGISTIC).
    """

    def __init__(self, sampler: Sampler) -> None:
        """
        Transform the samples of sampler, which should give uniform numbers in (0, 1).
        """

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The transformation of x, a vector of uniform numbers in (0, 1), as a new vector.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One transformed sample of the sampler."""

    def expected_length(self) -> float:
        """
        The expected length of a sample, used to normalize the step-size adaptation.
        """

class CauchyTransformer(SampleTransformer):
    """
    Standard Cauchy numbers: tan(pi (u - 0.5)) (SampleTranformerType.CAUCHY).

    The length of a Cauchy vector has no expectation; its expected length is an empirical approximation of the median, d (1 + 0.18 tanh(d / 4.07)).
    """

    def __init__(self, sampler: Sampler) -> None:
        """
        Transform the samples of sampler, which should give uniform numbers in (0, 1).
        """

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The transformation of x, a vector of uniform numbers in (0, 1), as a new vector.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One transformed sample of the sampler."""

    def expected_length(self) -> float:
        """
        The expected length of a sample, used to normalize the step-size adaptation.
        """

class DoubleWeibullTransformer(SampleTransformer):
    """
    Double Weibull numbers with shape 2 and scale 1, unit variance: -sqrt(-ln(2u)) for u < 0.5, sqrt(-ln(2 (1 - u))) otherwise (SampleTranformerType.DOUBLE_WEIBULL).
    """

    def __init__(self, sampler: Sampler) -> None:
        """
        Transform the samples of sampler, which should give uniform numbers in (0, 1).
        """

    def transform(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The transformation of x, a vector of uniform numbers in (0, 1), as a new vector.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One transformed sample of the sampler."""

    def expected_length(self) -> float:
        """
        The expected length of a sample, used to normalize the step-size adaptation.
        """
