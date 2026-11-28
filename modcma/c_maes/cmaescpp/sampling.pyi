"""
The samplers that generate the vectors z of the mutations y = A z.

The optimizer builds its sampler from the modules: a base sampler of uniform numbers in (0, 1)^d (`Uniform`, `Sobol` or `Halton`; `options.BaseSampler`), a transformer to the distribution of the mutations (`GaussianTransformer` and the others; `options.SampleTranformerType`), then `Orthogonal` around it with `Modules.orthogonal`, and `CachedSampler` with `constants.cache_samples` (not with SOBOL). With `Settings.fast_sampling`, `FastGaussian` replaces the default UNIFORM + GAUSSIAN. Mirrored sampling (`Modules.mirrored`) is not a sampler of the optimizer: it samples the first of each pair and negates it for the second (`Mirrored` is a standalone version). Calling a sampler returns one sample, a vector of length d. The samplers created here draw from the global generator (`set_seed`); the ones of an optimizer draw from its own.
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
        Prepare for populations of lamb individuals with these modules; called at the start and at every restart. `Orthogonal` adapts its number of samples per generation and `Mirrored` starts a new pair; most samplers do nothing.
        """

    def begin_generation(self) -> None:
        """
        A new generation starts; the optimizer calls it before it samples a population.

        `Orthogonal` starts a new block and `Mirrored` a new pair; most samplers do nothing.
        """

    def expected_length(self) -> float:
        """
        The expected length E||z|| of a sample. The optimizer reads it when it is created (`Weights.expected_length_z`: CSA, h_sigma, gradient injection) and threshold convergence at every generation. sqrt(d) for `Uniform`, `Sobol`, `Halton`, `Gaussian` and `PySampler`; the transformers and `FastGaussian` give their own, and `Orthogonal`, `CachedSampler` and `Mirrored` that of the sampler they wrap.
        """

class PySampler(Sampler):
    """
    A sampler that calls a Python function for every coordinate. It cannot be pickled or copied.
    """

    def __init__(self, d: int, function: Callable[[], float]) -> None:
        """Samples of dimension d, each coordinate a call function() -> float."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: d calls of the function."""

class Gaussian(Sampler):
    """
    Standard normal samples with the Box-Muller transform of mt19937 numbers.

    The optimizer builds its normal samples from a base sampler and a `GaussianTransformer` instead. Its `expected_length` is sqrt(d), not E||N(0, I)||.
    """

    def __init__(self, d: int) -> None:
        """Samples of dimension d."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class FastGaussian(Sampler):
    """
    Standard normal samples from xoshiro256++ and a ziggurat (Settings.fast_sampling).

    Faster than the default UNIFORM + GAUSSIAN sampler, but a different random stream.
    """

    def __init__(self, d: int) -> None:
        """Samples of dimension d, seeded from the global generator (`set_seed`)."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Uniform(Sampler):
    """
    Pseudo-random uniform samples in (0, 1)^d (mt19937); BaseSampler.UNIFORM.
    """

    def __init__(self, d: int) -> None:
        """Samples of dimension d."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Sobol(Sampler):
    """
    Points of the Sobol sequence in (0, 1)^d, in a random order; BaseSampler.SOBOL.

    The points are computed when the sampler is created: n of them, from index 2 of the sequence on (the first two, 0 and (1/2, ..., 1/2), are left out), with n = max(constants.cache_min_samples, the largest power of 2 <= constants.cache_max_doubles / d). They are drawn in a random order that repeats after all n; the order is the same on every platform for the same seed (#116).
    """

    def __init__(self, d: int) -> None:
        """
        Samples of dimension d, at most 1111 (the table of direction numbers): a larger d terminates the process.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: the next point in the random order."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Halton(Sampler):
    """
    Points of the Halton sequence in (0, 1)^d; BaseSampler.HALTON.

    Coordinate j uses the j-th prime as its base; the sequence starts at index 6. The optimizer uses the scrambled sequence.
    """

    def __init__(self, d: int, scrambled: bool = True) -> None:
        """
        Samples of dimension d; with scrambled, the digits are permuted with random permutations, one per digit position and base (drawn once, from the global generator; the same on every platform for the same seed, #116).
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: the next point of the sequence."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Mirrored(Sampler):
    """
    Mirrored samples: a sample of the wrapped sampler, then its negation, alternately; `reset` and `begin_generation` start a new pair.

    A standalone sampler: the optimizer does not use it. With `Modules.mirrored` the optimizer samples the pairs of each generation itself (offspring 2i and 2i + 1 are z and -z, and a rejected pair is sampled again as a pair).
    """

    def __init__(self, sampler: Sampler) -> None:
        """Mirror the samples of sampler."""

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: a new one, or the last negated."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class CachedSampler(Sampler):
    """
    A sampler that cycles through a fixed set of points.

    The optimizer wraps its sampler in one when `constants.cache_samples` is set (off by default), except with SOBOL, which keeps its points itself. A restart does not empty the cache.
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
        """
        The points (with a wrapped sampler: filled as they are drawn), a list of numpy views of them: writing to one changes the sampler's point. The views stay valid while the sampler exists (the points are never reallocated; #176).
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Orthogonal(Sampler):
    """
    Orthogonalized samples (Modules.orthogonal; Wang, Emmerich and Baeck 2019).

    The n_samples samples of a generation are orthogonalized in blocks: the first min(n_samples, d), then blocks of at most d (more than d vectors cannot be orthogonal). A block is drawn from the wrapped sampler when it is needed, orthonormalized as by the Gram-Schmidt process (a Householder QR decomposition with the signs corrected so that diag(R) > 0: the first sample keeps its direction), and each vector gets the length of the sample it came from. `begin_generation` starts a new generation; without it, every n_samples consecutive samples are one. Samples that replace rejected ones count towards the generation. Wang et al. draw the samples beyond the first d at random; here they are orthogonalized in further blocks.
    """

    def __init__(self, sampler: Sampler, n_samples: int) -> None:
        """
        Orthogonalize the samples of sampler, n_samples per generation. The optimizer uses n_samples = lambda, or ceil(lambda / 2) with mirrored sampling.
        """

    def __call__(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample: the next of the block."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class SampleTransformer(Sampler):
    """
    The base class of the transformers: map the uniform numbers in (0, 1) of a base sampler, coordinate by coordinate, to another distribution (options.SampleTranformerType).
    """

    def raw(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """One sample of the wrapped sampler, without the transformation."""

class IdentityTransformer(SampleTransformer):
    """
    No transformation: the uniform numbers of the base sampler as they are (SampleTranformerType.NONE), the raw base sampler, for testing.

    Its expected length is E||u|| of u uniform in (0, 1)^d.
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
        The expected length E||z|| of a transformed sample (see the class), which CSA, h_sigma, threshold convergence and gradient injection use.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class GaussianTransformer(SampleTransformer):
    """
    Standard normal numbers: the inverse normal CDF of each uniform number, or the Box-Muller transform of pairs with `constants.use_box_muller` (SampleTranformerType.GAUSSIAN, the default).

    Its expected length is that of a d-dimensional standard normal vector, in Hansen's approximation sqrt(d) (1 - 1/(4 d) + 1/(21 d^2)) (Hansen 2016; `FastGaussian` uses the same).
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
        The expected length E||z|| of a transformed sample (see the class), which CSA, h_sigma, threshold convergence and gradient injection use.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class UniformScaler(SampleTransformer):
    """
    Uniform numbers in (-sqrt(3), sqrt(3)), unit variance: -sqrt(3) + 2 sqrt(3) u (SampleTranformerType.SCALED_UNIFORM).

    Its expected length is E||z|| of this distribution in d dimensions, computed numerically (it was sqrt(d) before 2.0).
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
        The expected length E||z|| of a transformed sample (see the class), which CSA, h_sigma, threshold convergence and gradient injection use.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class LaplaceTransformer(SampleTransformer):
    """
    Laplace numbers with scale b = 1/sqrt(2), unit variance: b ln(2u) for u < 0.5, -b ln(2 (1 - u)) otherwise (SampleTranformerType.LAPLACE).

    Its expected length is E||z|| of this distribution in d dimensions, computed numerically (it was sqrt(d) before 2.0).
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
        The expected length E||z|| of a transformed sample (see the class), which CSA, h_sigma, threshold convergence and gradient injection use.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class LogisticTransformer(SampleTransformer):
    """
    Logistic numbers with scale s = sqrt(3)/pi, unit variance: s ln(u / (1 - u)) (SampleTranformerType.LOGISTIC).

    Its expected length is E||z|| of this distribution in d dimensions, computed numerically (it was sqrt(d) before 2.0).
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
        The expected length E||z|| of a transformed sample (see the class), which CSA, h_sigma, threshold convergence and gradient injection use.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
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
        The expected length E||z|| of a transformed sample (see the class), which CSA, h_sigma, threshold convergence and gradient injection use.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class DoubleWeibullTransformer(SampleTransformer):
    """
    Double Weibull numbers with shape 2 and scale 1, unit variance: -sqrt(-ln(2u)) for u < 0.5, sqrt(-ln(2 (1 - u))) otherwise (SampleTranformerType.DOUBLE_WEIBULL).

    Its expected length is Gamma(d + 1/2) / Gamma(d), exactly: each z_i^2 is exponential with mean 1, so ||z||^2 is Gamma(d, 1)-distributed (it was sqrt(d) before 2.0).
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
        The expected length E||z|| of a transformed sample (see the class), which CSA, h_sigma, threshold convergence and gradient injection use.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """
