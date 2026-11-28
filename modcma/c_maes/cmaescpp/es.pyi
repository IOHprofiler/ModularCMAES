"""
Two simple evolution strategies, as baselines.

OnePlusOneES is a (1+1)-ES with a success rule, MuCommaLambdaES a (mu, lambda)-ES with self-adapted coordinate-wise step sizes. Of the Modules they use only the sampler (sampler, sample_transformation, orthogonal) and bound_correction, with the fixed box [-5, 5]^d. mirrored samples no pairs here (ModularCMAES forms its mirrored pairs in the mutation step, #75); with orthogonal it only changes how the samples are grouped into orthogonal blocks. They draw from the global random generator (set_seed).
"""

from collections.abc import Callable
from typing import Annotated

import numpy
import numpy.typing
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp.bounds
import modcma.c_maes.cmaescpp.parameters
import modcma.c_maes.cmaescpp.sampling


class OnePlusOneES:
    """
    A (1+1)-ES with a success rule for the step size.

    Every step samples x1 = x + sigma z, evaluates it and keeps it when it is better than x (with a bound correction, only when it lies in the box [-5, 5]^d, #168). sigma is multiplied by exp(0.8)^decay after a success and by exp(-0.2)^decay otherwise, with decay = 1 / sqrt(d + 1).
    """

    def __init__(self, d: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], f0: float, sigma0: float = 1.0, budget: int = 10000, target: float = 1e-08, modules: modcma.c_maes.cmaescpp.parameters.Modules = ...) -> None:
        """
        A (1+1)-ES that starts from x0.

        Parameters
        ----------
        d : int
            The dimension.
        x0 : numpy.ndarray
            (d,) the starting point.
        f0 : float
            The objective value of x0 (not evaluated by the ES; it counts as the first evaluation, t = 1).
        sigma0 : float, default 1.0
            The initial step size.
        budget : int, default 10000
            The maximum number of evaluations, including x0.
        target : float, default 1e-8
            Stop when f is at most target.
        modules : Modules, default Modules()
            The sampler and the bound correction (RESAMPLE resamples up to 5 d times).
        """

    def __call__(self, arg: Callable[[numpy.typing.NDArray[numpy.float64]], float], /) -> None:
        """
        Run steps until t reaches budget or f is at most target.

        The argument is the objective function, called with a (d,) array.
        """

    def step(self, arg: Callable[[numpy.typing.NDArray[numpy.float64]], float], /) -> None:
        """
        One step: sample, evaluate, select and adapt sigma (one evaluation).

        The argument is the objective function, called with a (d,) array.
        """

    def sample(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """A candidate x + sigma z, bound corrected; not evaluated."""

    @property
    def d(self) -> int:
        """The dimension."""

    @d.setter
    def d(self, arg: int, /) -> None: ...

    @property
    def sigma(self) -> float:
        """The step size."""

    @sigma.setter
    def sigma(self, arg: float, /) -> None: ...

    @property
    def decay(self) -> float:
        """The exponent of the step-size update, 1 / sqrt(d + 1)."""

    @decay.setter
    def decay(self, arg: float, /) -> None: ...

    @property
    def x(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (d,) array: the current (best) point. With a bound correction a point outside the box [-5, 5]^d (RESAMPLE gives up after 5 d tries) does not replace it (#168).
        """

    @x.setter
    def x(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def f(self) -> float:
        """The objective value of x."""

    @f.setter
    def f(self, arg: float, /) -> None: ...

    @property
    def t(self) -> int:
        """The number of evaluations, including x0."""

    @t.setter
    def t(self, arg: int, /) -> None: ...

    @property
    def budget(self) -> int:
        """The maximum number of evaluations."""

    @budget.setter
    def budget(self, arg: int, /) -> None: ...

    @property
    def target(self) -> float:
        """Stop when f is at most target."""

    @target.setter
    def target(self, arg: float, /) -> None: ...

    @property
    def sampler(self) -> modcma.c_maes.cmaescpp.sampling.Sampler:
        """The sampler of z (sampling.Sampler), built from the modules."""

    @sampler.setter
    def sampler(self, arg: modcma.c_maes.cmaescpp.sampling.Sampler, /) -> None: ...

    @property
    def rejection_sampling(self) -> bool:
        """
        Whether points outside the bounds are resampled (bound_correction RESAMPLE).
        """

    @rejection_sampling.setter
    def rejection_sampling(self, arg: bool, /) -> None: ...

    @property
    def corrector(self) -> modcma.c_maes.cmaescpp.bounds.BoundCorrection:
        """
        The bound correction (bounds.BoundCorrection), by modules.bound_correction.
        """

    @corrector.setter
    def corrector(self, arg: modcma.c_maes.cmaescpp.bounds.BoundCorrection, /) -> None: ...

class MuCommaLambdaES:
    """
    A (mu, lambda)-ES with self-adapted coordinate-wise step sizes.

    lambda = 5 d offspring and mu = lambda // 4 parents. Offspring i gets the step sizes S_i = sigma * exp(tau_i N(0, I)) * exp(tau N(0, 1)) (elementwise) and the point m + S_i * z. The new m and sigma are the means of the mu best points and of their step sizes.
    """

    def __init__(self, d: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], sigma0: float = 1.0, budget: int = 10000, target: float = 1e-08, modules: modcma.c_maes.cmaescpp.parameters.Modules = ...) -> None:
        """
        A (mu, lambda)-ES with its mean at x0.

        Parameters
        ----------
        d : int
            The dimension.
        x0 : numpy.ndarray
            (d,) the initial mean (not evaluated).
        sigma0 : float, default 1.0
            The initial step size of every coordinate.
        budget : int, default 10000
            The maximum number of evaluations: a generation runs only when its lambda evaluations fit.
        target : float, default 1e-8
            Stop when f_min is at most target.
        modules : Modules, default Modules()
            The sampler and the bound correction (RESAMPLE resamples up to 5 d times).
        """

    def __call__(self, arg: Callable[[numpy.typing.NDArray[numpy.float64]], float], /) -> None:
        """
        Run generations while the next one fits the budget (e + lambda <= budget) and f_min is above target.

        The argument is the objective function, called with a (d,) array.
        """

    def step(self, arg: Callable[[numpy.typing.NDArray[numpy.float64]], float], /) -> None:
        """
        One generation: lambda evaluations, selection and recombination.

        The argument is the objective function, called with a (d,) array.
        """

    def sample(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        A point m + si * z for the (d,) step sizes si (the argument), bound corrected; not evaluated.
        """

    @property
    def d(self) -> int:
        """The dimension."""

    @d.setter
    def d(self, arg: int, /) -> None: ...

    @property
    def lamb(self) -> int:
        """The number of offspring, 5 d."""

    @lamb.setter
    def lamb(self, arg: int, /) -> None: ...

    @property
    def mu(self) -> int:
        """The number of parents, lambda // 4."""

    @mu.setter
    def mu(self, arg: int, /) -> None: ...

    @property
    def sigma(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """(d,) array: the step sizes of the mean."""

    @sigma.setter
    def sigma(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def m(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """(d,) array: the mean."""

    @m.setter
    def m(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def X(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        (d, lambda) array: the offspring of the last generation, sorted by f (not set before the first generation).
        """

    @X.setter
    def X(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def S(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        (d, lambda) array: the step sizes of the offspring, in the order of X (not set before the first generation).
        """

    @S.setter
    def S(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def f(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """(lambda,) array: the objective values of the offspring, sorted."""

    @f.setter
    def f(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def tau(self) -> float:
        """The learning rate of the common step-size factor, 1 / sqrt(d)."""

    @tau.setter
    def tau(self, arg: float, /) -> None: ...

    @property
    def tau_i(self) -> float:
        """
        The learning rate of the coordinate-wise step-size factors, 1 / d^0.25.
        """

    @tau_i.setter
    def tau_i(self, arg: float, /) -> None: ...

    @property
    def mu_inv(self) -> float:
        """1 / mu: the recombination weight."""

    @mu_inv.setter
    def mu_inv(self, arg: float, /) -> None: ...

    @property
    def f_min(self) -> float:
        """The best objective value found (+inf at first)."""

    @f_min.setter
    def f_min(self, arg: float, /) -> None: ...

    @property
    def x_min(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        (d,) array: the best point found (NaN before the first generation). With a bound correction only a point in the box [-5, 5]^d (#168).
        """

    @x_min.setter
    def x_min(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def t(self) -> int:
        """The number of generations."""

    @t.setter
    def t(self, arg: int, /) -> None: ...

    @property
    def e(self) -> int:
        """The number of evaluations."""

    @e.setter
    def e(self, arg: int, /) -> None: ...

    @property
    def budget(self) -> int:
        """The maximum number of evaluations."""

    @budget.setter
    def budget(self, arg: int, /) -> None: ...

    @property
    def target(self) -> float:
        """Stop when f_min is at most target."""

    @target.setter
    def target(self, arg: float, /) -> None: ...

    @property
    def sampler(self) -> modcma.c_maes.cmaescpp.sampling.Sampler:
        """The sampler of z (sampling.Sampler), built from the modules."""

    @sampler.setter
    def sampler(self, arg: modcma.c_maes.cmaescpp.sampling.Sampler, /) -> None: ...

    @property
    def sigma_sampler(self) -> modcma.c_maes.cmaescpp.sampling.Sampler:
        """The standard normal sampler of the coordinate-wise step-size factors."""

    @sigma_sampler.setter
    def sigma_sampler(self, arg: modcma.c_maes.cmaescpp.sampling.Sampler, /) -> None: ...

    @property
    def rejection_sampling(self) -> bool:
        """
        Whether points outside the bounds are resampled (bound_correction RESAMPLE).
        """

    @rejection_sampling.setter
    def rejection_sampling(self, arg: bool, /) -> None: ...

    @property
    def corrector(self) -> modcma.c_maes.cmaescpp.bounds.BoundCorrection:
        """
        The bound correction (bounds.BoundCorrection), by modules.bound_correction.
        """

    @corrector.setter
    def corrector(self, arg: modcma.c_maes.cmaescpp.bounds.BoundCorrection, /) -> None: ...
