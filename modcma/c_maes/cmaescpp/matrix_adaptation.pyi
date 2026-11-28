"""
Matrix adaptations: the mean and the shape of the search distribution.

The optimizer holds one as `es.p.adaptation`, selected by `Modules.matrix_adaptation` (`options.MatrixAdaptationType`). A sample is x = m + sigma * y with y = A z, z drawn by the sampler, and A the sampling matrix of the adaptation (C = A A^T). They differ in what they learn and in their cost per generation: O(d^3) with a decomposition (COVARIANCE, COVARIANCE_NO_EIGV, CMSA), O(d^2 lambda) for MA-ES, Cholesky-CMA-ES and xNES, O(d lambda) for SEPARABLE and NONE (see docs/performance.md).
"""

from typing import Annotated, Final

import numpy
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.parameters


class Metric:
    """
    The metric of a search distribution: its shape without the step size, from `Adaptation.metric()`.

    The samples are distributed as x = m + sigma * B (d * z), z ~ N(0, I), whatever factor of C the adaptation samples with, so they have the covariance sigma^2 C with C = B diag(d)^2 B^T: the one-sigma ellipsoid around m has the semi-axes sigma * d_i along the columns B_i.

    It describes the distribution that is sampled. `CovarianceAdaptation` and `CovarianceNoEigvAdaptation` update C every generation but decompose it only every `es.p.weights.lazy_update_interval` generations, and sample with the last decomposition, so the metric can be that many generations behind their `C` (with the default settings they decompose every generation up to d = 82). `coordinate_wise_variances`, the diagonal of the latest C, can be ahead of it by as much.

    Strictly, the metric (the matrix of the Mahalanobis distance) is C^-1; this holds the decomposition, from which C and C^-1 follow. A copy: it does not change with the optimizer.
    """

    @property
    def d(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The axis lengths, the square roots of the eigenvalues of C (pycma's D), shape (d,), in the order of the columns of `B`: ascending, the order of the eigensolver. For the diagonal adaptations (SEPARABLE, NONE) in the order of the coordinates.
        """

    @property
    def B(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')] | None:
        """
        The eigenvectors of C as columns, shape (d, d), in the order of `d`; None when C is diagonal (SEPARABLE, NONE), where B is the identity.
        """

    @property
    def is_diagonal(self) -> bool:
        """Whether C is diagonal (B is None)."""

    def eigenvalues(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The eigenvalues of C, d^2, in the order of `d`.

        Returns
        -------
        numpy.ndarray
            Shape (d,).
        """

    def inv_C(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        C^-1 = B diag(d)^-2 B^T, computed on call (O(d^3); a diagonal matrix when C is diagonal, where 1 / d^2 is the cheap form).

        On a convex-quadratic function with Hessian H, CMA-ES adapts C towards H^-1 up to a constant factor: Hansen (2016), The CMA Evolution Strategy: A Tutorial, Section 0.4, "we assume that the optimal covariance matrix equals to the inverse Hessian matrix, up to a constant factor" (an assumption with strong empirical support; "a rigorous proof is missing"). pycma uses the same relation (`inverse_hessian_scalar_correction` fits the factor). So C^-1 estimates the shape of the Hessian, not its scale; sigma does not give the scale, it follows the distance to the optimum. The relation needs a full matrix: SEPARABLE can only match a diagonal Hessian, and NONE keeps C = I.

        Returns
        -------
        numpy.ndarray
            Shape (d, d).
        """

    def inv_root_C(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        C^-1/2 = B diag(d)^-1 B^T, computed on call (O(d^3); a diagonal matrix when C is diagonal).

        Returns
        -------
        numpy.ndarray
            Shape (d, d).
        """

    def __repr__(self) -> str:
        """A summary of the state (mean, paths and matrices)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class Adaptation:
    """
    Base class of the matrix adaptations: the mean, the evolution path of the step size and the sampling transformation y = A z.

    Each generation, after selection, `adapt_evolution_paths` updates dm, dz and the paths, the step-size adaptation (`es.p.mutation`) updates sigma, and `adapt_matrix` updates the matrix. `compute_y` and `invert_y` apply A and A^-1.

    The matrices start at the identity. With `Settings.C0` the optimizer starts every run from a factor F of C0 instead, F F^T = C0 (#89): the eigendecomposition for COVARIANCE, the lower Cholesky factor for COVARIANCE_NO_EIGV, CHOLESKY and CMSA, the symmetric square root C0^1/2 for MATRIX and NATURAL_GRADIENT, and the square roots of the diagonal of C0 for SEPARABLE; NONE rejects C0.
    """

    @property
    def m(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """The mean (center) of the search distribution, shape (d,)."""

    @m.setter
    def m(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def m_old(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """The mean of the previous generation, shape (d,)."""

    @m_old.setter
    def m_old(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def dm(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The last step of the mean in units of the step size, (m - m_old) / s, shape (d,), with s per coordinate the step size recombined from the mu selected individuals, sum_i w_i S_i (Hansen 2016, Eq. (24)): sigma when they all have the step size sigma; sigma raised by the lower bound of an integer variable (then dm = sum_i w_i y_i, as Marty et al. 2024); the recombined sampled step size with sample_sigma, SA and LPXNES.
        """

    @dm.setter
    def dm(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def ps(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The evolution path of the step size (conjugate evolution path), shape (d,): ps = (1 - cs_path) ps + sqrt(cs_path (2 - cs_path) mueff_mean) dz, with the rates of `es.p.weights` (C^-1/2 dm instead of dz in `CovarianceAdaptation`, not in its subclass `CovarianceNoEigvAdaptation`). Read by CSA, by h_sigma (`hs`) and by the rank-one term of MATRIX. Starts at zeros, also after a restart.
        """

    @ps.setter
    def ps(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def dz(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The weighted mean of the z of the mu selected individuals, sum_i w_i z_i (of every individual, with the utilities of UTILITY), shape (d,); empty before the first generation.
        """

    @dz.setter
    def dz(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def dd(self) -> float:
        """The dimension d, as a float."""

    @dd.setter
    def dd(self, arg: float, /) -> None: ...

    @property
    def coordinate_wise_variances(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The variance of each coordinate under the sampling transformation, the diagonal of C = A A^T (without sigma^2), shape (d,). Used for the lower bound on the step size of integer variables, and by the restart criteria TolX and NoEffectCoord (and MaxSigma, MaxDSigma and MinDSigma where `keeps_metric()` is False, #80); a NaN or inf in it fails the matrix update.
        """

    @coordinate_wise_variances.setter
    def coordinate_wise_variances(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    def metric(self) -> Metric:
        """
        The metric of the sampled distribution: its axis lengths d and eigenvectors B, without sigma (see `Metric`).

        `CovarianceAdaptation` (not its subclass `CovarianceNoEigvAdaptation`), `SeparableAdaptation` and `NoAdaptation` keep it: it costs nothing. The others decompose C = F F^T of their sampling factor F (M, or A; CMSA its C) when asked, O(d^3), and reuse the result while `factor_version` is unchanged; nothing is computed unless something asks. A value written from Python to their matrices is seen after the next change of the factor (an update or a restart). NaN when the factor is not finite or the eigensolver fails.

        Returns
        -------
        Metric
            A copy.
        """

    def keeps_metric(self) -> bool:
        """
        Whether `metric()` costs nothing: True for `CovarianceAdaptation` (not COVARIANCE_NO_EIGV), `SeparableAdaptation` and `NoAdaptation`, which keep it up to date; False for the others, which decompose their matrix on request, O(d^3).

        By default the restart criteria read the metric only where this is True (#80). ConditionC and NoEffectAxis are created only then. MaxSigma, MaxDSigma and MinDSigma take the longest and the shortest axis from the metric then, and otherwise from the diagonal of C: the square roots of the largest and the smallest entry of `coordinate_wise_variances`.
        """

    @property
    def factor_version(self) -> int:
        """
        A counter increased wherever the sampling factor changes (an update of the matrix, a decomposition, a restart, the start from `Settings.C0`): `metric()` recomputes only when it changed. A write to the matrices from Python does not increase it.
        """

    def adapt_evolution_paths(self, pop: modcma.c_maes.cmaescpp.Population, weights: modcma.c_maes.cmaescpp.parameters.Weights, stats: modcma.c_maes.cmaescpp.parameters.Stats, settings: modcma.c_maes.cmaescpp.parameters.Settings, mu: int) -> None:
        """
        Update dm, dz and the evolution paths after selection.

        Computes dm = (m - m_old) / sum_i w_i S_i (per coordinate, over the mu best) and dz = sum_i w_i z_i over the mu best (every point, with the utilities of `RecombinationWeights.UTILITY`), then the update of the subclass, `adapt_evolution_paths_inner`. Called by `es.adapt()` before the step-size and matrix updates.

        Parameters
        ----------
        pop : Population
            The selected population (sorted, best first).
        weights : Weights
            The recombination weights and learning rates (`es.p.weights`).
        stats : Stats
            The statistics of the run (`es.p.stats`).
        settings : Settings
            The settings (`es.p.settings`).
        mu : int
            The number of selected individuals.
        """

    def adapt_evolution_paths_inner(self, pop: modcma.c_maes.cmaescpp.Population, weights: modcma.c_maes.cmaescpp.parameters.Weights, stats: modcma.c_maes.cmaescpp.parameters.Stats, settings: modcma.c_maes.cmaescpp.parameters.Settings) -> None:
        """
        The update of the evolution paths of the subclass (ps, and h_sigma (`hs`) and pc where there are), without recomputing dm and dz; the second part of `adapt_evolution_paths`. `NaturalGradientAdaptation` computes its natural gradient here (`compute_gradients`).

        Parameters
        ----------
        pop : Population
            The selected population (sorted, best first).
        weights : Weights
            The recombination weights and learning rates (`es.p.weights`).
        stats : Stats
            The statistics of the run (`es.p.stats`).
        settings : Settings
            The settings (`es.p.settings`).
        """

    def adapt_matrix(self, weights: modcma.c_maes.cmaescpp.parameters.Weights, population: modcma.c_maes.cmaescpp.Population, mu: int, settings: modcma.c_maes.cmaescpp.parameters.Settings, stats: modcma.c_maes.cmaescpp.parameters.Stats) -> bool:
        """
        Update the matrix of the search distribution from the selected population.

        In a (1+1)-ES (lambda = 1) the matrix is only updated after an improvement. A failed update (a failed decomposition, or a NaN or inf in the matrix) returns False; the optimizer stores it in `es.p.successful_adaptation`, which the restart criterion `UnableToAdapt` turns into a restart.

        Parameters
        ----------
        weights : Weights
            The recombination weights and learning rates (`es.p.weights`).
        population : Population
            The selected population (sorted, best first).
        mu : int
            The number of selected individuals.
        settings : Settings
            The settings (`modules.active`: negative weights for the worst individuals).
        stats : Stats
            The statistics; `last_update` and `n_updates` are updated.

        Returns
        -------
        bool
            Whether the update succeeded.
        """

    def restart(self, settings: modcma.c_maes.cmaescpp.parameters.Settings) -> None:
        """
        Reset the distribution for a restart: m = `settings.x0` (`settings.center` when it is not given), the paths to zero and the matrices to the identity. The optimizer then starts the matrix from `Settings.C0` when it is given (every run starts from it, #89), and the center placement sets m.

        Parameters
        ----------
        settings : Settings
            The settings.
        """

    def distance(self, u: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], v: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> float:
        """
        The Mahalanobis distance ||A^-1 (u - v)|| in the current search distribution (without sigma).

        Parameters
        ----------
        u, v : numpy.ndarray
            The points, shape (d,).

        Returns
        -------
        float
        """

    def distance_from_center(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> float:
        """
        The distance of x from the mean, `distance(m, x)`.

        Parameters
        ----------
        x : numpy.ndarray
            The point, shape (d,).

        Returns
        -------
        float
        """

    def compute_y(self, zi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The step y = A z of a sample z of the sampler.

        Parameters
        ----------
        zi : numpy.ndarray
            The sample z, shape (d,).

        Returns
        -------
        numpy.ndarray
            y, shape (d,).
        """

    def invert_x(self, xi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], sigma: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The step y = (x - m) / sigma of a point x (elementwise).

        Parameters
        ----------
        xi : numpy.ndarray
            The point x, shape (d,).
        sigma : numpy.ndarray
            The step sizes per coordinate, shape (d,) (a column of `pop.S`).

        Returns
        -------
        numpy.ndarray
            y, shape (d,).
        """

    def invert_y(self, yi: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The sample z = A^-1 y of a step y, the inverse of `compute_y`.

        `MatrixAdaptation` and `NaturalGradientAdaptation` first compute their inverse from the matrix when it is out of date (O(d^3), see `MatrixAdaptation.M_inv`).

        Parameters
        ----------
        yi : numpy.ndarray
            The step y, shape (d,).

        Returns
        -------
        numpy.ndarray
            z, shape (d,).
        """

    def __repr__(self) -> str:
        """A summary of the state (mean, paths and matrices)."""

    def adapt_evolution_paths_innner(self, pop: modcma.c_maes.cmaescpp.Population, weights: modcma.c_maes.cmaescpp.parameters.Weights, stats: modcma.c_maes.cmaescpp.parameters.Stats, settings: modcma.c_maes.cmaescpp.parameters.Settings, mu: int, lamb: int) -> None:
        """
        Deprecated: the old (misspelled) name of `adapt_evolution_paths_inner`; mu and lamb are accepted, as in 1.x, and not used.
        """

class NoAdaptation(Adaptation):
    """
    No matrix adaptation: only the mean and the step size are adapted, A = I (`MatrixAdaptationType.NONE`).

    The cheapest option, O(d lambda) per generation; for well-conditioned (sphere-like) functions, or as a baseline. Settings switches `Modules.active` off with it, and rejects `Settings.C0`.
    """

    def __init__(self, dimension: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Create the adaptation with mean x0 and the initial matrix (the identity).

        Parameters
        ----------
        dimension : int
            The dimension d.
        x0 : numpy.ndarray
            The initial mean m, shape (d,).
        """

    def __repr__(self) -> str:
        """A summary of the state (mean, paths and matrices)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class CovarianceAdaptation(Adaptation):
    """
    The covariance matrix adaptation of CMA-ES (`MatrixAdaptationType.COVARIANCE`, the default).

    C receives a rank-one update from the evolution path pc and a rank-mu update from the selected steps (with `Modules.active`, negative weights for the worst steps: with elitism the worst points of the (mu + lambda) pool, #161), and is decomposed as C = B diag(d)^2 B^T; samples are y = B diag(d) z. C is updated every generation; the eigendecomposition only every `es.p.weights.lazy_update_interval` = 1 / (10 d (c1 + cmu)) generations (with the default settings every generation up to d = 82). The eigendecomposition costs O(d^3) (LAPACK from d = `constants.lapack_min_dim`), the rank updates O(d^2 lambda). Settings replaces it by COVARIANCE_NO_EIGV when the step-size adaptation is not CSA, also for lambda0 = 1, which runs SR (unless `always_compute_eigv`). Hansen & Ostermeier (2001); Hansen (2016), The CMA Evolution Strategy: A Tutorial.
    """

    def __init__(self, dimension: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Create the adaptation with mean x0 and the initial matrix (the identity).

        Parameters
        ----------
        dimension : int
            The dimension d.
        x0 : numpy.ndarray
            The initial mean m, shape (d,).
        """

    @property
    def pc(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The evolution path of the rank-one update of C, shape (d,) (zeros at the start).
        """

    @pc.setter
    def pc(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def d(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The square roots of the eigenvalues of C at the last decomposition (the standard deviations along the principal axes), shape (d,), in the order of the columns of B (ones at the start without `Settings.C0`). With B, the `metric()`. In `CovarianceNoEigvAdaptation` filled only by `metric()`.
        """

    @d.setter
    def d(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def B(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The eigenvectors of C at the last decomposition as columns, shape (d, d) (the identity at the start without `Settings.C0`). In `CovarianceNoEigvAdaptation` filled only by `metric()`.
        """

    @B.setter
    def B(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def C(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The covariance matrix (without sigma^2), symmetric, shape (d, d) (the identity at the start, or `Settings.C0`).
        """

    @C.setter
    def C(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def A(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The sampling matrix, y = A z, shape (d, d): B diag(d) (the lower triangular Cholesky factor of C in `CovarianceNoEigvAdaptation`).
        """

    @A.setter
    def A(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def hs(self) -> bool:
        """
        The indicator h_sigma (Hansen 2016, App. A, symbols of Fig. 6): False when ||ps|| / sqrt(1 - (1 - cs_path)^(2 (g + 1))), with g the generation of the run, reaches `es.p.weights.expected_length_ps` = (1.4 + 2 / (d + 1)) E||z||: the step size is far too small. False stalls the update of pc, and the decay of C compensates the variance that pc loses (see `adapt_covariance_matrix`); True otherwise, and always in the (1+1)-ES. With `StepSizeAdaptation.XNES` and `Modules.active` (not with UTILITY weights), False also drops the negative weights of the generation's update (#175).
        """

    @hs.setter
    def hs(self, arg: bool, /) -> None: ...

    def adapt_covariance_matrix(self, weights: modcma.c_maes.cmaescpp.parameters.Weights, modules: modcma.c_maes.cmaescpp.parameters.Modules, population: modcma.c_maes.cmaescpp.Population, mu: int) -> None:
        """
        Update C with the rank-one (pc) and rank-mu (selected steps) updates: C <- (1 + c1 delta(h_sigma) - c1 - cmu sum(w)) C + c1 pc pc^T + cmu sum_i v_i y_i y_i^T, with delta(h_sigma) = (1 - h_sigma) cc (2 - cc), sum(w) over the weights w_i, and v_i the weights after the rescaling below (Hansen 2016, Eqs. (46), (47)).

        With `modules.active` the whole population is used, the negative weights of the worst steps rescaled by d / ||C^-1/2 y_i||^2 (Hansen 2016, Eq. (46)); otherwise the mu best. With elitism (`population.n_pool`) the negative weights go to the points that the selection dropped, the worst of the (mu + lambda) pool (#161). With `StepSizeAdaptation.XNES` (not with UTILITY weights) a generation with h_sigma = 0 has no negative weights (#175). Does not decompose C.

        Parameters
        ----------
        weights : Weights
            The recombination weights and learning rates (`es.p.weights`).
        modules : Modules
            The modules.
        population : Population
            The selected population (sorted, best first).
        mu : int
            The number of selected individuals.
        """

    def perform_eigendecomposition(self, stats: modcma.c_maes.cmaescpp.parameters.Settings) -> bool:
        """
        Decompose C = B diag(d)^2 B^T and set A = B diag(d), O(d^3).

        Uses Eigen or LAPACK (dsyevd, dsyevr) as `constants.eigen_solver` says. In `CovarianceNoEigvAdaptation`: the Cholesky factorization C = A A^T instead. Increases `factor_version`.

        Parameters
        ----------
        stats : Settings
            The settings; only `verbose` is used (the argument keeps its name of 1.x).

        Returns
        -------
        bool
            False when the decomposition fails or gives an eigenvalue that is not positive or not finite (in `CovarianceNoEigvAdaptation`: when the factorization fails).
        """

    def adapt_ps(self, weights: modcma.c_maes.cmaescpp.parameters.Weights) -> None:
        """
        Update ps = (1 - cs_path) ps + sqrt(cs_path (2 - cs_path) mueff_mean) C^-1/2 dm, with C^-1/2 = B diag(d)^-1 B^T of the last decomposition (dz instead of C^-1/2 dm in `CovarianceNoEigvAdaptation`).

        Parameters
        ----------
        weights : Weights
            The recombination weights and learning rates (`es.p.weights`).
        """

    def __repr__(self) -> str:
        """A summary of the state (mean, paths and matrices)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class SeparableAdaptation(Adaptation):
    """
    Separable CMA-ES: a diagonal covariance matrix (`MatrixAdaptationType.SEPARABLE`).

    Only the d variances are learned, with the rank-one and rank-mu updates of CMA-ES (Hansen 2016, Eq. (47)) applied per coordinate, with pycma's separable learning rates (c1_sep, cmu_sep: see `Weights.c1`, `Weights.cmu`); samples are y = d * z. With `Modules.active` the negative weights are in the sum of the decay, and are multiplied by d / sum_j (y_ij / d_j)^2, d over the squared Mahalanobis length of the step in the diagonal C, as in pycma (GaussDiagonalSampler.update; #73). O(d lambda) per generation: for separable functions, and for high dimension, where a full matrix is slow to learn. Ros & Hansen (2008).
    """

    def __init__(self, dimension: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Create the adaptation with mean x0 and the initial matrix (the identity).

        Parameters
        ----------
        dimension : int
            The dimension d.
        x0 : numpy.ndarray
            The initial mean m, shape (d,).
        """

    @property
    def c(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The diagonal of C, the variance of each coordinate (without sigma^2), shape (d,), at least 1e-12 after an update (ones at the start, or the diagonal of `Settings.C0`).
        """

    @c.setter
    def c(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def pc(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The evolution path of the rank-one update, shape (d,) (zeros at the start).
        """

    @pc.setter
    def pc(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def d(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The standard deviations sqrt(c), shape (d,): y = d * z (ones at the start without `Settings.C0`).
        """

    @d.setter
    def d(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def hs(self) -> bool:
        """
        The indicator h_sigma: False when ||ps|| is long (see `CovarianceAdaptation.hs`; Hansen 2016, App. A, symbols of Fig. 6), which stalls the update of pc, and the decay of c compensates the variance that pc loses; True otherwise. With `StepSizeAdaptation.XNES` and `Modules.active` (not with UTILITY weights), False also drops the negative weights of the generation's update (#175).
        """

    @hs.setter
    def hs(self, arg: bool, /) -> None: ...

    def __repr__(self) -> str:
        """A summary of the state (mean, paths and matrices)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class MatrixAdaptation(Adaptation):
    """
    MA-ES, the matrix adaptation evolution strategy (`MatrixAdaptationType.MATRIX`).

    Learns the sampling matrix M directly, y = M z, without a covariance matrix, an evolution path pc or a decomposition: M = (1 - c1/2 - cmu/2 sum(w)) M + (c1/2) (M ps) ps^T + (cmu/2) sum_i w_i y_i z_i^T, with sum(w) = 1 for the positive weights. O(d^2 lambda) per generation (BLAS from d = `constants.blas_min_dim`), so it suits high dimension. Beyer & Sendhoff (2017).

    MA-ES has no active update. With `Modules.active` this is the active update of the CMA-ES (Hansen 2016, Eq. (47)) written for M to first order, as MA-ES derives its own update: sum(w) includes the negative weights, and the negative weights are multiplied by d / ||z_i||^2. With elitism the negative weights go to the worst points of the (mu + lambda) pool, as in the other adaptations (#161).

    With a step-size adaptation other than CSA, h_sigma (`hs`) drops the rank-one term, and its c1/2 in the decay, in a generation in which ||ps|| is long, as COVARIANCE stalls pc (Hansen 2016, App. A, symbols of Fig. 6; #160). MA-ES has no h_sigma: with CSA, which updates sigma from the same path, the update is MA-ES's.
    """

    def __init__(self, dimension: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Create the adaptation with mean x0 and the initial matrix (the identity).

        Parameters
        ----------
        dimension : int
            The dimension d.
        x0 : numpy.ndarray
            The initial mean m, shape (d,).
        """

    @property
    def M(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The sampling matrix, y = M z, shape (d, d) (the identity at the start without `Settings.C0`).
        """

    @M.setter
    def M(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def M_inv(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The inverse of M, shape (d, d).

        With elitism or TPA_INPLACE (lambda > 1) it is updated with M every generation: M <- M F with F = a I + U D U^T, where a is the decay, U = [ps, the z of the update] and D = diag(c1/2, cmu/2 w) (0 for c1/2 when `hs` is False), so M_inv <- F^-1 M_inv by the Woodbury identity, O(k d^2) for k terms; up to d = `inverse_recompute_max_dim` (30) it is computed exactly from M after every `inverse_recompute_interval` (400) generations of a run, which bounds the rounding error that the update accumulates with the active update at small d (the maintainer's choice, #74). When the update cannot be applied (`inverse_fallbacks`), and without elitism and TPA_INPLACE, it is computed only when `invert_y` needs it (points moved by the bound correction, injected points, the distances of repelling; an LU inverse from d = `constants.blas_min_dim`, else the pseudo-inverse), so it can lag behind M. Before 2.0 the update with elitism was not the inverse of M F, and drifted from the inverse of M.
        """

    @M_inv.setter
    def M_inv(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def hs(self) -> bool:
        """
        The indicator h_sigma: False when ||ps|| is long (Hansen 2016, App. A, symbols of Fig. 6), which skips the rank-one term of the generation; always True with CSA and in the (1+1)-ES (#160). With `StepSizeAdaptation.XNES` and `Modules.active` (not with UTILITY weights), False also drops the negative weights of the generation's update (#175).
        """

    @hs.setter
    def hs(self, arg: bool, /) -> None: ...

    @property
    def inverse_fallbacks(self) -> int:
        """
        The generations with elitism or TPA_INPLACE in which the Woodbury update of `M_inv` could not be applied, since the adaptation was created (read-only; restarts do not reset it): the decay a = 1 - c1/2 - cmu/2 sum(w) was not positive, or the k x k system (F itself when k >= d) was singular or badly conditioned (the ratio of its smallest to its largest pivot below 1e-8). M_inv is then computed when `invert_y` needs it, and the update continues from it.
        """

    inverse_recompute_interval: Final[int] = ...
    """
    With elitism or TPA_INPLACE and d <= `inverse_recompute_max_dim`, M_inv is computed exactly from M after every this many generations of a run, and updated by the Woodbury identity in between (400, the maintainer's choice on #74). Measured on the rotated ellipsoid (condition 1e6, 5 seeds, #159): without it, the largest entry of |M M_inv - I| over 5000 generations reached 2.4 to 26 with the active update at d = 10 and 1.3e-8 to 6.4e-8 at d = 20; with it, at most 1.4e-9 at d = 10 and 1.5e-11 at d = 20.
    """

    inverse_recompute_max_dim: Final[int] = ...
    """
    The largest dimension with the exact recomputation of M_inv every `inverse_recompute_interval` generations (30, the maintainer's choice on #74): the smallest d at which the drift without it stayed below 1e-10 over 5000 generations in every run measured (at most 4.3e-11 at d = 30, below 1e-12 from d = 40 on, with the active update; the runs of `inverse_recompute_interval`).
    """

    def __repr__(self) -> str:
        """A summary of the state (mean, paths and matrices)."""

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class CholeskyAdaptation(Adaptation):
    """
    Cholesky-CMA-ES: the covariance matrix kept as its triangular Cholesky factor (`MatrixAdaptationType.CHOLESKY`).

    The rank-one (pc) and rank-mu updates of C = A A^T are applied to A as rank-one Cholesky updates, one per selected individual, so there is no decomposition: O(d^2 lambda) per generation and O(d^2) memory. Krause et al. (2016).

    With `Modules.active` the negative weights, multiplied by d / ||A^-1 y_i||^2 as in COVARIANCE, are rank-one downdates, and A is scaled by sqrt(1 - c1 - cmu sum(w)) with the negative weights in the sum. A downdate whose result would not be positive definite is skipped (`skipped_downdates`). With elitism the negative weights go to the worst points of the (mu + lambda) pool (#161).

    With a step-size adaptation other than CSA, h_sigma (`hs`) stalls the update of pc and A is scaled by sqrt(1 + c1 delta(h_sigma) - c1 - cmu sum(w)), as in COVARIANCE (Hansen 2016, App. A, symbols of Fig. 6; #160).
    """

    def __init__(self, dimension: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Create the adaptation with mean x0 and the initial matrix (the identity).

        Parameters
        ----------
        dimension : int
            The dimension d.
        x0 : numpy.ndarray
            The initial mean m, shape (d,).
        """

    @property
    def A(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The lower triangular Cholesky factor of C, y = A z, shape (d, d) (the identity at the start without `Settings.C0`).
        """

    @A.setter
    def A(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def pc(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The evolution path of the rank-one update, shape (d,) (zeros at the start).
        """

    @pc.setter
    def pc(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def hs(self) -> bool:
        """
        The indicator h_sigma: False when ||ps|| is long (Hansen 2016, App. A, symbols of Fig. 6), which stalls the update of pc; always True with CSA and in the (1+1)-ES (#160). With `StepSizeAdaptation.XNES` and `Modules.active` (not with UTILITY weights), False also drops the negative weights of the generation's update (#175).
        """

    @hs.setter
    def hs(self, arg: bool, /) -> None: ...

    @property
    def skipped_downdates(self) -> int:
        """
        The rank-one downdates of the active update that were skipped, since the adaptation was created (read-only; restarts do not reset it).

        A downdate is skipped when its result would not be positive definite: the factor from before the downdates of the generation is restored and they are applied again without it. The rescaled negative weights prevent this in exact arithmetic; UTILITY's weights do not.
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class SelfAdaptation(Adaptation):
    """
    CMSA-ES, covariance matrix self-adaptation (`MatrixAdaptationType.CMSA`).

    C = (1 - 1/tc) C + (1/tc) sum_i w_i y_i y_i^T with tc = 1 + d (d + 1) / (2 mueff), without an evolution path, then a Cholesky factorization C = A A^T every generation (O(d^3); LAPACK from d = `constants.lapack_min_dim`). With CMSA the default mu is lambda / 4. CMSA has no active update: Settings switches `Modules.active` off (#73), and with UTILITY weights the positive utilities are used. The CMSA-ES of the paper adapts the step size by self-adaptation (`StepSizeAdaptation.SA`). Beyer & Sendhoff (2008).
    """

    def __init__(self, dimension: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Create the adaptation with mean x0 and the initial matrix (the identity).

        Parameters
        ----------
        dimension : int
            The dimension d.
        x0 : numpy.ndarray
            The initial mean m, shape (d,).
        """

    @property
    def A(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The lower triangular Cholesky factor of C, y = A z, shape (d, d) (the identity at the start without `Settings.C0`).
        """

    @A.setter
    def A(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def C(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The covariance matrix (without sigma^2), shape (d, d) (the identity at the start, or `Settings.C0`).
        """

    @C.setter
    def C(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class CovarianceNoEigvAdaptation(CovarianceAdaptation):
    """
    CMA-ES with a Cholesky factorization instead of an eigendecomposition (`MatrixAdaptationType.COVARIANCE_NO_EIGV`).

    C is updated as in `CovarianceAdaptation` and factorized as C = A A^T with A lower triangular, every `es.p.weights.lazy_update_interval` generations as COVARIANCE decomposes it (Eigen, or LAPACK dpotrf from d = `constants.lapack_min_dim`, see `constants.cholesky_solver`), which is several times cheaper than an eigendecomposition (docs/performance.md). B and d are filled only by `metric()`, from A A^T, the C of the last factorization. ps is updated with dz instead of C^-1/2 dm, which only CSA needs: Settings selects this adaptation in place of COVARIANCE for the other step-size adaptations.
    """

    def __init__(self, dimension: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Create the adaptation with mean x0 and the initial matrix (the identity).

        Parameters
        ----------
        dimension : int
            The dimension d.
        x0 : numpy.ndarray
            The initial mean m, shape (d,).
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """

class NaturalGradientAdaptation(Adaptation):
    """
    xNES, exponential natural evolution strategies (`MatrixAdaptationType.NATURAL_GRADIENT`).

    The sampling matrix is updated multiplicatively, A = A expm(cc G), with the natural gradient G = sum_i w_i (z_i z_i^T - I) made trace-free (its trace part, `sigma_g`, goes to the step size with `StepSizeAdaptation.XNES`) and cc = `es.p.weights.cc`, xNES's eta_B / 2 by default (1 / (4 d^1.5) in the (1+1)-ES). With fewer samples k in the gradient than d (mu, or every point with `Modules.active`) and d >= 16 the matrix exponential is computed exactly from a thin QR of the samples, O(d^2 k); otherwise as a dense d x d exponential. Glasmachers et al. (2010).

    With `Modules.active` and DEFAULT weights Settings uses the utilities of xNES (`RecombinationWeights.UTILITY`, which sum to zero) instead of the negative weights of CMA-ES; with `StepSizeAdaptation.XNES` and UTILITY this is xNES.
    """

    def __init__(self, dimension: int, x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]) -> None:
        """
        Create the adaptation with mean x0 and A = I (the step size sets the spread).

        Parameters
        ----------
        dimension : int
            The dimension d.
        x0 : numpy.ndarray
            The initial mean m, shape (d,).
        """

    @property
    def A(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The sampling matrix, y = A z, shape (d, d) (the identity at the start without `Settings.C0`).
        """

    @A.setter
    def A(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def A_inv(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The inverse of A, shape (d, d). Recomputed only when `invert_y` needs it (an LU inverse from d = `constants.blas_min_dim`, else the pseudo-inverse), so it can lag behind A (the identity at the start without `Settings.C0`).
        """

    @A_inv.setter
    def A_inv(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def G(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The natural gradient G = Z diag(u) Z^T - alpha I as a dense (d, d) matrix, computed on access from `gradient_samples` (Z), `gradient_weights` (u) and `gradient_shift` (alpha); the update does not form it.
        """

    @property
    def gradient_samples(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')]:
        """
        The samples Z of the last gradient, the z of the population as columns, shape (d, k): the mu best, or with `Modules.active` every point (with elitism, the mu best and the worst of the (mu + lambda) pool, #161).
        """

    @gradient_samples.setter
    def gradient_samples(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='F')], /) -> None: ...

    @property
    def gradient_weights(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]:
        """
        The weights u of the samples, shape (k,), not rescaled (with `Modules.active`, negative weights for the worst; with elitism, the worst of the (mu + lambda) pool, #161).
        """

    @gradient_weights.setter
    def gradient_weights(self, arg: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], /) -> None: ...

    @property
    def gradient_shift(self) -> float:
        """
        The shift alpha of G: the sum of the weights plus `sigma_g`, which makes G trace-free (the sum of the weights alone in a (1+1)-ES).
        """

    @gradient_shift.setter
    def gradient_shift(self, arg: float, /) -> None: ...

    @property
    def sigma_g(self) -> float:
        """
        trace(G) / d of the gradient before the shift: its isotropic (step-size) part, removed from G and used by `StepSizeAdaptation.XNES`, sigma *= exp(cs / 2 * sigma_g), with cs xNES's eta_sigma by default (see `Weights.cs`). With elitism XNES computes it from the offspring alone instead (#172). 0 in the (1+1)-ES.
        """

    @sigma_g.setter
    def sigma_g(self, arg: float, /) -> None: ...

    def compute_gradients(self, pop: modcma.c_maes.cmaescpp.Population, weights: modcma.c_maes.cmaescpp.parameters.Weights, settings: modcma.c_maes.cmaescpp.parameters.Settings) -> None:
        """
        Compute the natural gradient of the population: `gradient_samples`, `gradient_weights`, `gradient_shift` and `sigma_g`. Called by `adapt_evolution_paths`.

        Parameters
        ----------
        pop : Population
            The selected population (sorted, best first).
        weights : Weights
            The recombination weights and learning rates (`es.p.weights`).
        settings : Settings
            The settings (`es.p.settings`).
        """

    def __getstate__(self) -> object:
        """
        The state for pickle and copy (#91): the bytes of the object's C++ serialization (the format is described in the guide's page on saving and resuming a run), or (bytes, dict) with the attributes of a Python subclass. TypeError when the object holds a module that the library cannot save, such as a sampler defined in Python.
        """

    def __setstate__(self, state: object) -> None:
        """
        Restore the object from a state of __getstate__ (pickle and copy). ValueError when the state cannot be loaded: not a modcma state, written by a newer modcma, truncated, corrupted, or the state of another class; TypeError when it is not bytes.
        """
