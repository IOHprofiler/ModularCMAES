"""Main implementation of Modular CMA-ES."""
from inspect import Parameter
import os
from itertools import islice
from typing import List, Callable

import numpy as np

from .parameters import Parameters
from .population import Population
from .utils import timeit, ert


class ModularCMAES:
    r"""The main class of the configurable CMA ES continous optimizer.

    Attributes
    ----------
    _fitness_func: callable
        The objective function to be optimized
    parameters: Parameters
        All the parameters of the CMA ES algorithm are stored in
        the parameters object. Note if a parameters object is not
        explicitly passed, all \*args and \**kwargs passed into the
        constructor of a ModularCMAES are directly passed into
        the constructor of a Parameters object.

    See Also
    --------
    modcma.parameters.Parameters

    """

    parameters: "Parameters"
    _fitness_func: Callable

    def __init__(
        self, fitness_func: Callable, *args, parameters=None, **kwargs
    ) -> None:
        """Set _fitness_func and forwards all other parameters to Parameters object."""
        self._fitness_func = fitness_func
        self.parameters = (
            parameters
            if isinstance(parameters, Parameters)
            else Parameters(*args, **kwargs)
        )

    def mutate(self) -> None:
        """Apply mutation operation.

        First, a directional vector zi is sampled from a sampler object
        as defined in the self.parameters object. Then, this zi vector is
        multiplied with the eigenvalues D, and the dot product is taken with the
        eigenvectors B of the covariance matrix C in order to create a scaled
        directional mutation vector yi. By scaling this vector with current population
        mean m, and the step size sigma, a new individual xi is created. The
        self.fitness_func is called in order to compute the fitness of the newly created
        individuals.

        If the step size adaptation method is 'tpa', two less 'normal'
        individuals are created.

        #TODO: make bound correction vectorized and integrate with tpa
        """
        perform_tpa = bool(
            self.parameters.step_size_adaptation == "tpa"
            and self.parameters.old_population
        )

        n_offspring = int(self.parameters.lambda_ - (2 * perform_tpa))

        if self.parameters.step_size_adaptation == 'lp-xnes' or self.parameters.sample_sigma:
            s = np.random.lognormal(
                np.log(self.parameters.sigma),
                self.parameters.beta, size=n_offspring
            )
        else:
            s = np.ones(n_offspring) * self.parameters.sigma

        z = np.hstack(tuple(islice(self.parameters.sampler, n_offspring)))
        if self.parameters.threshold_convergence:
            z = scale_with_threshold(z, self.parameters.threshold)

        y = np.dot(self.parameters.B, self.parameters.D * z)
        x = self.parameters.m + (s * y)
        x, n_out_of_bounds = correct_bounds(
            x, 
            self.parameters.ub,
            self.parameters.lb,
            self.parameters.bound_correction
        )
        self.parameters.n_out_of_bounds += n_out_of_bounds
        
        if not self.parameters.sequential and self.parameters.vectorized_fitness:
            f = self._fitness_func(x.T)
            self.parameters.used_budget += len(x.T)       
        else:
            f = np.empty(n_offspring, object)
            for i in range(n_offspring):
                f[i] = self.fitness_func(x[:, i])
                if self.sequential_break_conditions(i, f[i]):
                    f = f[:i]
                    s = s[:i]
                    x = x[:, :i]
                    y = y[:, :i]
                    break

        if perform_tpa:
            yt, xt, ft = tpa_mutation(self.fitness_func, self.parameters)
            y = np.c_[yt, y]
            x = np.c_[xt, x]
            f = np.r_[ft, f]
            s = np.r_[np.repeat(self.parameters.sigma, 2), s]

        self.parameters.population = Population(x, y, f, s)

    def select(self) -> None:
        """Selection of best individuals in the population.

        The population is sorted according to their respective fitness
        values. Normally, the mu best individuals would be selected afterwards.
        However, because the option of active update is available, (and we could
        potentially need the mu worst individuals) the lambda best indivduals are
        selected. In recombination, only the mu best individuals are used to recompute
        the mean, so implicited selection happens there.

        If elistism is selected as an option, the mu best individuals of the old
        population are added to the pool of indivduals before sorting.

        If selection is to be performed pairwise, the only the best individuals
        of sequential pairs are used, the others are discarded. The intended
        use for this functionality is with mirrored sampling, in order to counter the
        bias generated by this sampling method. This method cannot be performed when there
        is an odd number of individuals in the population.
        """
        if self.parameters.mirrored == "mirrored pairwise":
            if not len(self.parameters.population.f) % 2 == 0:
                raise ValueError(
                    "Cannot perform pairwise selection with "
                    "an odd number of indivuduals"
                )
            indices = [
                int(np.argmin(x) + (i * 2))
                for i, x in enumerate(
                    np.split(
                        self.parameters.population.f,
                        len(self.parameters.population.f) // 2,
                    )
                )
            ]
            self.parameters.population = self.parameters.population[indices]

        if self.parameters.elitist and self.parameters.old_population:
            self.parameters.population += self.parameters.old_population[
                : self.parameters.mu
            ]
        self.parameters.population.sort()

        self.parameters.population = self.parameters.population[
            : self.parameters.lambda_
        ]

        if self.parameters.population.f[0] < self.parameters.fopt:
            self.parameters.fopt = self.parameters.population.f[0]
            self.parameters.xopt = self.parameters.population.x[:, 0]

    def recombine(self) -> None:
        """Recombination of new individuals.

        In the CMAES, recombination is not as explicit as in for example
        a genetic algorithm. In the CMAES, recombination happens though the
        moving of the mean m, by multiplying the old mean with a weighted
        combination of the current mu best individuals.
        TODO: check if this should be moved to parameters
        """
        self.parameters.m_old = self.parameters.m.copy()
        self.parameters.m = self.parameters.m_old + (
            1
            * (
                (
                    self.parameters.population.x[:, : self.parameters.mu]
                    - self.parameters.m_old
                )
                @ self.parameters.pweights
            ).reshape(-1, 1)
        )

    def step(self) -> bool:
        """The step method runs one iteration of the optimization process.

        The method is called within the self.run loop. There, a while loop runs
        until this step function returns a Falsy value.

        Returns
        -------
        bool
            Denoting whether to keep running this step function.

        """
        self.mutate()
        self.select()
        self.recombine()
        self.parameters.adapt()
        return not any(self.break_conditions)

    def sequential_break_conditions(self, i: int, f: float) -> bool:
        """Indicator whether there are any sequential break conditions.

        Parameters
        ----------
        i: int
            The number of individuals already generated this current
            generation.
        f: float
            The fitness of that individual

        Returns
        -------
        bool

        """
        if self.parameters.sequential:
            return (
                f < self.parameters.fopt
                and i >= self.parameters.seq_cutoff
                and (self.parameters.mirrored != "mirrored pairwise" or i % 2 == 0)
            )
        return False

    def run(self):
        """Run the step method until step method retuns a falsy value.

        Returns
        -------
        ModularCMAES

        """
        while self.step():
            pass
        return self

    @property
    def break_conditions(self) -> List[bool]:
        """A list with break conditions based on the parameters of the CMA-ES.

        Returns
        -------
        [bool, bool]

        """
        if self.parameters.n_generations:
            return [self.parameters.t >= self.parameters.n_generations]
        return [
            self.parameters.target >= self.parameters.fopt,
            self.parameters.used_budget >= self.parameters.budget,
        ]

    def fitness_func(self, x: np.ndarray) -> float:
        """Wrapper function for calling self._fitness_func.

        Adds 1 to self.parameters.used_budget for each fitnes function
        call.

        Parameters
        ----------
        x: np.ndarray
            array on which to call the objective/fitness function

        Returns
        -------
        float

        """
        self.parameters.used_budget += 1
        return self._fitness_func(x.flatten())

    def __repr__(self):
        """Representation of ModularCMA-ES."""
        return f"<{self.__class__.__qualname__}: {self._fitness_func}>"

    def __str__(self):
        """String representation of ModularCMA-ES."""
        return repr(self)


def tpa_mutation(fitness_func: Callable, parameters: "Parameters") -> None:
    """Helper function for applying the tpa mutation step.

    The code was mostly taken from the ModEA framework,
    and there a slight differences with the procedure as defined in:
    Nikolaus Hansen. CMA-ES with two-point step-size adaptation.CoRR, abs/0805.0231,2008.
    The function should not be used outside of the ModularCMAES optimizer

    Parameters
    ----------
    fitness_func: typing.Callable
        A fitness function to be optimized
    parameters: Parameters
        A modcma Parameters object
    x: list
        A list of new individuals
    y: list
        A list of new mutation vectors
    f: list
        A list of fitnesses

    """
    yi = (parameters.m - parameters.m_old) / parameters.sigma
    y = np.c_[yi, -yi]
    x = parameters.m + (parameters.sigma * y[:, :2])
    f = np.array(list(map(fitness_func, x[:, :2].T)))
    if f[1] < f[0]:
        parameters.rank_tpa = -parameters.a_tpa
    else:
        parameters.rank_tpa = parameters.a_tpa + parameters.b_tpa
    return y, x, f


def scale_with_threshold(z: np.ndarray, threshold: float) -> np.ndarray:
    """Function for scaling a vector z to have length > threshold.

    Used for threshold convergence.

    Parameters
    ----------
    z : np.ndarray
        the vector to be scaled
    threshold : float
        the length threshold the vector should at least be

    Returns
    -------
    np.ndarray
        a scaled version of z

    """
    length = np.linalg.norm(z, axis=0)
    mask = length < threshold
    z[:, mask] *= (threshold + (threshold - length[mask])) / length[mask]
    return z


def correct_bounds(
    x: np.ndarray, ub: np.ndarray, lb: np.ndarray, correction_method: str
) -> np.ndarray:
    """Bound correction function.

    Rescales x to fall within the lower lb and upper
    bounds ub specified. Available strategies are:
    - None: Don't perform any boundary correction
    - unif_resample: Resample each coordinate out of bounds uniformly within bounds
    - mirror: Mirror each coordinate around the boundary
    - COTN: Resample each coordinate out of bounds using the one-sided normal
    distribution with variance 1/3 (bounds scaled to [0,1])
    - saturate: Set each out-of-bounds coordinate to the boundary
    - toroidal: Reflect the out-of-bounds coordinates to the oposite bound inwards

    Parameters
    ----------
    x: np.ndarray
        vector of which the bounds should be corrected
    ub: float
        upper bound
    lb: float
        lower bound
    correction_method: string
        type of correction to perform

    Returns
    -------
    np.ndarray
        bound corrected version of x
    bool
        whether the population was out of bounds

    Raises
    ------
    ValueError
        When an unkown value for correction_method is provided

    """
    out_of_bounds = np.logical_or(x > ub, x < lb)
    n_out_of_bounds = out_of_bounds.max(axis=0).sum()
    if n_out_of_bounds == 0 or correction_method is None:
        return x, n_out_of_bounds

    try:
        _, n = x.shape
    except ValueError:
        n = 1
    ub, lb = np.tile(ub, n)[out_of_bounds], np.tile(lb, n)[out_of_bounds]
    y = (x[out_of_bounds] - lb) / (ub - lb)

    if correction_method == "mirror":
        x[out_of_bounds] = lb + (ub - lb) * np.abs(
            y - np.floor(y) - np.mod(np.floor(y), 2)
        )
    elif correction_method == "COTN":
        x[out_of_bounds] = lb + (ub - lb) * np.abs(
            (y > 0) - np.abs(np.random.normal(0, 1 / 3, size=y.shape))
        )
    elif correction_method == "unif_resample":
        x[out_of_bounds] = np.random.uniform(lb, ub)
    elif correction_method == "saturate":
        x[out_of_bounds] = lb + (ub - lb) * (y > 0)
    elif correction_method == "toroidal":
        x[out_of_bounds] = lb + (ub - lb) * np.abs(y - np.floor(y))
    else:
        raise ValueError(f"Unknown argument: {correction_method} for correction_method")
    return x, n_out_of_bounds


@timeit
def evaluate_bbob(
    fid,
    dim,
    iterations=50,
    label="",
    logging=False,
    data_folder=None,
    seed=42,
    instance=1,
    target_precision=1e-8,
    return_optimizer=False,
    **kwargs,
):
    """Helper function to evaluate a ModularCMAES on the BBOB test suite.

    Parameters
    ----------
    fid: int
        The id of the function 1 - 24
    dim: int
        The dimensionality of the problem
    iterations: int = 50
        The number of iterations to be performed.
    label: str = ''
        The label to be given to the run, used for logging with BBOB
    logging: bool = False
        Specifies whether to use logging
    data_folder: str = None
        File path where to store data when logging = True
    seed: int = 42
        The random seed to be used
    instance: int = 1
        The bbob function instance
    target_precision: float = 1e-8
        The target precision for the objective function value
    return_optimizer: bool = False
        Whether to return the optimizer
    **kwargs
        These are directly passed into the instance of ModularCMAES,
        in this manner parameters can be specified for the optimizer.

    Returns
    -------
    evals
        The number of evaluations for each run of the optimizer
    fopts
        The best fitness values for each run of the optimizer

    """
    # This speeds up the import, this import is quite slow, so import it lazy here
    # pylint: disable=import-outside-toplevel
    import ioh

    evals, fopts = np.array([]), np.array([])
    if seed:
        np.random.seed(seed)
    fitness_func = ioh.get_problem(
        fid, dimension=dim, instance=instance
    )

    if logging:
        data_location = data_folder if os.path.isdir(data_folder) else os.getcwd()
        logger = ioh.logger.Analyzer(root=data_location, folder_name=f"{label}F{fid}_{dim}D")
        fitness_func.attach_logger(logger)

    print(
        f"Optimizing function {fid} in {dim}D for target "
        f"{target_precision} with {iterations} iterations."
    )

    for idx in range(iterations):
        if idx > 0:
            fitness_func.reset()
        target = fitness_func.objective.y + target_precision

        optimizer = ModularCMAES(fitness_func, dim, target=target, **kwargs).run()
        evals = np.append(evals, fitness_func.state.evaluations)
        fopts = np.append(fopts, fitness_func.state.current_best_internal.y)

    result_string = (
        "FCE:\t{:10.8f}\t{:10.4f}\n"
        "ERT:\t{:10.4f}\t{:10.4f}\n"
        "{}/{} runs reached target"
    )
    print(
        result_string.format(
            np.mean(fopts),
            np.std(fopts),
            *ert(evals, optimizer.parameters.budget),
            iterations,
        )
    )
    if return_optimizer:
        return evals, fopts, optimizer
    return evals, fopts


def fmin(func, dim, maxfun=None, **kwargs):
    """Minimize a function using the modular CMA-ES.

    Parameters
    ----------
    func: callable
        The objective function to be minimized.
    dim: int
        The dimensionality of the problem
    maxfun: int = None
        Maximum number of function evaluations to make.
    **kwargs
        These are directly passed into the instance of ModularCMAES,
        in this manner parameters can be specified for the optimizer.

    Returns
    -------
    xopt
        The variables which minimize the function during this run
    fopt
        The value of function at found xopt
    evals
        The number of evaluations performed

    """
    cma = ModularCMAES(func, dim, budget=maxfun, **kwargs).run()
    return cma.parameters.xopt, cma.parameters.fopt, cma.parameters.used_budget
