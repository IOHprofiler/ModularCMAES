import warnings
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Optional, Union, cast

import numpy as np
from ConfigSpace import (
    AndConjunction,
    ConfigurationSpace,
    Categorical,
    EqualsCondition,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    GreaterThanCondition,
    InCondition,
    LessThanCondition,
    NormalFloatHyperparameter,
    OrConjunction,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from .cmaescpp import (
    constants,
    sampling,  # pyright: ignore[reportMissingModuleSource]
    mutation,  # pyright: ignore[reportMissingModuleSource]
    matrix_adaptation,  # pyright: ignore[reportMissingModuleSource]
    selection,  # pyright: ignore[reportMissingModuleSource]
    parameters,  # pyright: ignore[reportMissingModuleSource]
    bounds,  # pyright: ignore[reportMissingModuleSource]
    restart,  # pyright: ignore[reportMissingModuleSource]
    options,  # pyright: ignore[reportMissingModuleSource]
    repelling,  # pyright: ignore[reportMissingModuleSource]
    Population,
    Parameters,
    ModularCMAES,
    center,  # pyright: ignore[reportMissingModuleSource]
    es,  # pyright: ignore[reportMissingModuleSource]
    set_seed,
    build_info,
    compute_ert,
    _internal,  # pyright: ignore[reportMissingModuleSource]
)
from . import utils  # deprecated: set_seed, build_info and compute_ert are in this namespace

from .cmaescpp.parameters import (  # pyright: ignore[reportMissingModuleSource]
    Settings,
    Modules,
)  # pyright: ignore[reportMissingModuleSource]


def _use_scipy_lapack() -> None:
    """Give the C++ library scipy's LAPACK for the eigendecomposition in high dimension.

    The macOS wheels link Apple Accelerate themselves; the Linux and Windows wheels have no LAPACK
    of their own and use scipy's (a dependency of modcma, already imported by modcma). Without it
    the library falls back to Eigen's eigensolver. See docs/performance.md.
    """
    if build_info()["lapack"] != "none":
        return
    try:
        from scipy.linalg import cython_blas, cython_lapack  # pyright: ignore

        lapack, blas = cython_lapack.__pyx_capi__, cython_blas.__pyx_capi__
        capsules = {name: lapack[name] for name in ("dsyevd", "dsyevr", "dpotrf", "dgetrf", "dgetri")}
        capsules["dgemm"] = blas["dgemm"]
        _internal.set_lapack(capsules, "scipy")
    except Exception as error:  # pragma: no cover - depends on the scipy build
        warnings.warn(f"modcma: scipy's LAPACK is not usable ({error}); using Eigen", stacklevel=2)


_use_scipy_lapack()


def _get_module_options(name: str) -> tuple:
    if not hasattr(Modules, name):
        raise NameError(f"Modules has no member {name}")

    default_value = getattr(Modules(), name)
    if isinstance(default_value, bool):
        return (default_value, not default_value)

    module_class = default_value.__class__
    if issubclass(module_class, Enum):
        other_values = [
            x.name for x in module_class.__members__.values() if x is not default_value
        ]
        return tuple([default_value.name] + other_values)
    raise TypeError(f"{name} has a unparsable type {type(default_value)}")


def get_all_module_options() -> dict:
    """The options of every module of `Modules`.

    Returns
    -------
    dict
        Maps the name of every `Modules` attribute to a tuple of its options, the default first.
        A bool module gives `(default, not default)`; an enum module gives the names (str) of
        its options in `modcma.c_maes.options`, e.g. `"COVARIANCE"`.
    """
    return {
        name: _get_module_options(name)
        for name in dir(Modules)
        if not name.startswith("_")
    }


def _make_numeric_parameter(
    name: str, dim: int, lb: float, ub: float
) -> Union[UniformIntegerHyperparameter, NormalFloatHyperparameter]:
    """An absolute numeric hyperparameter (`lambda0`, `sigma0`), with the default of `Settings`."""
    default = getattr(Settings(dim), name)

    if isinstance(default, int):
        return UniformIntegerHyperparameter(name, lb, ub, default, log=True)

    elif isinstance(default, float):
        db = min(default - lb, ub - default)
        return NormalFloatHyperparameter(name, default, 0.3 * db, lb, ub)

    raise TypeError(
        f"default value for {name} ({default}) "
        f"has an unparsable type {type(default)}"
    )


#: Options that `get_configspace` leaves out of a module, though `Modules` has them (#81, F22 of
#: #76: the raw uniform samples have mean 0.5, which fails everywhere; a testing option).
_EXCLUDED_OPTIONS = {"sample_transformation": ("NONE",)}

#: The numeric parameters of the space that are a factor of the default of the chosen modules
#: (`<name>_factor`): the name, and the range of the factor. The range is an open decision
#: (#81): [0.1, 10] on a log scale is the proposal of the issue; the factor of mu is at most 2,
#: since mu is at most lambda (the default mu is lambda / 2 or lambda / 4).
_FACTOR_RANGES = {
    "mu0": (0.1, 2.0),
    "cs": (0.1, 10.0),
    "cc": (0.1, 10.0),
    "cmu": (0.1, 10.0),
    "c1": (0.1, 10.0),
    "damps": (0.1, 10.0),
}

#: When a hyperparameter has no effect (#81). Each entry is read from the C++ code: a
#: hyperparameter is inactive only where the code never reads it. A hyperparameter is active when
#: any of its parents has an option that is not excluded here. The evidence (src/):
#:
#: * `damps`: read only by CSA, MSR, PSR and SR (mutation.cpp:217, 283, 320, 372). TPA reads cs
#:   (249), XNES, MXNES, LPXNES read cs (333, 340, 351, 363), SA neither (381-415). No other file
#:   reads it (weights.cpp stores it, and uses cs for its default only).
#: * `cc`: read by the path pc of COVARIANCE, COVARIANCE_NO_EIGV (matrix_adaptation.cpp:155, 163),
#:   SEPARABLE (386, 399), CHOLESKY (558) and the (1+1) path (144, 376, 553), and by the step
#:   of NATURAL_GRADIENT (822). MATRIX (444-452) and CMSA (624-632) and NONE have no pc.
#: * `cs`: updates the path ps in every adaptation (matrix_adaptation.cpp:107, 134) and is read
#:   directly by TPA, MSR, PSR and the xNES rules (mutation.cpp:249, 282, 319, 333-363). ps is
#:   used by CSA (mutation.cpp:217), by MATRIX (matrix_adaptation.cpp:482-485) and, through h_sigma,
#:   by COVARIANCE and COVARIANCE_NO_EIGV (152-155) and SEPARABLE (383-386). The default of cc
#:   uses it for NATURAL_GRADIENT (weights.cpp:78). SR (mutation.cpp:372) and SA never read it,
#:   and neither do NONE, CHOLESKY (546-558) or CMSA (624-632) read ps: with both, cs is unread.
#: * `c1`, `cmu`: read by COVARIANCE, COVARIANCE_NO_EIGV (matrix_adaptation.cpp:175, 200, 204;
#:   `lazy_update_interval`, weights.cpp:197, used at 328), SEPARABLE (407, 412), MATRIX (474,
#:   475) and CHOLESKY (576-594). They also scale the negative weights (weights.cpp:186-189), which
#:   the update of every adaptation reads with `active` (for_population_into: 166, 402, 465, 592,
#:   648, 771), so CMSA and NATURAL_GRADIENT read them only with `active`. NONE reads no
#:   weights (it has no matrix update).
#: * `active`: read by every adaptation but NONE (matrix_adaptation.cpp:165, 401, 464, 590, 647,
#:   770); Settings switches it off for NONE and for lambda0 = 1 (settings.hpp:150, 158).
#: * `mu0`: Settings sets mu = 1 for lambda0 = 1 (settings.hpp:155).
#: * `elitist`, `sequential_selection`, `weights`: Settings overrides them for lambda0 = 1, a
#:   (1+1)-ES (settings.hpp:153-161). `ssa` is overridden too (SR), but not before Settings turns
#:   COVARIANCE into COVARIANCE_NO_EIGV for an `ssa` other than CSA (settings.hpp:125-130), so it
#:   stays active. The (1+1)-ES runs SR, which reads `damps` (mutation.cpp:372), so `damps_factor`
#:   is also active for lambda0 = 1.
#: * IPOP and BIPOP become RESTART for lambda0 = 1 (settings.hpp:163-165): forbidden.
#: * `center_placement`, `repelling_type`: used when a run restarts (parameters.cpp:128,
#:   `Parameters::begin_restart`, repelling.cpp:174). The restart criteria of every strategy but
#:   STOP include UnableToAdapt (restart_criteria.cpp:376), so NONE restarts when the adaptation
#:   fails. With STOP the criteria stop the optimizer instead (c_maes.cpp:101, 225-232). The
#:   other criteria exist from RESTART on (restart_criteria.cpp:379).
#: * `repelling_metric`: read by CoverageRepelling and AdaptiveRepelling only (repelling.cpp:28,
#:   47); `repelling.hpp:235` gives `NoRepelling` for NONE.
_INACTIVE_FOR = {
    "damps_factor": {"ssa": ("TPA", "XNES", "MXNES", "LPXNES", "SA")},
    "cc_factor": {"matrix_adaptation": ("NONE", "MATRIX", "CMSA")},
    "cs_factor": {"ssa": ("SR", "SA"), "matrix_adaptation": ("NONE", "CHOLESKY", "CMSA")},
    "center_placement": {"restart_strategy": ("STOP",)},
    "repelling_type": {"restart_strategy": ("STOP",)},
    "repelling_metric": {"repelling_type": ("NONE",)},
}
#: Modules that Settings overrides for lambda0 = 1
_ONE_PLUS_ONE_OVERRIDDEN = ("elitist", "sequential_selection", "weights")
#: `c1` and `cmu` also when `active` with these matrix adaptations (see above)
_ACTIVE_ONLY_MATRIX = ("CMSA", "NATURAL_GRADIENT")
_NO_MATRIX_UPDATE_OF_C = ("NONE", "CMSA", "NATURAL_GRADIENT")


def _in_condition(child, parent, options: tuple, excluded: tuple = ()):
    """`child` is active when `parent` has an option that is not in `excluded`."""
    active = [o for o in options if o not in excluded]
    return InCondition(child, parent, active)


def _factor_parameter(name: str) -> UniformFloatHyperparameter:
    lower, upper = _FACTOR_RANGES[name]
    return UniformFloatHyperparameter(f"{name}_factor", lower, upper, default_value=1.0, log=True)


def _any_of(conditions: list):
    return conditions[0] if len(conditions) == 1 else OrConjunction(*conditions)


def _add_conditions(cspace: ConfigurationSpace, options: dict, remove_duplicates: bool) -> None:
    """The conditions of the inactive options, and the forbidden duplicates (#81)."""
    present = set(cspace.keys())
    ssa, matrix = cspace["ssa"], cspace["matrix_adaptation"]

    # active: needs a matrix, and a population (lambda0 = 1 switches it off)
    active: list[Any] = [
        _in_condition(cspace["active"], matrix, options["matrix_adaptation"], ("NONE",))
    ]
    if "lambda0" in present:
        active.append(GreaterThanCondition(cspace["active"], cspace["lambda0"], 1))
    cspace.add(active[0] if len(active) == 1 else AndConjunction(*active))

    for child, parents in _INACTIVE_FOR.items():
        if child not in present:
            continue
        conditions: list[Any] = [
            _in_condition(cspace[child], cspace[parent], options[parent], excluded)
            for parent, excluded in parents.items()
        ]
        if child == "damps_factor" and "lambda0" in present:
            conditions.append(LessThanCondition(cspace[child], cspace["lambda0"], 2))
        cspace.add(_any_of(conditions))

    for child in ("c1_factor", "cmu_factor"):
        if child not in present:
            continue
        read_always = _in_condition(
            cspace[child], matrix, options["matrix_adaptation"], _NO_MATRIX_UPDATE_OF_C
        )
        with_active = AndConjunction(
            InCondition(cspace[child], matrix, list(_ACTIVE_ONLY_MATRIX)),
            EqualsCondition(cspace[child], cspace["active"], True),
        )
        cspace.add(OrConjunction(read_always, with_active))

    if "mu0_factor" in present:
        cspace.add(GreaterThanCondition(cspace["mu0_factor"], cspace["lambda0"], 1))
    if "lambda0" in present:
        for child in _ONE_PLUS_ONE_OVERRIDDEN:
            cspace.add(GreaterThanCondition(cspace[child], cspace["lambda0"], 1))

    if remove_duplicates and "CSA" in options["ssa"]:
        # Settings turns COVARIANCE into COVARIANCE_NO_EIGV unless ssa is CSA
        others = [o for o in options["ssa"] if o != "CSA"]
        cspace.add(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(matrix, "COVARIANCE"), ForbiddenInClause(ssa, others)
            )
        )
        if "lambda0" in present and "restart_strategy" in options:
            ipop = [o for o in ("IPOP", "BIPOP") if o in options["restart_strategy"]]
            cspace.add(
                ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cspace["lambda0"], 1),
                    ForbiddenInClause(cspace["restart_strategy"], ipop),
                )
            )


def get_configspace(
    dim: Optional[int] = None,
    add_popsize: bool = True,
    add_sigma: bool = True,
    add_learning_rates: bool = True,
    remove_duplicates: bool = True,
) -> ConfigurationSpace:
    """A ConfigSpace configuration space of the modules and the numeric settings, for tuning.

    Every module of `Modules` is a categorical hyperparameter with its options from
    `get_all_module_options` (default: the default of `Modules`), except that
    `sample_transformation` has no `NONE` (raw uniform samples, a testing option that fails
    everywhere). The numeric hyperparameters are:

    - `lambda0`: an integer in [1, 50 * dim], log scale, with the default of `Settings(dim)`;
    - `mu0_factor`: mu as a factor of its default (lambda / 2, or lambda / 4 for `SA` and
      `CMSA`), in [0.1, 2], log scale, default 1; mu is at least 1 and at most lambda;
    - `sigma0`: a normal distribution around its default, truncated to [1e-15, 1e15];
    - `cs_factor`, `cc_factor`, `cmu_factor`, `c1_factor`, `damps_factor`: a factor of the
      default of the chosen modules, in [0.1, 10], log scale, default 1.

    A factor is relative to the default of the modules of the configuration, which differs per
    step-size rule and matrix adaptation, so the default configuration is the standard CMA-ES and
    every other configuration is as far from its own default (see `settings_from_config`).

    Hyperparameters without an effect are inactive (ConfigSpace conditions, absent from a sampled
    configuration):

    - `active` needs a `matrix_adaptation` other than `NONE` and `lambda0` above 1;
    - `center_placement` and `repelling_type` are inactive for `restart_strategy` `STOP` (the
      criteria stop the run; `NONE` still restarts when the adaptation fails), and
      `repelling_metric` needs `repelling_type` other than `NONE`;
    - `cs_factor` is inactive when `ssa` is `SR` or `SA` and `matrix_adaptation` is `NONE`,
      `CHOLESKY` or `CMSA`; `damps_factor` for `TPA`, `XNES`, `MXNES`, `LPXNES` and `SA`;
    - `cc_factor` is inactive for `matrix_adaptation` `NONE`, `MATRIX` and `CMSA`;
    - `cmu_factor` and `c1_factor` are inactive for `NONE`, and for `CMSA` and
      `NATURAL_GRADIENT` without `active` (with `active` they scale the negative weights);
    - `mu0_factor`, `elitist`, `sequential_selection` and `weights` need `lambda0` above 1
      (a (1+1)-ES has mu = 1, is elitist, with equal weights); it runs `SR` whatever `ssa` is, but
      `ssa` stays, since it decides whether `COVARIANCE` becomes `COVARIANCE_NO_EIGV`; `damps_factor`
      is active for `lambda0` = 1, which runs `SR`.

    A hyperparameter is inactive only where the C++ code never reads it; the evidence, with file
    and line, is in the comment of `_INACTIVE_FOR` in the source.

    `Settings` replaces `matrix_adaptation=COVARIANCE` by `COVARIANCE_NO_EIGV` for every `ssa` but
    `CSA`, so the two are one configuration there. With `remove_duplicates`, `COVARIANCE` with
    another `ssa` than `CSA` is forbidden, and so is `IPOP` or `BIPOP` with `lambda0` = 1
    (`RESTART` there). Two duplicates stay: `mirrored=PAIRWISE` makes an odd `lambda0` even, so
    `lambda0` = k and k + 1 (k odd) are one configuration; and with mu = lambda (`mu0_factor` at
    its largest) `active` and `sequential_selection` have nothing to act on (there are no
    negative weights, no unselected offspring). The space does not forbid them.

    The numeric parameters not in this list (`ipop_factor`, the threshold of threshold
    convergence, ...) are not hyperparameters.

    Turn the result into `Settings` with `settings_from_config`.

    Parameters
    ----------
    dim : int, optional
        The dimension, for the numeric defaults. Without it, and with any numeric
        hyperparameter, a warning is issued and dim = 2 is used.
    add_popsize : bool, default True
        Add `lambda0` and `mu0_factor`.
    add_sigma : bool, default True
        Add `sigma0`.
    add_learning_rates : bool, default True
        Add `cs_factor`, `cc_factor`, `cmu_factor`, `c1_factor` and `damps_factor`.
    remove_duplicates : bool, default True
        Forbid the configurations that `Settings` turns into another one of the space
        (`COVARIANCE` with an `ssa` other than `CSA`).

    Returns
    -------
    ConfigSpace.ConfigurationSpace
        The configuration space.
    """
    options = {
        name: tuple(o for o in values if o not in _EXCLUDED_OPTIONS.get(name, ()))
        for name, values in get_all_module_options().items()
    }
    cspace = ConfigurationSpace()
    for name, values in options.items():
        cspace.add(Categorical(name, values, default=values[0]))

    if dim is None:
        if add_popsize or add_sigma or add_learning_rates:
            warnings.warn(
                "Filling configspace with default numeric values for dim=2, since no dim was "
                "provided and add_popsize, add_sigma or add_learning_rates is True"
            )
        dim = 2

    if add_popsize:
        cspace.add(_make_numeric_parameter("lambda0", dim, 1, 50 * dim))
        cspace.add(_factor_parameter("mu0"))

    if add_sigma:
        cspace.add(_make_numeric_parameter("sigma0", dim, 1e-15, 1e15))

    if add_learning_rates:
        for name in ("cs", "cc", "cmu", "c1", "damps"):
            cspace.add(_factor_parameter(name))

    _add_conditions(cspace, options, remove_duplicates)
    return cspace


def set_module(modules: Modules, name: str, value: Union[str, bool, Enum]) -> bool:
    """Set one module of a `Modules` object by name, if it is a module.

    Parameters
    ----------
    modules : Modules
        The configuration to change (in place).
    name : str
        The name of a `Modules` attribute, e.g. `"matrix_adaptation"`.
    value : str, enum member or bool
        For an enum module, the option, by name (e.g. `"MATRIX"`) or as a member of its enum
        (e.g. `options.MatrixAdaptationType.MATRIX`); for a bool module, a bool.

    Returns
    -------
    bool
        True when `name` is a module and was set, False otherwise (nothing changes).
    """
    if hasattr(modules, name):
        attr_class = type(getattr(modules, name))
        if issubclass(attr_class, Enum) and not isinstance(value, attr_class):
            value = getattr(attr_class, cast(str, value))
        setattr(modules, name, value)
        return True
    return False


def settings_from_dict(dim: int, **config: Any) -> Settings:
    """`Settings` from keyword arguments that mix module options and settings.

    A keyword that names a `Modules` attribute sets that module (an enum option by its name,
    e.g. `matrix_adaptation="MATRIX"`); every other keyword goes to the `Settings` constructor
    (e.g. `budget`, `lb`, `lambda0`, `seed`).

    Parameters
    ----------
    dim : int
        The dimension.
    **config
        Module options and `Settings` keyword arguments.

    Returns
    -------
    Settings
        The settings, with the modules set.

    Examples
    --------
    >>> settings = settings_from_dict(10, active=True, matrix_adaptation="MATRIX", budget=5000)
    """
    modules = Modules()
    via_settings = {}
    for name, value in dict(config).items():
        if set_module(modules, name, value):
            continue
        via_settings[name] = value
    settings = Settings(dim, modules, **via_settings)
    return settings


def settings_from_config(dim: int, config: Mapping[str, Any], **kwargs: Any) -> Settings:
    """`Settings` from a configuration of the space of `get_configspace`.

    The module hyperparameters set the modules. `lambda0`, `sigma0` and any other name that is
    not a module go to `Settings` as they are, also when they equal the default. A hyperparameter
    `<name>_factor` (`mu0`, `cs`, `cc`, `cmu`, `c1`, `damps`) multiplies the default of the chosen
    modules, with the population size of the configuration: for the factor 1 (or an inactive
    hyperparameter, which is absent from `config`) nothing is passed, and `Settings` computes the
    default itself. The factors are relative to the defaults of the configuration without the
    other factors: `cc_factor = 2` doubles the `cc` that `Settings` gives for the chosen modules,
    also when `cs_factor` changes the default `cc` of `NATURAL_GRADIENT`. mu is rounded, at least
    1 and at most lambda.

    Parameters
    ----------
    dim : int
        The dimension.
    config : ConfigSpace.Configuration or mapping
        The configuration, e.g. from a tuner or `get_configspace(dim).sample_configuration()`; or a
        dict of the same names. A dict may also give the numeric `Settings` arguments themselves
        (`cs`, `mu0`, ...), as the spaces before 2.0 did.
    **kwargs
        Further `Settings` keyword arguments, e.g. `lb`, `ub` or `budget`.

    Returns
    -------
    Settings
        The settings of the configuration.
    """
    via_settings = kwargs
    modules = Modules()
    factors = {}
    for name, value in dict(config).items():
        if set_module(modules, name, value):
            continue
        base = name[: -len("_factor")]
        if name.endswith("_factor") and base in _FACTOR_RANGES:
            if value != 1.0:
                factors[base] = value
            continue
        via_settings[name] = value

    if factors:
        # the defaults of these modules at this lambda, from Settings and Weights
        reference = Settings(dim, modules, **via_settings)
        weights = Parameters(reference).weights
        for name, factor in factors.items():
            if name == "mu0":
                mu = round(reference.mu0 * factor)
                via_settings[name] = min(max(1, mu), reference.lambda0)
            else:
                via_settings[name] = getattr(weights, name) * factor

    settings = Settings(dim, modules, **via_settings)
    return settings


def fmin(
    func: Callable[[np.ndarray], float],
    x0: Any,
    sigma0: float,
    budget: int,
    **kwargs: Any,
) -> tuple[np.ndarray, float, int, ModularCMAES]:
    """Minimize a function with the modular CMA-ES.

    Builds `Settings` with `settings_from_dict`, runs a `ModularCMAES` until a break condition
    holds and returns the best solution found.

    Parameters
    ----------
    func : callable
        The objective function, called with a (d,) numpy array; returns a float.
    x0 : array_like
        The starting point; its length is the dimension d.
    sigma0 : float
        The initial step size (rule of thumb: 0.3 * (ub - lb)).
    budget : int
        The maximum number of function evaluations, at least one generation (lambda0, and
        TPA's two points); a smaller budget raises ValueError.
    **kwargs
        Module options and `Settings` keyword arguments, as for `settings_from_dict`, e.g.
        `active=True`, `matrix_adaptation="MATRIX"`, `lb=..., ub=...`, `target=1e-8`.

    Returns
    -------
    xopt : numpy.ndarray
        (d,) the best point found (a copy: it does not change if the run continues).
    fopt : float
        The objective value of xopt.
    evals : int
        The number of evaluations used.
    es : ModularCMAES
        The optimizer, with its state in `es.p`.

    Examples
    --------
    >>> import numpy as np
    >>> from modcma.c_maes import fmin
    >>> xopt, fopt, evals, es = fmin(lambda x: float(np.sum(x**2)), np.ones(5), 0.5, 2000)
    """
    x0 = np.asarray(x0, dtype=float)
    settings = settings_from_dict(len(x0), x0=x0, sigma0=sigma0, budget=budget, **kwargs)
    es = ModularCMAES(settings)
    es(func)

    return (
        es.p.stats.global_best.x,
        es.p.stats.global_best.y,
        es.p.stats.evaluations,
        es,
    )


__all__ = (
    "fmin",
    "settings_from_dict",
    "settings_from_config",
    "get_configspace",
    "get_all_module_options",
    "Settings",
    "Modules",
    "constants",
    "set_seed",
    "build_info",
    "compute_ert",
    "sampling",
    "mutation",
    "matrix_adaptation",
    "selection",
    "parameters",
    "bounds",
    "restart",
    "options",
    "repelling",
    "Population",
    "Parameters",
    "ModularCMAES",
    "center",
    "es",
)
