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
    ForbiddenGreaterThanRelation,
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
from ._warm_start import warm_start

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

    settings = Parameters(Settings(dim))
    default = getattr(settings.weights, name, None)
    if default is None:
        default = getattr(settings.settings, name)

    if isinstance(default, int):
        return UniformIntegerHyperparameter(name, lb, ub, default, log=True)

    elif isinstance(default, float):
        db = min(default - lb, ub - default)
        return NormalFloatHyperparameter(name, default, 0.3 * db, lb, ub)

    raise TypeError(
        f"default value for {name} ({default}) "
        f"has an unparsable type {type(default)}"
    )


#: The lower bound of `cs`, `cc` and `damps` (#169). The ranges are [0, 1] (`damps`: [0, 10]) with
#: a normal distribution of standard deviation 0.3 * min(default - lb, ub - default) around the
#: default (0.37, 0.45, 1.37 at dim 5), so a draw within 1e-6 of 0 is a 3-sigma event at least
#: a million times over, and a lower bound of 1e-6 moves that deviation by at most 3e-7 (relative
#: 1e-6). It is the same kind of bound as the 1e-15 of `sigma0`. The defaults are unchanged.
_MIN_POSITIVE_RATE = 1e-6

#: Options that `get_configspace` leaves out of a module, though `Modules` has them (#81, F22 of
#: #76: the raw uniform samples have mean 0.5, which fails everywhere; a testing option).
_EXCLUDED_OPTIONS = {"sample_transformation": ("NONE",)}

#: When a hyperparameter has no effect (#81). Each entry is read from the C++ code: a
#: hyperparameter is inactive only where the code never reads it. A hyperparameter is active when
#: any of its parents has an option that is not excluded here. The evidence (src/):
#:
#: * `damps`: read only by CSA, MSR, PSR, SR and TPA_INPLACE (mutation.cpp, their `adapt`). TPA
#:   reads cs, XNES, MXNES, LPXNES read cs, SA neither. No other file reads it (weights.cpp stores
#:   it, and uses cs for its default only).
#: * `cc`: read by the path pc of COVARIANCE, COVARIANCE_NO_EIGV (matrix_adaptation.cpp:155, 163),
#:   SEPARABLE (386, 399), CHOLESKY (558) and the (1+1) path (144, 376, 553), and by the step
#:   of NATURAL_GRADIENT (822). MATRIX (444-452) and CMSA (624-632) and NONE have no pc.
#: * `cs`: is read directly by TPA, MSR, PSR and the xNES rules (mutation.cpp, their `adapt`), and
#:   is the rate of the path ps, at most 1 (`Weights.cs_path`, Adaptation::adapt_ps), for every
#:   step-size rule. ps is used by CSA (CSA::adapt), by MATRIX (its rank-one update and h_sigma)
#:   and, through h_sigma, by COVARIANCE, COVARIANCE_NO_EIGV, SEPARABLE and, with a rule other
#:   than CSA, CHOLESKY (#160). SR and SA never read cs, and neither do NONE, CMSA or
#:   NATURAL_GRADIENT read ps: with both, cs is unread. (Before #72 the default cc of
#:   NATURAL_GRADIENT was cs / 2. CHOLESKY in a (1+1)-ES does not read ps; cs stays active
#:   there.)
#: * `c1`, `cmu`: read by COVARIANCE, COVARIANCE_NO_EIGV (also through `lazy_update_interval`),
#:   SEPARABLE, MATRIX and CHOLESKY. They also scale the negative weights (Weights::init_weights),
#:   which the update of every adaptation reads with `active` (for_population_into), so
#:   NATURAL_GRADIENT reads them only with `active`. NONE and CMSA read no negative weights.
#: * `active`: read by every adaptation but NONE and CMSA; Settings switches it off for NONE and
#:   CMSA (#73) and for lambda0 = 1 (the Settings constructor).
#: * `mu0`: Settings sets mu = 1 for lambda0 = 1 (settings.hpp:211).
#: * `elitist`, `sequential_selection`, `weights`, `ssa`: Settings overrides them for lambda0 = 1,
#:   a (1+1)-ES (settings.hpp:209-222). Settings turns COVARIANCE into COVARIANCE_NO_EIGV after
#:   that, from the `ssa` that runs, SR (settings.hpp:224-232, #131), so `ssa` is never read and
#:   COVARIANCE with lambda0 = 1 is COVARIANCE_NO_EIGV: forbidden. The (1+1)-ES runs SR, which
#:   reads `damps` (mutation.cpp:372), so `damps` is active for lambda0 = 1.
#: * IPOP and BIPOP become RESTART for lambda0 = 1 (settings.hpp:219-221): forbidden.
#: * `mirrored` PAIRWISE becomes NONE for lambda0 = 1 (settings.hpp:170-171, #136): forbidden.
#: * `center_placement`, `repelling_type`: used when a run restarts (parameters.cpp:127,
#:   `Parameters::begin_restart`, repelling.cpp:174). The restart criteria of every strategy but
#:   STOP include UnableToAdapt (restart_criteria.cpp:403), so NONE restarts when the adaptation
#:   fails. With STOP the criteria stop the optimizer instead (c_maes.cpp:101, 225-232). The
#:   other criteria exist from RESTART on (restart_criteria.cpp:405).
#: * `repelling_metric`: read by CoverageRepelling and AdaptiveRepelling only (repelling.cpp:28,
#:   47); `repelling.hpp:235` gives `NoRepelling` for NONE.
_INACTIVE_FOR = {
    "damps": {"ssa": ("TPA", "XNES", "MXNES", "LPXNES", "SA")},
    "cc": {"matrix_adaptation": ("NONE", "MATRIX", "CMSA")},
    "cs": {
        "ssa": ("SR", "SA"),
        "matrix_adaptation": ("NONE", "CMSA", "NATURAL_GRADIENT"),
    },
    "center_placement": {"restart_strategy": ("STOP",)},
    "repelling_type": {"restart_strategy": ("STOP",)},
    "repelling_metric": {"repelling_type": ("NONE",)},
}
#: Modules that Settings overrides for lambda0 = 1
_ONE_PLUS_ONE_OVERRIDDEN = ("elitist", "sequential_selection", "weights", "ssa")
#: `c1` and `cmu` also when `active` with these matrix adaptations (see above)
_ACTIVE_ONLY_MATRIX = ("NATURAL_GRADIENT",)
#: Settings switches `active` off with these matrix adaptations (#73 for CMSA)
_NO_ACTIVE_MATRIX = ("NONE", "CMSA")
_NO_MATRIX_UPDATE_OF_C = ("NONE", "CMSA", "NATURAL_GRADIENT")


def _in_condition(child, parent, options: tuple, excluded: tuple = ()):
    """`child` is active when `parent` has an option that is not in `excluded`."""
    active = [o for o in options if o not in excluded]
    return InCondition(child, parent, active)


def _any_of(conditions: list):
    return conditions[0] if len(conditions) == 1 else OrConjunction(*conditions)


def _add_conditions(cspace: ConfigurationSpace, options: dict, remove_duplicates: bool) -> None:
    """The conditions of the inactive options, and the forbidden duplicates (#81)."""
    present = set(cspace.keys())
    ssa, matrix = cspace["ssa"], cspace["matrix_adaptation"]

    # active: needs a matrix other than CMSA, and a population (lambda0 = 1 switches it off)
    active: list[Any] = [
        _in_condition(cspace["active"], matrix, options["matrix_adaptation"], _NO_ACTIVE_MATRIX)
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
        if child == "damps" and "lambda0" in present:
            conditions.append(LessThanCondition(cspace[child], cspace["lambda0"], 2))
        cspace.add(_any_of(conditions))

    for child in ("c1", "cmu"):
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

    if "mu0" in present:
        cspace.add(GreaterThanCondition(cspace["mu0"], cspace["lambda0"], 1))
    if "lambda0" in present:
        for child in _ONE_PLUS_ONE_OVERRIDDEN:
            cspace.add(GreaterThanCondition(cspace[child], cspace["lambda0"], 1))

    if not remove_duplicates:
        return
    if "CSA" in options["ssa"]:
        # Settings turns COVARIANCE into COVARIANCE_NO_EIGV unless ssa is CSA
        others = [o for o in options["ssa"] if o != "CSA"]
        cspace.add(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(matrix, "COVARIANCE"), ForbiddenInClause(ssa, others)
            )
        )
    if "lambda0" not in present:
        return
    # A (1+1)-ES runs SR whatever ssa is, so COVARIANCE becomes COVARIANCE_NO_EIGV (#131)
    if "COVARIANCE" in options["matrix_adaptation"]:
        cspace.add(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(cspace["lambda0"], 1),
                ForbiddenEqualsClause(matrix, "COVARIANCE"),
            )
        )
    if "restart_strategy" in options:
        ipop = [o for o in ("IPOP", "BIPOP") if o in options["restart_strategy"]]
        cspace.add(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(cspace["lambda0"], 1),
                ForbiddenInClause(cspace["restart_strategy"], ipop),
            )
        )
    # Settings turns PAIRWISE into NONE for lambda0 = 1 (#136)
    if "PAIRWISE" in options.get("mirrored", ()):
        cspace.add(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(cspace["lambda0"], 1),
                ForbiddenEqualsClause(cspace["mirrored"], "PAIRWISE"),
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

    - `lambda0` and `mu0`: integers in [1, 50 * dim], log scale; mu0 > lambda0 is forbidden;
    - `sigma0`: a normal distribution around its default, truncated to [1e-15, 1e15];
    - `cs`, `cc`, `cmu`, `c1` in [0, 1] and `damps` in [0, 10]: normal distributions around
      their defaults. The lower bound of `cs`, `cc` and `damps` is 1e-6, because `Settings`
      rejects a value of 0 or less; `cmu` and `c1` may be 0.

    The ranges of `c1` and `cmu` are independent, so a configuration can have c1 + cmu > 1. `Settings`
    rejects it (C would lose positive definiteness, Hansen 2016): `settings_from_config` raises
    `ValueError` when it builds the `Settings`, not later when the optimizer is created. A tuner
    should record such a configuration as a crashed run. Every other draw of the space is valid.

    The defaults are those of `Settings(dim)` with the default modules (CSA with COVARIANCE). The
    normal distributions have standard deviation 0.3 * min(default - lower, upper - default).
    `settings_from_config` treats a value equal to its default as "the default of the chosen
    modules", which differs per step-size rule and matrix adaptation.

    Hyperparameters without an effect are inactive (ConfigSpace conditions, absent from a sampled
    configuration):

    - `active` needs a `matrix_adaptation` other than `NONE` and `CMSA`, and `lambda0` above 1
      (`Settings` switches it off otherwise);
    - `center_placement` and `repelling_type` are inactive for `restart_strategy` `STOP` (the
      criteria stop the run; `NONE` still restarts when the adaptation fails), and
      `repelling_metric` needs `repelling_type` other than `NONE`;
    - `cs` is inactive when `ssa` is `SR` or `SA` and `matrix_adaptation` is `NONE`, `CMSA`
      or `NATURAL_GRADIENT`; `damps` for `TPA`, `XNES`, `MXNES`, `LPXNES` and `SA`;
    - `cc` is inactive for `matrix_adaptation` `NONE`, `MATRIX` and `CMSA`;
    - `cmu` and `c1` are inactive for `NONE` and `CMSA`, and for `NATURAL_GRADIENT` without
      `active` (with `active` they scale its negative weights);
    - `mu0`, `elitist`, `sequential_selection`, `weights` and `ssa` need `lambda0` above 1
      (a (1+1)-ES has mu = 1, is elitist, with equal weights, and runs `SR` whatever `ssa` is);
      `damps` is active for `lambda0` = 1, which runs `SR`.

    A hyperparameter is inactive only where the C++ code never reads it; the evidence, with file
    and line, is in the comment of `_INACTIVE_FOR` in the source.

    `Settings` replaces `matrix_adaptation=COVARIANCE` by `COVARIANCE_NO_EIGV` for every `ssa` but
    `CSA`, and for `lambda0` = 1 (which runs `SR`), so the two are one configuration there. With
    `remove_duplicates`, `COVARIANCE` with another `ssa` than `CSA` or with `lambda0` = 1 is
    forbidden, and so are `IPOP` or `BIPOP` (`RESTART` there) and `mirrored=PAIRWISE` (`NONE`
    there) with `lambda0` = 1. Two duplicates stay:
    `mirrored=PAIRWISE` makes an odd `lambda0` above 1 even, so `lambda0` = k and k + 1 (k odd,
    k > 1) are one configuration; and with mu = lambda
    `active` and `sequential_selection` have nothing to act on (there are no negative weights, no
    unselected offspring). The space does not forbid them.

    The numeric parameters not in this list (`ipop_factor`, the threshold of threshold
    convergence, ...) are not hyperparameters.

    Turn the result into `Settings` with `settings_from_config`.

    Parameters
    ----------
    dim : int, optional
        The dimension, for the numeric defaults. Without it, and with any numeric
        hyperparameter, a warning is issued and dim = 2 is used.
    add_popsize : bool, default True
        Add `lambda0` and `mu0`.
    add_sigma : bool, default True
        Add `sigma0`.
    add_learning_rates : bool, default True
        Add `cs`, `cc`, `cmu`, `c1` and `damps`.
    remove_duplicates : bool, default True
        Forbid the configurations that `Settings` turns into another one of the space
        (`COVARIANCE` with an `ssa` other than `CSA`; `COVARIANCE`, `IPOP`, `BIPOP` or
        `mirrored=PAIRWISE` with `lambda0` = 1).

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
        cspace.add(_make_numeric_parameter("mu0", dim, 1, 50 * dim))
        cspace.add(ForbiddenGreaterThanRelation(cspace["mu0"], cspace["lambda0"]))

    if add_sigma:
        cspace.add(_make_numeric_parameter("sigma0", dim, 1e-15, 1e15))

    if add_learning_rates:
        for name in ("cs", "cc", "cmu", "c1", "damps"):
            ub = 10.0 if name == "damps" else 1.0
            # `Settings` rejects cs, cc and damps <= 0 (#147), so their range starts above 0 (#169).
            # c1 and cmu may be 0 (no rank-one or rank-mu update), so theirs starts at 0.
            lb = _MIN_POSITIVE_RATE if name in ("cs", "cc", "damps") else 0.0
            cspace.add(_make_numeric_parameter(name, dim, lb, ub))

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

    The module hyperparameters set the modules. A numeric hyperparameter is passed to `Settings`
    only when it differs from its default in `get_configspace(dim)`; a value equal to the default
    stands for the default of the chosen modules, which `Settings` computes itself (the defaults of
    the step-size rules, the matrix adaptations and of mu = lambda / 4 differ from those of
    `CSA` with `COVARIANCE`, which the space has). A different value is passed as it is. `mu0` is
    the exception: it is passed whenever `lambda0` is passed, and whenever it differs from its
    default, so a `mu0` equal to the default is kept next to a `lambda0` that differs from its own
    (the default mu0 is lambda0 // 2 of the default lambda0). A `mu0` above `lambda0` raises
    `ValueError` (`Settings` rejects it; the space forbids it).

    Parameters
    ----------
    dim : int
        The dimension.
    config : ConfigSpace.Configuration or mapping
        The configuration, e.g. from a tuner or `get_configspace(dim).sample_configuration()`; or a
        dict of the same names.
    **kwargs
        Further `Settings` keyword arguments, e.g. `lb`, `ub` or `budget`.

    Returns
    -------
    Settings
        The settings of the configuration.
    """
    via_settings = kwargs
    default_config = dict(get_configspace(dim).get_default_configuration())
    modules = Modules()
    for name, value in dict(config).items():
        if set_module(modules, name, value):
            continue
        if name == "mu0":
            continue
        if name not in default_config or default_config[name] != value:
            via_settings[name] = value

    mu0 = dict(config).get("mu0")
    if mu0 is not None and ("lambda0" in via_settings or default_config.get("mu0") != mu0):
        via_settings["mu0"] = mu0

    settings = Settings(dim, modules, **via_settings)
    return settings


# The presets: published algorithms that are a selection of modules. Each entry has the keyword
# arguments of `settings_from_dict` and the source of the algorithm (see `preset`).
_PRESETS: dict[str, tuple[dict[str, Any], str]] = {
    "cma-es": ({}, "Hansen & Ostermeier 2001; Hansen 2016"),
    "active-cma-es": ({"active": True}, "Jastrebski & Arnold 2006"),
    "sep-cma": ({"matrix_adaptation": "SEPARABLE"}, "Ros & Hansen 2008"),
    "ma-es": ({"matrix_adaptation": "MATRIX"}, "Beyer & Sendhoff 2017"),
    "cholesky-cma": ({"matrix_adaptation": "CHOLESKY"}, "Krause, Arbones & Igel 2016"),
    "cmsa-es": ({"matrix_adaptation": "CMSA", "ssa": "SA"}, "Beyer & Sendhoff 2008"),
    "xnes": (
        {"matrix_adaptation": "NATURAL_GRADIENT", "ssa": "XNES", "weights": "UTILITY"},
        "Glasmachers et al. 2010",
    ),
    "ipop": ({"restart_strategy": "IPOP"}, "Auger & Hansen 2005"),
    "bipop": ({"restart_strategy": "BIPOP"}, "Hansen 2009"),
    "(1+1)-cma-es": ({"lambda0": 1}, "Igel, Suttorp & Hansen 2006"),
}


def preset_names() -> tuple:
    """The names of the presets of `preset`, in the order of the guide."""
    return tuple(_PRESETS)


def preset(name: str, dim: int, **config: Any) -> Settings:
    """`Settings` of a published algorithm that is a selection of modules.

    Also available as `Settings.preset(name, dim, ...)`, and as `fmin(..., preset=name)`. The
    names are matched without regard to case; `_` and `-` are the same.

    ================  ==========================================  ==============================
    name              modules                                     source
    ================  ==========================================  ==============================
    `cma-es`          the defaults (COVARIANCE, CSA)              Hansen & Ostermeier 2001;
                                                                  Hansen 2016
    `active-cma-es`   `active=True`                               Jastrebski & Arnold 2006
    `sep-cma`         `matrix_adaptation=SEPARABLE`               Ros & Hansen 2008
    `ma-es`           `matrix_adaptation=MATRIX`                  Beyer & Sendhoff 2017
    `cholesky-cma`    `matrix_adaptation=CHOLESKY`                Krause, Arbones & Igel 2016
    `cmsa-es`         `matrix_adaptation=CMSA`, `ssa=SA`          Beyer & Sendhoff 2008
    `xnes`            `matrix_adaptation=NATURAL_GRADIENT`,       Glasmachers et al. 2010
                      `ssa=XNES`, `weights=UTILITY`
    `ipop`            `restart_strategy=IPOP`                     Auger & Hansen 2005
    `bipop`           `restart_strategy=BIPOP`                    Hansen 2009
    `(1+1)-cma-es`    `lambda0=1`                                 Igel, Suttorp & Hansen 2006
    ================  ==========================================  ==============================

    The preset is the selection of modules; the learning rates and the population size keep the
    defaults that `Settings` computes for those modules. The `(1+1)-cma-es` is the elitist (1+1)
    strategy that `lambda0=1` gives: mu = 1,
    plus selection, the 1/5th-type success rule for sigma (target success rate 2/11) and a
    covariance update from a success-based evolution path. It is built from the modules of modcma,
    and is not a reimplementation of the published algorithm, whose update differs in details.

    Parameters
    ----------
    name : str
        The name of the preset, see above. Unknown names raise ValueError, listing the valid ones.
    dim : int
        The dimension.
    **config
        Module options and `Settings` keyword arguments, as for `settings_from_dict`. They are
        applied after the preset and override it: `preset("sep-cma", 10, active=True)` is
        SEPARABLE with the active update, and `preset("ipop", 10, restart_strategy="BIPOP")` is
        BIPOP.

    Returns
    -------
    Settings
        The settings of the preset.

    Examples
    --------
    >>> settings = preset("sep-cma", 10, budget=5000)
    >>> settings.modules.matrix_adaptation.name
    'SEPARABLE'
    """
    key = str(name).strip().lower().replace("_", "-")
    if key not in _PRESETS:
        raise ValueError(f"unknown preset {name!r}; the presets are: {', '.join(_PRESETS)}")
    return settings_from_dict(dim, **{**_PRESETS[key][0], **config})


# Settings is a binding class; the preset is a Python helper, attached as a static method
_preset = preset  # `fmin` has an argument of that name
Settings.preset = staticmethod(preset)  # pyright: ignore[reportAttributeAccessIssue]


def fmin(
    func: Callable[[np.ndarray], float],
    x0: Any,
    sigma0: float,
    budget: int,
    preset: Optional[str] = None,
    gradient: Optional[Callable[[np.ndarray], Any]] = None,
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
        The maximum number of function evaluations. It must hold the first generation (lambda0,
        and TPA's two points) and the evaluation of the final center of mass that ends every
        run; a smaller budget raises ValueError.
    preset : str, optional
        The name of a published algorithm (see `modcma.preset`), e.g. `"sep-cma"`. The keyword
        arguments below override it.
    gradient : callable, optional
        The gradient of `func`, called with a (d,) numpy array; returns d numbers. Every
        generation, it is called at the center of mass, and a point along the negative gradient
        from there replaces one sample
        (gradient injection, `ModularCMAES.inject_gradient`). It costs no evaluations; a bad
        gradient costs the replaced sample.
    **kwargs
        Module options and `Settings` keyword arguments, as for `settings_from_dict`, e.g.
        `active=True`, `matrix_adaptation="MATRIX"`, `lb=..., ub=...`, `target=1e-8`.

    Returns
    -------
    xopt : numpy.ndarray
        (d,) the best point found, `es.p.stats.global_best.x` (a copy: it does not change if
        the run continues). With a bound correction, the best point in [lb, ub] (#168).
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
    config = dict(x0=x0, sigma0=sigma0, budget=budget, **kwargs)
    if preset is None:
        settings = settings_from_dict(len(x0), **config)
    else:
        settings = _preset(preset, len(x0), **config)
    es = ModularCMAES(settings)
    es(func, gradient=gradient)

    return (
        es.p.stats.global_best.x,
        es.p.stats.global_best.y,
        es.p.stats.evaluations,
        es,
    )


__all__ = (
    "fmin",
    "warm_start",
    "settings_from_dict",
    "settings_from_config",
    "preset",
    "preset_names",
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
