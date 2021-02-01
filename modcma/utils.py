"""Implementation of various utilities used in ModularCMA-ES package."""

import warnings
import typing
from inspect import Signature, Parameter, getmodule
from functools import wraps
from time import time

import numpy as np


class Descriptor:
    """Data descriptor."""

    def __set_name__(self, owner, name):
        """Set name attribute."""
        self.name = name

    def __set__(self, instance, value):
        """Set value on instance."""
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        """Delete attribute from the instance __dict__."""
        del instance.__dict__[self.name]


class InstanceOf(Descriptor):
    """Data descriptor checks for correct types."""

    def __init__(self, dtype):
        """Set dtype."""
        self.dtype = dtype

    def __set__(self, instance, value):
        """Set the value of instance to value, checks type of argument.

        Raises
        ------
        TypeError
            If type of the argument does not match self.dtype      

        """
        if type(value) != type(None):

            if (
                type(value) != self.dtype
                and not (
                    isinstance(value, np.generic) and type(value.item()) == self.dtype
                )
                and str(self.dtype)[1:] != value.__class__.__name__
            ):
                raise TypeError(
                    "{} should be {} got type {}: {}".format(
                        self.name, self.dtype, type(value), str(value)[:50]
                    )
                )
            if hasattr(value, "__copy__"):
                value = value.copy()
        super().__set__(instance, value)


class AnyOf(Descriptor):
    """Descriptor, checks of value is Any of a specified sequence of options."""

    def __init__(self, options=None):
        """Set options."""
        self.options = options

    def __set__(self, instance, value):
        """Set the value of instance to value, checks value of argument to match self.options.

        Raises
        ------
        TypeError
            If type of the argument does not match self.dtype        

        """
        if value not in self.options:
            raise ValueError(
                "{} should be any of [{}]. Got: {}".format(
                    self.name, self.options, value)
            )
        super().__set__(instance, value)


class AnnotatedStructMeta(type):
    """Metaclass for class for AnnotatedStruct.

    Wraps all parameters defined in the class body with
    __annotations__ into a signature. It additionally wraps each
    parameter into a descriptor using __annotations__,
    allowing for type checking.
    Currently, only two types of descriptors are implementated,
    InstanceOf and typing.AnyOf, the first implements simple type validation,
    the latter implements validation though the use of sequence of
    allowed values.
    """

    def __new__(cls: typing.Any, name: str, bases: tuple, attrs: dict) -> typing.Any:
        """Control instance creation of classes that have AnnotatedStructMeta as metaclass.

        All cls attributes that are defined in __annotations__ are wrapped
        into either an typing.AnyOf or an InstanceOf descriptor, depending on
        the type of the annotation. If the annotation is a sequence, the first
        element is used as a default value.

        Parameters
        ----------
        cls: typing.Any
            The empty body of the class to be instantiated
        name: str
            The name of the cls
        bases: dict
            The base classes of the cls
        attrs: dict
            The attributes of the cls

        Returns
        -------
        A new cls object

        """
        parameters = []
        for key, annotation in attrs.get("__annotations__", {}).items():
            default_value = attrs.get(key, Parameter.empty)

            if isinstance(annotation, (list, tuple)):
                attrs[key] = AnyOf(annotation)
            else:
                if (
                    not type(annotation) == type
                    and getmodule(type(annotation)) != typing
                ):
                    raise TypeError(
                        f"Detected wrong format for annotations of AnnotatedStruct.\n\t"
                        f"Format should be <name>: <type> = <default_value>\n\t"
                        f"Got: {name}: {annotation} = {default_value}"
                    )
                attrs[key] = InstanceOf(annotation)
            parameters.append(
                Parameter(
                    name=key,
                    default=default_value,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                )
            )

        clsobj = super().__new__(cls, name, bases, attrs)
        setattr(clsobj, "__signature__", Signature(parameters=parameters))
        return clsobj


class AnnotatedStruct(metaclass=AnnotatedStructMeta):
    """Custom class for defining structs.

    Automatically sets parameters defined in the signature.
    AnnotatedStruct objects, and children thereof, require
    the following structure:
        class Foo(AnnotatedStruct):
            variable_wo_default : type
            variable_w_default  : type = value

    The metaclass will automatically assign a decriptor object
    to every variable, allowing for type checking.
    The init function will be dynamically generated, and user specified values
    in the *args **kwargs, will override the defaults.
    The *args will follow the order as defined in the class body:
        i.e. (variable_wo_default, variable_w_default,)

    Attributes
    ----------
    __signature__: Signature
        The calling signature, instantiated by the metaclass
    __bound__ : Signature
        The bound signature, bound to the *args and **kwargs

    """

    def __init__(self, *args, **kwargs) -> None:
        """Bind *args and **kwargs to a signature instantiated by the metaclass."""
        self.__bound__ = self.__signature__.bind(*args, **kwargs)
        self.__bound__.apply_defaults()
        for name, value in self.__bound__.arguments.items():
            setattr(self, name, value)

    def __repr__(self) -> str:
        """Representation for a AnnotatedStruct object."""
        return "<{}: ({})>".format(
            self.__class__.__qualname__,
            ", ".join(
                "{}={}".format(name, getattr(self, name))
                for name, value in self.__bound__.arguments.items()
            ),
        )

    def set_default(self, name: str, default_value: typing.Any) -> None:
        """Helper method to set default parameters."""
        current = getattr(self, name)
        if type(current) == type(None):
            setattr(self, name, default_value)


def timeit(func):
    """Decorator function for timing the excecution of a function.

    Parameters
    ----------
    func: typing.Callable
        The function to be timed

    Returns
    -------
    typing.Callable
        a wrapped function

    """
    @wraps(func)
    def inner(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        print("Time elapsed", time() - start)
        return res

    return inner


def ert(evals, budget):
    """Computed the expected running time of a list of evaluations.

    Parameters
    ----------
    evals: list
        a list of running times (number of evaluations)
    budget: int
        the maximum number of evaluations

    Returns
    -------
    float
        The expected running time

    float
        The standard deviation of the expected running time
    int
        The number of successful runs

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evals = np.array(evals)
            n_succ = (evals < budget).sum()
            _ert = float(evals.sum()) / int(n_succ)
        return _ert, np.std(evals), n_succ
    except ZeroDivisionError:
        return float("inf"), np.nan, 0
