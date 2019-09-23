from collections import OrderedDict
from collections.abc import Sequence
from inspect import Signature, Parameter
from datetime import datetime
from functools import wraps
from time import time
import numpy as np

from bbob import bbobbenchmarks, fgeneric
from Constants import DEFAULT_TARGET_DISTANCES, DISTANCE_TO_TARGET


class Descriptor:
    '''Data descriptor'''

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class InstanceOf(Descriptor):
    def __init__(self, dtype):
        self.dtype = dtype

    def __set__(self, instance, value):
        if type(value) != type(None):
            if type(value) != self.dtype and (
                isinstance(value, np.generic) and type(
                    np.asscalar(value)) != self.dtype):
                raise TypeError("{} should be {}".format(
                    self.name, self.dtype))
            if hasattr(value, '__copy__'):
                value = value.copy()
        super().__set__(instance, value)

    @property
    def __doc__(self):
        return super().__doc__ + " checks for type {}".format(self.dtype)


class AnyOf(Descriptor):
    def __init__(self, options=None):
        self.options = options

    def __set__(self, instance, value):
        if value not in self.options:
            raise TypeError("{} should any of {}".format(
                self.name, self.options
            ))
        super().__set__(instance, value)

    @property
    def __doc__(self):
        return (
            super().__doc__ + " checks if value is any of: [{}]".format(
                ', '.join(map(str, self.options))
            )
        )


class AnnotatedStructMeta(type):
    '''Metaclass for class for AnnotatedStruct.
    Wraps all parameters defined in the class body with 
    __annotations__ into a signature. It additionally wraps each 
    parameter into a descriptor using __annotations__, allowing for type checking. 
    '''
    @classmethod
    def __prepare__(cls, name, bases):
        return OrderedDict()

    def __new__(cls, name, bases, attrs):
        parameters = []
        for key, value in attrs.get('__annotations__', {}).items():
            default_value = attrs.get(key, Parameter.empty)
            if isinstance(default_value, Sequence):
                attrs[key] = AnyOf(default_value)
                parameters.append(Parameter(name=key, default=default_value[0],
                                            kind=Parameter.POSITIONAL_OR_KEYWORD))
            else:
                attrs[key] = InstanceOf(value)
                parameters.append(Parameter(name=key, default=default_value,
                                            kind=Parameter.POSITIONAL_OR_KEYWORD))

        clsobj = super().__new__(cls, name, bases, attrs)
        setattr(clsobj, '__signature__', Signature(parameters=parameters))
        return clsobj


class AnnotatedStruct(metaclass=AnnotatedStructMeta):
    '''Custom class for defining structs. Automatically 
    sets parameters defined in the signature. 
    '''

    def __init__(self, *args, **kwargs) -> None:
        self.__bound__ = self.__signature__.bind(*args, **kwargs)
        self.__bound__.apply_defaults()
        for name, value in self.__bound__.arguments.items():
            setattr(self, name, value)

    def __repr__(self) -> None:
        return "<{}: ({})>".format(
            self.__class__.__qualname__, ', '.join(
                "{}={}".format(name, value)
                for name, value in self.__bound__.arguments.items()
            )
        )


def _scale_with_threshold(z, threshold):
    length = np.linalg.norm(z)
    if length < threshold:
        new_length = threshold + (threshold - length)
        z *= (new_length / length)
    return z


def _keepInBounds(x, l_bound, u_bound):
    """
        This function transforms x to t w.r.t. the low and high
        boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
        described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
        for Parameter Optimization and Their Applications to Medical Image
        Analysis" as alorithm 6.

        :param x:       Column vector to be kept in bounds
        :param l_bound: Lower bound column vector
        :param u_bound: Upper bound column vector
        :returns:       An in-bounds kept version of the column vector ``x``
    """

    y = (x - l_bound) / (u_bound - l_bound)
    # Local storage to prevent double calls
    floor_y = np.floor(y)
    I = np.mod(floor_y, 2) == 0
    yprime = np.zeros(np.shape(y))
    yprime[I] = np.abs(y[I] - floor_y[I])
    yprime[~I] = 1.0 - np.abs(y[~I] - floor_y[~I])

    x = l_bound + (u_bound - l_bound) * yprime
    return x


def _correct_bounds(x, ub, lb):
    out_of_bounds = np.logical_or(x > ub, x < lb)
    y = (x[out_of_bounds] - lb) / (ub - lb)
    x[out_of_bounds] = lb + (
        ub - lb) * (1. - np.abs(y - np.floor(y)))
    return x


def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        print("Time elapsed", time() - start)
        return res
    return inner


def ert(evals, budget):
    try:
        evals = np.array(evals)
        _ert = evals.sum() / (evals < budget).sum()
    except:
        _ert = float('inf')
    return _ert, np.std(evals)


@timeit
def evaluate(ffid, d, optimizer_class, *args, iterations=50, label='', logging=False, all_funcs=False, **kwargs):
    evals, fopts = np.array([]), np.array([])
    if logging:
        label = 'D{}_{}_{}'.format(
            d, label, datetime.now().strftime("%m"))
        fitness_func = fgeneric.LoggingFunction(
            "/home/jacob/Code/thesis/data/{}".format(label), label)
    for i in range(iterations):
        func, target = bbobbenchmarks.instantiate(ffid, iinstance=1)
        if i == 0:
            print("Optimizing function {} in {}D for target {} + {}".format(ffid, d, target,
                                                                            DISTANCE_TO_TARGET[ffid - 1]))
        if not logging:
            fitness_func = func
        else:
            target = fitness_func.setfun(
                *(func, target)
            ).ftarget
        optimizer = optimizer_class(fitness_func, target, d, *args,
                                    rtol=DISTANCE_TO_TARGET[ffid - 1],
                                    ** kwargs)
        optimizer.run()
        evals = np.append(evals, optimizer.parameters.used_budget)
        fopts = np.append(fopts, optimizer.parameters.fopt)

    print("FCE:\t{:10.8f}\t{:10.4f}\nERT:\t{:10.4f}\t{:10.4f}".format(
        np.mean(fopts), np.std(fopts), *ert(evals, optimizer.parameters.budget)
    ))
    return evals, fopts
