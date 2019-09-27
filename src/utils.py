from collections import OrderedDict, abc
from typing import Callable
from inspect import Signature, Parameter
from datetime import datetime
from functools import wraps
from time import time
import numpy as np

from .bbob import bbobbenchmarks, fgeneric

DISTANCE_TO_TARGET = [pow(10, p) for p in [
    -8.,  # 1
    -8.,  # 2
    .4,  # 3
    .8,  # 4
    -8.,  # 5
    -8.,  # 6
    .0,  # 7
    -8.,  # 8
    -8.,  # 9
    -8.,  # 10
    -8.,  # 11
    -8.,  # 12
    -8.,  # 13
    -8.,  # 14
    .4,  # 15
    -2.,  # 16
    -4.4,  # 17
    -4.0,  # 18
    -.6,  # 19
    .2,  # 20
    -.6,  # 21
    .0,  # 22
    -.8,  # 23
    1.0,  # 24
]]


class Descriptor:
    '''Basic data descriptor'''

    def __set_name__(self, owner, name):
        '''Set name ne
        '''
        self.name = name

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class InstanceOf(Descriptor):
    def __init__(self, dtype):
        self.dtype = dtype

    def __set__(self, instance, value):
        if type(value) != type(None):
            if type(value) != self.dtype and not (
                isinstance(value, np.generic) and type(
                    np.asscalar(value)) == self.dtype):
                raise TypeError("{} should be {}".format(
                    self.name, self.dtype))
            if hasattr(value, '__copy__'):
                value = value.copy()
        instance.__dict__[self.name] = value

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
        instance.__dict__[self.name] = value

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
            if isinstance(default_value, abc.Sequence):
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
    '''
    __signature__: Signature

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
    ''''''
    length = np.linalg.norm(z)
    if length < threshold:
        new_length = threshold + (threshold - length)
        z *= (new_length / length)
    return z


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
def evaluate(optimizer_class, fid, dim, iterations=50, label='', logging=False, all_funcs=False, seed=42, **kwargs):
    evals, fopts = np.array([]), np.array([])
    np.random.seed(seed)
    if logging:
        label = 'D{}_{}_{}'.format(
            dim, label, datetime.now().strftime("%m"))
        fitness_func = fgeneric.LoggingFunction(
            "/home/jacob/Code/thesis/data/{}".format(label), label)
    for i in range(iterations):
        func, target = bbobbenchmarks.instantiate(fid, iinstance=1)
        rtol = DISTANCE_TO_TARGET[fid - 1]
        if i == 0:
            print(
                "Optimizing function {} in {}D for target {} + {}".format(fid, dim, target, rtol))
        if not logging:
            fitness_func = func
        else:
            target = fitness_func.setfun(
                *(func, target)
            ).ftarget
        optimizer = optimizer_class(
            fitness_func, dim, target, rtol, **kwargs).run()
        evals = np.append(evals, optimizer.parameters.used_budget)
        fopts = np.append(fopts, optimizer.parameters.fopt)

    print("FCE:\t{:10.8f}\t{:10.4f}\nERT:\t{:10.4f}\t{:10.4f}".format(
        np.mean(fopts), np.std(fopts), *ert(evals, optimizer.parameters.budget)
    ))
    return evals, fopts


def sphere_function(x, fopt=79.48):
    '''Sphere function'''
    return (np.linalg.norm(x.flatten()) ** 2) + fopt
