import os
import warnings
import typing
from collections import OrderedDict, abc

from inspect import Signature, Parameter, getmodule
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
    '''Data descriptor'''

    def __set_name__(self, owner, name):
        '''Set name attribute '''
        self.name = name

    def __delete__(self, instance):
        '''Delete attribute from the instance __dict__'''
        del instance.__dict__[self.name]


class InstanceOf(Descriptor):
    '''Data descriptor checks for correct types. '''

    def __init__(self, dtype):
        self.dtype = dtype
        self.__doc__ += "Type: {}".format(self.dtype)

    def __set__(self, instance, value):
        if type(value) != type(None):
            if type(value) != self.dtype and not (
                    isinstance(value, np.generic) and type(
                        np.asscalar(value)) == self.dtype)\
                    and str(self.dtype)[1:] != value.__class__.__name__:
                    # we should find another way for the last statement
                raise TypeError("{} should be {}".format(
                    self.name, self.dtype))
            if hasattr(value, '__copy__'):
                value = value.copy()
        instance.__dict__[self.name] = value


class AnyOf(Descriptor):
    '''Descriptor, checks of value is Any of a specified sequence of options. '''

    def __init__(self, options=None):
        self.options = options
        self.__doc__ += "Options: [{}]".format(
            ', '.join(map(str, self.options))
        )

    def __set__(self, instance, value):
        if value not in self.options:
            raise TypeError("{} should .Any of {}".format(
                self.name, self.options
            ))
        instance.__dict__[self.name] = value


class AnnotatedStructMeta(type):
    '''Metaclass for class for AnnotatedStruct.

    Wraps all parameters defined in the class body with 
    __annotations__ into a signature. It additionally wraps each 
    parameter into a descriptor using __annotations__, 
    allowing for type checking. 
    Currently, only two types of descriptors are implementated,
    InstanceOf and typing.AnyOf, the first implements simple type validation,
    the latter implements validation though the use of sequence of
    allowed values. 
    '''

    @classmethod
    def __prepare__(cls: typing.Any, name: str, bases: tuple) -> OrderedDict:
        '''Normally, __prepare__ returns an empty dictionairy,
        now an OrderedDict is returned. This allowes for ordering 
        the parameters (*args). 

        Parameters
        ----------
        cls: typing.Any
            The empty body of the class to be instantiated
        name: str
            The name of the cls
        bases: tuple
            The base classes of the cls 

        Returns
        -------
        OrderedDict        
        '''
        return OrderedDict()

    def __new__(cls: typing.Any, name: str, bases: tuple, attrs: dict) -> typing.Any:
        '''Controls instance creation of classes that have AnnotatedStructMeta as metaclass
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
        '''
        parameters = []
        for key, value in attrs.get('__annotations__', {}).items():
            default_value = attrs.get(key, Parameter.empty)
            if isinstance(default_value, abc.Sequence):
                attrs[key] = AnyOf(default_value)
                parameters.append(Parameter(name=key, default=default_value[0],
                                            kind=Parameter.POSITIONAL_OR_KEYWORD))
            else:
                if not type(value) == type and getmodule(type(value)) != typing:
                    raise TypeError(
                        f"Detected wrong format for annotations of AnnotatedStruct.\n\t"
                        f"Format should be <name>: <type> = <default_value>\n\t"
                        f"Got: type = {value}"
                    )
                attrs[key] = InstanceOf(value)
                parameters.append(Parameter(name=key, default=default_value,
                                            kind=Parameter.POSITIONAL_OR_KEYWORD))

        clsobj = super().__new__(cls, name, bases, attrs)
        setattr(clsobj, '__signature__', Signature(parameters=parameters))
        return clsobj


class AnnotatedStruct(metaclass=AnnotatedStructMeta):
    '''Custom class for defining structs. 

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


def _tpa_mutation(fitness_func: typing.Callable, parameters: "Parameters", x: list, y: list, f: list) -> None:
    '''Helper function for applying the tpa mutation step.
    The code was mostly taken from the ModEA framework, 
    and there a slight differences with the procedure as defined in:
        Nikolaus Hansen. CMA-ES with two-point 
        step-size adaptation.CoRR, abs/0805.0231,2008.
    The function should not be used outside of the ConfigurableCMAES optimizer

    Parameters
    ----------
    fitness_func: typing.Callable
        A fitness function to be optimized
    parameters: Parameters
        A CCMAES Parameters object
    x: list
        A list of new individuals
    y: list
        A list of new mutation vectors
    f: list
        A list of fitnesses
    '''

    yi = ((parameters.m - parameters.m_old) /
          parameters.sigma)
    y.extend([yi, -yi])
    x.extend([
        parameters.m + (parameters.sigma * yi),
        parameters.m + (parameters.sigma * -yi)
    ])
    f.extend(list(map(fitness_func, x)))
    if f[1] < f[0]:
        parameters.rank_tpa = -parameters.a_tpa
    else:
        parameters.rank_tpa = (
            parameters.a_tpa + parameters.b_tpa)


def _scale_with_threshold(z, threshold):
    '''Function for scaling a vector z to have length > threshold

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
    '''

    length = np.linalg.norm(z)
    if length < threshold:
        new_length = threshold + (threshold - length)
        z *= (new_length / length)
    return z


def _correct_bounds(x, ub, lb):
    '''Bound correction function
    Rescales x to fall within the lower lb and upper
    bounds ub specified.

    Parameters 
    ----------
    x: np.ndarray
        vector of which the bounds should be corrected
    ub: float
        upper bound
    ub: float
        lower bound

    Returns
    -------
    np.ndarray
        bound corrected version of x
    '''

    out_of_bounds = np.logical_or(x > ub, x < lb)
    y = (x[out_of_bounds] - lb) / (ub - lb)
    x[out_of_bounds] = lb + (
        ub - lb) * (1. - np.abs(y - np.floor(y)))
    return x


def timeit(func):
    '''Decorator function for timing the excecution of
    a function.

    Parameters
    ----------
    func: typing.Callable
        The function to be timed

    Returns
    -------
    typing.Callable
        a wrapped function
    '''
    @wraps(func)
    def inner(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        print("Time elapsed", time() - start)
        return res
    return inner


def ert(evals, budget):
    '''Computed the expected running time of 
    a list of evaluations.

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
    '''
    if any(evals):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                evals = np.array(evals)
                _ert = evals.sum() / (evals < budget).sum()
            return _ert, np.std(evals)
        except:
            pass
    return float('inf'), 0.


@timeit
def evaluate(
        optimizer_class,
        fid,
        dim,
        iterations=50,
        label='',
        logging=False,
        data_folder=None,
        seed=42,
        **kwargs):
    '''Helper function to evaluate an optimizer on the BBOB test suite. 

    Parameters
    ----------
    optimizer_class: Optimizer
        An instance of Optimizer or a child thereof, i.e. CannonicalCMAES
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
    seed: int = 42 
        The random seed to be used
    **kwargs
        These are directly passed into the instance of optimizer_class,
        in this manner parameters can be specified for the optimizer. 

    Returns
    -------
    list
        The number of evaluations for each run of the optimizer
    fopts
        The best fitness values for each run of the optimizer
    '''

    evals, fopts = np.array([]), np.array([])
    if seed:
        np.random.seed(seed)
    if logging:
        label = 'D{}_{}_{}'.format(
            dim, label, datetime.now().strftime("%B"))
        data_location = os.path.join(data_folder if os.path.isdir(
            data_folder or ''
        ) else os.getcwd(), label)
        fitness_func = fgeneric.LoggingFunction(data_location, label)
    for i in range(iterations):
        func, target = bbobbenchmarks.instantiate(fid, iinstance=1)
        rtol = DISTANCE_TO_TARGET[fid - 1]
        if i == 0:
            print(
                (
                    "{}\nOptimizing function {} in {}D for target {} + {}"
                    " with {} iterations."
                ).format(label, fid, dim, target, rtol, iterations))
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
    '''Sphere function

    Parameters
    ----------
    x: np.ndarray
    fopt: float

    Returns
    -------
    float
    '''

    return (np.linalg.norm(x.flatten()) ** 2) + fopt
