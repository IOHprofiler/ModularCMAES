from abc import ABCMeta, abstractproperty, abstractmethod
from dataclasses import dataclass
from collections import Counter

from typing import Optional
import pulp
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util


tfd = tfp.distributions
tfb = tfp.bijectors
psd_kernels = tfp.math.psd_kernels

tf_zero = tf.constant(0., tf.float64)
tf_hundredths = tf.constant(0.01, tf.float64)
tf_one = tf.constant(1., tf.float64)
tf_two = tf.constant(2., tf.float64)


class GP_kernel_base_interface(metaclass=ABCMeta):
    ''' basic interface for all kernels '''

    def __init__(self, settings, *args, **kwargs):
        self.settings = settings

    @abstractmethod
    def kernel(self) -> psd_kernels.AutoCompositeTensorPsdKernel:
        pass

    @abstractproperty
    def dof(self) -> int:
        return 0

    @abstractmethod
    def get_variables(self):
        return []


class _BasicCombinedKernel(GP_kernel_base_interface):
    _cls_first: GP_kernel_base_interface = None
    _cls_second: GP_kernel_base_interface = None
    _uid = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obj_first = self._cls_first(*args, **kwargs)
        self._obj_second = self._cls_second(*args, **kwargs)

    @property
    def dof(self) -> int:
        return self._obj_first.dof + self._obj_second.dof

    def get_variables(self):
        return self._obj_first.get_variables() + self._obj_second.get_variables()

    @abstractmethod
    def kernel(self) -> psd_kernels.AutoCompositeTensorPsdKernel:
        pass

class GP_kernel_meta(ABCMeta):
    ''' adds basic operations to all kernel classes '''

    def __add__(cls, other):
        new_uid = tuple(sorted(cls._uid + other._uid))

        ''' adds addition of kernel classes '''
        class SumKernel(_BasicCombinedKernel, metaclass=GP_kernel_meta):
            _cls_first = cls
            _cls_second = other
            _uid = new_uid

            def kernel(self):
                return self._obj_first.kernel() + self._obj_second.kernel()

        SumKernel.__name__ = cls.__name__ + '_a_' + other.__name__
        SumKernel.__qualname__ = cls.__qualname__ + '_a_' + other.__qualname__
        return SumKernel

    def __mul__(cls, other):
        new_uid = []
        for e1 in other._uid:
            assert isinstance(e1, tuple)
            for e2 in cls._uid:
                assert isinstance(e2, tuple)
                new_uid.append(tuple(sorted(e1 + e2)))
        new_uid = tuple(sorted(new_uid))

        ''' adds multiplication of kernel classes '''
        class MulKernel(_BasicCombinedKernel, metaclass=GP_kernel_meta):
            _cls_first = cls
            _cls_second = other
            _uid = new_uid

            def kernel(self):
                return self._obj_first.kernel() * self._obj_second.kernel()

        MulKernel.__name__ = cls.__name__ + '_m_' + other.__name__
        MulKernel.__qualname__ = cls.__qualname__ + '_m_' + other.__qualname__
        return MulKernel


class GP_kernel_concrete_base(GP_kernel_base_interface, metaclass=GP_kernel_meta):
    NOISE_PRETRANSFORMED_STDDEV = 0.01
    KERNEL_CLS: psd_kernels.AutoCompositeTensorPsdKernel = None

    BASIS: tuple = ()
    CONSTANTS: dict = {}

    VAR_DEFAULT: dict = {}
    VAR_BIJECTORS: dict = {}

    _uid = tuple()

    def __init__(self, *args, **kwargs):
        '''
            creates transformed variables and initializes them
        '''
        super().__init__(*args, **kwargs)
        self._create_variables()
        self._create_basis(*args, **kwargs)

    @property
    def dof(self):
        dof = 0
        # parameters
        for p in self.variables.values():
            dof += int(tf.reduce_prod(p.shape))
        # basis
        for b in self.basis:
            dof += b.dof
        return dof

    def kernel(self):
        ''' returns tf kernel '''
        kernels = [b.kernel() for b in self.basis]
        return self.KERNEL_CLS(
            *kernels,
            **self.variables,
            **self.CONSTANTS
        )

    def _create_basis(self, *args, **kwargs):
        self.basis = []
        for base in self.BASIS:
            b = base(*args, **kwargs)
            self.basis.append(b)

    def _create_variables(self):
        ''' creates variables '''
        d = self.settings.d
        self.variables = {}
        for vname in self.VAR_DEFAULT:
            initial_value = self.VAR_DEFAULT[vname]
            if callable(initial_value):
                initial_value = initial_value(d)

            p = tfp.util.TransformedVariable(
                initial_value=initial_value,
                bijector=self.VAR_BIJECTORS[vname](),
                dtype=tf.float64,
                trainable=True,
                name=vname,
            )
            noise = tf.random.normal(tf.shape(p.pretransformed_input),
                                     stddev=self.NOISE_PRETRANSFORMED_STDDEV,
                                     dtype=tf.float64)
            p.pretransformed_input.assign_add(noise)
            self.variables[vname] = p

    def get_variables(self):
        return list(self.variables.items())

    @staticmethod
    def create_class(kernel_params, basis=None):
        if basis is None:
            basis = tuple()
        new_name = kernel_params.kernel.__name__
        if kernel_params.name is not None:
            new_name = kernel_params.name

        class GP_kernel_concrete(GP_kernel_concrete_base):
            KERNEL_CLS = kernel_params.kernel
            BASIS = basis
            VAR_DEFAULT = kernel_params.defaults
            VAR_BIJECTORS = kernel_params.bijectors
            CONSTANTS = kernel_params.constants
            _uid = ((new_name,),)

        GP_kernel_concrete.__name__ = new_name
        GP_kernel_concrete.__qualname__ = new_name
        return GP_kernel_concrete

    @staticmethod
    def create_functor(kernel_params):
        def output(*basis):
            assert(kernel_params.arity == len(basis) or kernel_params.arity < 0)
            output = GP_kernel_concrete_base.create_class(kernel_params, basis=basis)

            new_name = kernel_params.name + '___' + \
                "__".join((b.__name__ for b in basis)) + '___'
            output.__name__ = new_name
            output.__qualname__ = new_name
            return output
        output.__name__ = kernel_params.name
        return output


@dataclass
class KernelGroup:
    kernels: tuple
    defaults: Optional[dict] = None
    bijectors: Optional[dict] = None
    arity: int = 0


@dataclass
class Kernel:
    kernel: psd_kernels.AutoCompositeTensorPsdKernel = None
    name: Optional[str] = None
    defaults: Optional[dict] = None
    constants: Optional[dict] = None
    bijectors: Optional[dict] = None
    arity: int = 0

    def __post_init__(self):
        if self.bijectors is None:
            self.bijectors = {}
        if self.constants is None:
            self.constants = {}
        if self.defaults is None:
            self.defaults = {}

        if self.name is None:
            self.name = self.kernel.__name__

        # remove None
        to_remove = set(varname for varname in self.defaults if self.defaults[varname] is None)

        # remove variables that are not set already
        if issubclass(self.kernel, psd_kernels.AutoCompositeTensorPsdKernel):
            all_variables = self.kernel.parameter_properties(tf.float64)
            extra = set(self.defaults).difference(set(all_variables.keys()))
            to_remove = to_remove.union(extra)

        # final removal
        for varname in to_remove:
            del self.defaults[varname]

        # try to fill up bijectors
        if issubclass(self.kernel, psd_kernels.AutoCompositeTensorPsdKernel):
            all_variables = self.kernel.parameter_properties(tf.float64)

            # fill up bijectors automatically
            for varname in set(self.defaults).difference(self.bijectors.keys()):
                try:
                    p_prop = all_variables[varname]
                    self.bijectors[varname] = p_prop.default_constraining_bijector_fn
                except KeyError:
                    raise NotImplementedError(f'Cannot fill up bijector {varname}')


# GENERATION OF KERNELS ...

_combined_kernel_options = [
    KernelGroup(
        kernels=(
            psd_kernels.MaternFiveHalves,
            psd_kernels.MaternOneHalf,
            psd_kernels.MaternThreeHalves,
            psd_kernels.RationalQuadratic,
            psd_kernels.ExponentiatedQuadratic,
        ),
        defaults={
            'amplitude':            tf_one,
            'length_scale':         tf_one,
            'inverse_length_scale': None,
            'scale_mixture_rate':   tf_one,
        },
    ),
    KernelGroup(
        kernels=(
            psd_kernels.FeatureScaled,
            psd_kernels.PointwiseExponential,
        ),
        defaults={
            'kernel': None,
            'scale_diag': lambda d: tf.ones(d, dtype=tf.float64),
            'inverse_scale_diag': None
        },
        bijectors={
            'scale_diag': lambda: tfb.SoftmaxCentered()
        },
        arity=1
    ),
    Kernel(
        kernel=psd_kernels.ExpSinSquared,  # a.k.a. 'periodic kernel'
        defaults={
            'amplitude':            tf_one,
            'length_scale':         tf_one,
            'inverse_length_scale': None,
            'period':               tf_one*100,
        },
    ),
    # KERNELS :
    Kernel(
        kernel=psd_kernels.Parabolic,
        defaults={
            'amplitude':            tf_one,
            'length_scale':         tf_hundredths,
        }
    ),
    Kernel(
        kernel=psd_kernels.ExponentialCurve,
        defaults={
            'concentration': tf_two,
            'rate': tf_two,
        },
        bijectors={
            'concentration':
                (lambda: tfb.softplus.Softplus(low=dtype_util.eps(tf.float64))),
            'rate':
                (lambda: tfb.softplus.Softplus(low=dtype_util.eps(tf.float64))),
        }
    ),
    Kernel(
        kernel=psd_kernels.Constant,
        defaults={'constant': tf_hundredths},
    ),
    Kernel(
        kernel=psd_kernels.Linear,
        defaults={
            'bias_amplitude':  None,
            'slope_amplitude': None,
            'bias_variance':   tf_one,
            'slope_variance':  tf_one,
            'shift':           tf_zero,
        }
    ),
    Kernel(
        kernel=psd_kernels.Polynomial,
        name='Quadratic',
        defaults={
            'bias_amplitude':  None,
            'slope_amplitude': None,
            'bias_variance':   tf_one,
            'slope_variance':  tf_one,
            'shift':           tf_zero,
        },
        constants={
            'exponent':        tf.constant(2., tf.float64),
        }
    ),
    Kernel(
        kernel=psd_kernels.Polynomial,
        name='Cubic',
        defaults={
            'bias_amplitude':  None,
            'slope_amplitude': None,
            'bias_variance':   tf_one,
            'slope_variance':  tf_one,
            'shift':           tf_zero,
        },
        constants={
            'exponent':        tf.constant(3., tf.float64),
        }
    )
    # psd_kernels.ChangePoint,  <smooth interpolation between kernels>
    # problem n-arity

    # psd_kernels.FeatureTransformed, <arbitrary transformation to the input>
    # problem: what transformation ?

    # psd_kernels.KumaraswamyTransformed, <?>
    # problem: feature \in <0,1> requirement

    # psd_kernels.SchurComplement,
    # problem: i don't understand it

    # psd_kernels.SpectralMixture
    # problem: logits & locs & scales is probably different paradigma

    # psd_kernels.GeneralizedMatern, cannot differentiate by df
    # problem: the generalization does not add differentiable component
]

_concrete_kernel_options = []

# 1. convert kernel group to Kernel objects
for element in _combined_kernel_options:
    if isinstance(element, KernelGroup):
        for kernel in element.kernels:
            _concrete_kernel_options.append(Kernel(
                kernel=kernel,
                arity=element.arity,
                defaults=None if element.defaults is None else element.defaults.copy(),
                bijectors=None if element.bijectors is None else element.bijectors.copy(),
            ))
    elif isinstance(element, Kernel):
        _concrete_kernel_options.append(element)


basic_kernels = []
functor_kernels = []

for kernel in _concrete_kernel_options:
    if kernel.arity == 0:
        output = GP_kernel_concrete_base.create_class(kernel)
        basic_kernels.append(output)
    elif kernel.arity == 1:
        output = GP_kernel_concrete_base.create_functor(kernel)
        functor_kernels.append(output)


def kernel_similarity_measure_jaccard(k1, k2) -> float:
    ''' kernel similarity based on modified jaccard distance '''
    k1 = Counter(k1._uid)
    k2 = Counter(k2._uid)

    intersection = k1 & k2
    union = k1 | k2
    return intersection.total() / union.total()


def kernel_similarity_measure_best_matching(k1, k2) -> float:
    ''' kernel similarity measure based on optional matching
    of sub modified jaccard distances '''
    c1 = [Counter(e1) for e1 in k1._uid]
    c2 = [Counter(e2) for e2 in k2._uid]

    score = [[(a & b).total() / (a | b).total()
             for a in c1] for b in c2
             ]

    problem = pulp.LpProblem("Matching kernels", pulp.LpMaximize)
    # variables
    y = pulp.LpVariable.dicts("pair", [(i, j)
                                       for i in range(len(k1._uid))
                                       for j in range(len(k2._uid))
                                       ], cat='Binary'
                              )
    # objective
    problem += pulp.lpSum([
        score[j][i] * y[(i, j)]
        for i in range(len(k1._uid)) for j in range(len(k2._uid))])
    # constraints
    for i in range(len(k1._uid)):
        problem += pulp.lpSum(y[(i, j)] for j in range(len(k2._uid))) <= 1
    for j in range(len(k2._uid)):
        problem += pulp.lpSum(y[(i, j)] for i in range(len(k1._uid))) <= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    if problem.sol_status != 1:
        return 0.

    obj_value = problem.objective.value()
    if isinstance(obj_value, float):
        return obj_value / (len(k1._uid) + len(k2._uid) - obj_value)
    return 0.

