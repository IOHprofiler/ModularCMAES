from collections.abc import Callable
from typing import Annotated

import numpy
from numpy.typing import NDArray

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.matrix_adaptation
import modcma.c_maes.cmaescpp.options
import modcma.c_maes.cmaescpp.parameters


class ThresholdConvergence:
    def __init__(self) -> None: ...

    @property
    def init_threshold(self) -> float: ...

    @init_threshold.setter
    def init_threshold(self, arg: float, /) -> None: ...

    @property
    def decay_factor(self) -> float: ...

    @decay_factor.setter
    def decay_factor(self, arg: float, /) -> None: ...

    def scale(self, population: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')], diameter: float, budget: int, evaluations: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

class NoThresholdConvergence(ThresholdConvergence):
    def __init__(self) -> None: ...

class SequentialSelection:
    def __init__(self, mirror: modcma.c_maes.cmaescpp.options.Mirror, seq_cuttoff_factor: float = 1.0) -> None: ...

    def break_conditions(self, i: int, f: float, fopt: float, mirror: modcma.c_maes.cmaescpp.options.Mirror, mu: int) -> bool: ...

class NoSequentialSelection(SequentialSelection):
    def __init__(self, mirror: modcma.c_maes.cmaescpp.options.Mirror, seq_cuttoff_factor: float = 1.0) -> None: ...

class SigmaSampler:
    def __init__(self, dimension: float) -> None: ...

    def apply_integer_bounds(self, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

    def sample(self, sigma: float, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

class NoSigmaSampler(SigmaSampler):
    def __init__(self, dimension: float) -> None: ...

class Strategy:
    def __init__(self, threshold_convergence: ThresholdConvergence, sequential_selection: SequentialSelection, sigma_sampler: SigmaSampler, sigma0: float) -> None: ...

    def adapt(self, weights: modcma.c_maes.cmaescpp.parameters.Weights, dynamic: modcma.c_maes.cmaescpp.matrix_adaptation.Adaptation, population: modcma.c_maes.cmaescpp.Population, old_population: modcma.c_maes.cmaescpp.Population, stats: modcma.c_maes.cmaescpp.parameters.Stats, lamb: int) -> None: ...

    def mutate(self, objective: Callable[[Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]], float], n_offspring: int, parameters: modcma.c_maes.cmaescpp.Parameters) -> None: ...

    @property
    def threshold_convergence(self) -> ThresholdConvergence: ...

    @threshold_convergence.setter
    def threshold_convergence(self, arg: ThresholdConvergence, /) -> None: ...

    @property
    def sequential_selection(self) -> SequentialSelection: ...

    @sequential_selection.setter
    def sequential_selection(self, arg: SequentialSelection, /) -> None: ...

    @property
    def sigma_sampler(self) -> SigmaSampler: ...

    @sigma_sampler.setter
    def sigma_sampler(self, arg: SigmaSampler, /) -> None: ...

    @property
    def sigma(self) -> float: ...

    @sigma.setter
    def sigma(self, arg: float, /) -> None: ...

    @property
    def s(self) -> float: ...

    @s.setter
    def s(self, arg: float, /) -> None: ...

class CSA(Strategy):
    pass

class TPA(Strategy):
    @property
    def a_tpa(self) -> float: ...

    @a_tpa.setter
    def a_tpa(self, arg: float, /) -> None: ...

    @property
    def b_tpa(self) -> float: ...

    @b_tpa.setter
    def b_tpa(self, arg: float, /) -> None: ...

    @property
    def rank_tpa(self) -> float: ...

    @rank_tpa.setter
    def rank_tpa(self, arg: float, /) -> None: ...

class MSR(Strategy):
    pass

class PSR(Strategy):
    @property
    def success_ratio(self) -> float: ...

    @success_ratio.setter
    def success_ratio(self, arg: float, /) -> None: ...

class XNES(Strategy):
    pass

class MXNES(Strategy):
    pass

class LPXNES(Strategy):
    pass

class SR(Strategy):
    pass

class SA(Strategy):
    pass
