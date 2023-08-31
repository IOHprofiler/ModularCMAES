from typing import Callable

import c_maes.cmaescpp
import c_maes.cmaescpp.options
import numpy

class SequentialSelection:
    def __init__(
        self,
        mirror: c_maes.cmaescpp.options.Mirror,
        mu: int,
        seq_cuttoff_factor: float = ...,
    ) -> None: ...
    def break_conditions(
        self, i: int, f: float, fopt: float, mirror: c_maes.cmaescpp.options.Mirror
    ) -> bool: ...

class SigmaSampler:
    beta: float
    def __init__(self, dimension: float) -> None: ...
    def sample(self, sigma: float, population: c_maes.cmaescpp.Population) -> None: ...

class ThresholdConvergence:
    def __init__(self) -> None: ...
    def scale(self, z: numpy.ndarray[numpy.float64[m, n]], stats, bounds) -> None: ...

class NoThresholdConvergence(ThresholdConvergence):
    def __init__(self) -> None: ...

class Strategy:
    def adapt(
        self,
        weights,
        dynamic,
        population: c_maes.cmaescpp.Population,
        old_population: c_maes.cmaescpp.Population,
        stats,
        strategy,
    ) -> None: ...
    def sample_sigma(self, population: c_maes.cmaescpp.Population) -> None: ...
    @property
    def cs(self) -> float: ...
    @property
    def s(self) -> float: ...
    @property
    def sequential_selection(self) -> SequentialSelection: ...
    @property
    def sigma(self) -> float: ...
    @property
    def sigma0(self) -> float: ...
    @property
    def sigma_sampler(self) -> SigmaSampler: ...
    @property
    def threshold_convergence(self) -> ThresholdConvergence: ...

class CSA(Strategy):
    def __init__(
        self,
        threshold_convergence: ThresholdConvergence,
        sequential_selection: SequentialSelection,
        sigma_sampler: SigmaSampler,
        cs: float,
        damps: float,
        sigma0: float,
    ) -> None: ...
    def adapt_sigma(
        self,
        weights,
        dynamic,
        population: c_maes.cmaescpp.Population,
        old_pop: c_maes.cmaescpp.Population,
        stats,
        lamb: int,
    ) -> None: ...
    def mutate(
        self,
        objective: Callable[[numpy.ndarray[numpy.float64[m, 1]]], float],
        n_offspring: int,
        parameters,
    ) -> None: ...
    @property
    def damps(self) -> float: ...

class LPXNES(CSA):
    def __init__(
        self,
        threshold_convergence: ThresholdConvergence,
        sequential_selection: SequentialSelection,
        sigma_sampler: SigmaSampler,
        cs: float,
        damps: float,
        sigma0: float,
    ) -> None: ...

class MSR(CSA):
    def __init__(
        self,
        threshold_convergence: ThresholdConvergence,
        sequential_selection: SequentialSelection,
        sigma_sampler: SigmaSampler,
        cs: float,
        damps: float,
        sigma0: float,
    ) -> None: ...

class MXNES(CSA):
    def __init__(
        self,
        threshold_convergence: ThresholdConvergence,
        sequential_selection: SequentialSelection,
        sigma_sampler: SigmaSampler,
        cs: float,
        damps: float,
        sigma0: float,
    ) -> None: ...

class NoSequentialSelection(SequentialSelection):
    def __init__(
        self,
        mirror: c_maes.cmaescpp.options.Mirror,
        mu: int,
        seq_cuttoff_factor: float = ...,
    ) -> None: ...

class NoSigmaSampler(SigmaSampler):
    def __init__(self, dimension: float) -> None: ...

class PSR(CSA):
    success_ratio: float
    def __init__(
        self,
        threshold_convergence: ThresholdConvergence,
        sequential_selection: SequentialSelection,
        sigma_sampler: SigmaSampler,
        cs: float,
        damps: float,
        sigma0: float,
    ) -> None: ...

class TPA(CSA):
    a_tpa: float
    b_tpa: float
    rank_tpa: float
    def __init__(
        self,
        threshold_convergence: ThresholdConvergence,
        sequential_selection: SequentialSelection,
        sigma_sampler: SigmaSampler,
        cs: float,
        damps: float,
        sigma0: float,
    ) -> None: ...

class XNES(CSA):
    def __init__(
        self,
        threshold_convergence: ThresholdConvergence,
        sequential_selection: SequentialSelection,
        sigma_sampler: SigmaSampler,
        cs: float,
        damps: float,
        sigma0: float,
    ) -> None: ...
