from typing import Callable

import modcma.c_maes.cmaescpp
import modcma.c_maes.cmaescpp.options
import numpy

class CSA(Strategy):
    damps: float
    def __init__(
        self,
        threshold_convergence: ThresholdConvergence,
        sequential_selection: SequentialSelection,
        sigma_sampler: SigmaSampler,
        cs: float,
        damps: float,
        sigma0: float,
    ) -> None: ...
    def mutate(
        self, objective: Callable[[numpy.ndarray], float], n_offspring: int, parameters
    ) -> None: ...

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
        mirror: modcma.c_maes.cmaescpp.options.Mirror,
        mu: int,
        seq_cuttoff_factor: float = ...,
    ) -> None: ...

class NoSigmaSampler(SigmaSampler):
    def __init__(self, dimension: float) -> None: ...

class NoThresholdConvergence(ThresholdConvergence):
    def __init__(self) -> None: ...

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

class SequentialSelection:
    def __init__(
        self,
        mirror: modcma.c_maes.cmaescpp.options.Mirror,
        mu: int,
        seq_cuttoff_factor: float = ...,
    ) -> None: ...
    def break_conditions(
        self,
        i: int,
        f: float,
        fopt: float,
        mirror: modcma.c_maes.cmaescpp.options.Mirror,
    ) -> bool: ...

class SigmaSampler:
    beta: float
    def __init__(self, dimension: float) -> None: ...
    def sample(
        self, sigma: float, population: modcma.c_maes.cmaescpp.Population, beta: float
    ) -> None: ...

class Strategy:
    s: float
    sequential_selection: SequentialSelection
    sigma: float
    sigma_sampler: SigmaSampler
    threshold_convergence: ThresholdConvergence
    def __init__(self, *args, **kwargs) -> None: ...
    def adapt(
        self,
        weights,
        dynamic,
        population: modcma.c_maes.cmaescpp.Population,
        old_population: modcma.c_maes.cmaescpp.Population,
        stats,
        lamb: int,
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

class ThresholdConvergence:
    decay_factor: float
    init_threshold: float
    def __init__(self) -> None: ...
    def scale(
        self,
        population: modcma.c_maes.cmaescpp.Population,
        diameter: float,
        budget: int,
        evaluations: int,
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
