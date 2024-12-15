import modcma.c_maes.cmaescpp.options
import numpy

class Modules:
    active: bool
    bound_correction: modcma.c_maes.cmaescpp.options.CorrectionMethod
    center_placement: modcma.c_maes.cmaescpp.options.CenterPlacement
    elitist: bool
    matrix_adaptation: modcma.c_maes.cmaescpp.options.MatrixAdaptationType
    mirrored: modcma.c_maes.cmaescpp.options.Mirror
    orthogonal: bool
    repelling_restart: bool
    restart_strategy: modcma.c_maes.cmaescpp.options.RestartStrategy
    sample_sigma: bool
    sampler: modcma.c_maes.cmaescpp.options.BaseSampler
    sequential_selection: bool
    ssa: modcma.c_maes.cmaescpp.options.StepSizeAdaptation
    threshold_convergence: bool
    weights: modcma.c_maes.cmaescpp.options.RecombinationWeights
    sample_transformation: modcma.c_maes.cmaescpp.options.SampleTranformerType
    def __init__(self) -> None: ...

class Settings:
    budget: int
    c1: float | None
    cc: float | None
    cmu: float | None
    cs: float | None
    dim: int
    lambda0: int
    lb: numpy.ndarray
    max_generations: int | None
    modules: Modules
    mu0: int
    sigma0: float
    target: float | None
    ub: numpy.ndarray
    verbose: bool
    volume: float
    x0: numpy.ndarray | None
    def __init__(
        self,
        dim: int,
        modules: Modules | None = ...,
        target: float | None = ...,
        max_generations: int | None = ...,
        budget: int | None = ...,
        sigma0: float | None = ...,
        lambda0: int | None = ...,
        mu0: int | None = ...,
        x0: numpy.ndarray | None = ...,
        lb: numpy.ndarray | None = ...,
        ub: numpy.ndarray | None = ...,
        cs: float | None = ...,
        cc: float | None = ...,
        cmu: float | None = ...,
        c1: float | None = ...,
        verbose: bool = ...,
    ) -> None: ...

class Solution:
    e: int
    t: int
    x: numpy.ndarray
    y: float
    def __init__(self) -> None: ...

class Stats:
    centers: list[Solution]
    current_avg: float
    current_best: Solution
    evaluations: int
    global_best: Solution
    solutions: list[Solution]
    t: int
    def __init__(self) -> None: ...

class Weights:
    c1: float
    cc: float
    cmu: float
    mueff: float
    mueff_neg: float
    negative: numpy.ndarray
    positive: numpy.ndarray
    weights: numpy.ndarray
    def __init__(self, dimension: int, mu0: int, lambda0: int, modules) -> None: ...
