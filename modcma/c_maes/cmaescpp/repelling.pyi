import modcma.c_maes.cmaescpp as cma
import numpy

class TabooPoint:
    delta: float
    shrinkage: float
    n_rep: int
    solution: cma.Solution
    
    def __init__(self, solution: cma.Solution, delta: float) -> None: ...
    def rejects(self, xi: numpy.ndarray, p: cma.Parameters, attempts: int) -> bool: ...
    def shares_basin(self, xi: numpy.ndarray, p: cma.Parameters) -> bool: ...
    

class Repelling:
    archive: list[TabooPoint]
    attempts: int
    
    def __init__(self) -> None: ...
    def is_rejected(self, xi: numpy.ndarray, p: cma.Parameters) -> bool: ...
    def update_archive(self, p: cma.Parameters) -> bool: ...


class NoRepelling(Repelling):
    ...