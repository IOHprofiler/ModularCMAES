import warnings
from collections import deque

import numpy as np
from . import cmaescpp


class AskTellCMAES:
    def __init__(
        self,
        dimension: int = None,
        settings: cmaescpp.parameters.Settings = None,
        modules: cmaescpp.parameters.Modules = None,
        parameters: cmaescpp.Parameters = None,
    ):
        if {settings, dimension, parameters} == {None}:
            raise TypeError(
                "Either dimension or settings or parametrs should be passed"
            )

        if parameters is not None:
            self.cma = cmaescpp.ModularCMAES(parameters)
        elif settings is not None:
            self.cma = cmaescpp.ModularCMAES(settings)
        else:
            settings = cmaescpp.parameters.Settings(dimension, modules)
            self.cma = cmaescpp.ModularCMAES(settings)

        self.ask_queue = deque()

    @property
    def is_ask_queue_empty(self):
        return len(self.ask_queue) == 0

    def register_individual(self, x: np.ndarray) -> float:
        self.ask_queue.append(x.reshape(-1, 1))
        return float("nan")

    def ask(self) -> np.ndarray:
        """Retrieve the next indivual from the ask_queue.

        If the ask_queue is empty mutate is called in order to fill it.

        Returns
        -------
        np.ndarray

        """
        if self.cma.break_conditions():
            raise StopIteration("Break conditions reached, ignoring call to: ask")

        if self.is_ask_queue_empty:
            self.cma.mutate(self.register_individual)
        return self.ask_queue.popleft()

    def tell(self, xi: np.ndarray, fi: float):
        """Process a provided fitness value fi for a given individual xi.

        Parameters
        ----------
        xi: np.ndarray
            An individual previously returned by ask()
        fi: float
            The fitness value for xi
        Raises
        ------
        RuntimeError
            When ask() is not called before tell()
        ValueError
            When an unknown xi is provided to the method

        Warns
        -----
        UserWarning
            When the same xi is provided more than once

        """

        if self.cma.break_conditions():
            raise StopIteration("Break conditions reached, ignoring call to: tell ")

        if self.is_ask_queue_empty:
            pass
        
        indices, *_ = np.where((self.cma.p.pop.X == xi).all(axis=0))
        if len(indices) == 0:
            breakpoint()
            raise ValueError("Unkown xi provided")
        
        f_copy = self.cma.p.pop.f.copy()
        for index in indices:
            if np.isnan(self.cma.p.pop.f[index]):
                f_copy[index] = fi
                break
            else:
                warnings.warn("Repeated call to tell with same xi", UserWarning)
                f_copy[index] = fi
                
        self.cma.p.pop.f = f_copy
        
        if self.is_ask_queue_empty and not np.isnan(f_copy).any():
            self.cma.select()
            self.cma.recombine()
            
            self.cma.adapt(self.register_individual) # this needs 
            self.cma.mutate(self.register_individual)
            
        # breakpoint()

