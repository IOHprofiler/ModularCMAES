from typing import List, Callable
import numpy as np


class Optimizer:
    '''Abstract class for optimizer objects '''
    parameters: "Parameters"
    _fitness_func: Callable

    def run(self):
        '''Runs the step method until step method retuns a falsy value

        Returns
        -------
        Optimizer
        '''
        while self.step():
            pass
        return self

    @property
    def break_conditions(self) -> List[bool]:
        '''Returns a list with break conditions based on the
        interal state (parameters) of the optimization algorithm.

        Returns
        -------
        [bool, bool]
        '''
        return [
            self.parameters.target >= self.parameters.fopt,
            self.parameters.used_budget >= self.parameters.budget
        ]

    def fitness_func(self, x: np.ndarray) -> float:
        '''Wrapper function for calling self._fitness_func
        adds 1 to self.parameters.used_budget for each fitnes function
        call.

        Parameters
        ----------
        x: np.ndarray
            array on which to call the objective/fitness function

        Returns
        -------        
        float
        '''
        self.parameters.used_budget += 1
        return self._fitness_func(x.flatten())

    def step(self):
        ''' Abstract method for calling one iteration 
        of the optimization procedure.  

        Raises
        ------
        NotImplementedError
        '''
        raise NotImplementedError()
