import numpy as np
from .configurablecmaes import ConfigurableCMAES


class AskTellCMAES(ConfigurableCMAES):
    def __init__(self, *args, **kwargs) -> None:
        '''Initialization function.
        Fills the super init's first argument with an empty callable, in order
        to resolve parameter constraints.
        '''
        super().__init__(lambda x: None, *args, ** kwargs)
        self.start_mutation()

    def start_mutation(self) -> None:
        '''Starts the mutation generator, call next once in to instantiate xi & yi'''
        self.mutation_generator = self.get_mutation_generator()
        self.yi, self.xi = next(self.mutation_generator)

    def run(self) -> None:
        '''Method not availble on this interface'''
        raise NotImplemented()

    def step(self) -> None:
        '''Method not availble on this interface'''
        raise NotImplemented()

    def ask(self) -> np.ndarray:
        '''Returns a single new search point'''
        return self.xi

    def tell(self, f: float) -> None:
        '''Updates the state of the CMA-ES with the fitness of the last search point
        Sends the fitness value of the search point (self.xi) back into the 
        mutation generator, and increments the evalutation budget. 
        If the mutation generator is exhausted, selection, recombination and 
        adaptation is performed and the mutation generator is reinitialized.
        '''
        try:
            self.yi, self.xi = self.mutation_generator.send(f)
            self.parameters.used_budget += 1
        except StopIteration:
            self.select()
            self.recombine()
            self.parameters.adapt()
            self.start_mutation()
