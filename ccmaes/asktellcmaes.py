import numpy as np
from .configurablecmaes import ConfigurableCMAES

class AskTellCMAES(ConfigurableCMAES):
    'Ask tell interface for the ConfigurableCMAES'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(None, *args, **kwargs)
        self.reset_mutation_genator()


    def reset_mutation_genator(self) -> None:
        self.mutation_generator = self.get_mutation_generator()
        self.yi, self.xi = next(self.mutation_generator)

    def ask(self) -> np.ndarray:
        return self.xi        

    def tell(self, xi:np.ndarray, fi: float):
        if xi != self.xi:
            print("")
        try:
            self.yi, self.xi = self.mutation_generator.send(fi)
        except StopIteration:
            self.select()
            self.recombine()
            self.parameters.adapt()
            self.reset_mutation_genator()


    def step(self):
        raise NotImplementedError("Step is undefined in this interface")

    def run(self):
        raise NotImplementedError("Run is undefined in this interface")

    # def ask_lambda(self):
    #     x, y = self.mutate()
    #     self.x_asked = x
    #     self.y_asked = y
    #     return x
    
    # def tell_lambda(self, X, fitness):
    #     if X != self.x_asked:
    #         return
    #     self.parameters.population = Population(
    #         np.hstack(self.x_asked),
    #         np.hstack(self.y_asked),
    #         np.array(fitness))
    #     self.parameters.used_budget += len(fitness)
    #     self.select()
    #     self.recombine()
    #     self.parameters.adapt()



