import numpy as np
from .configurablecmaes import ConfigurableCMAES
from .parameters import Parameters
from .population import Population
from .utils import _correct_bounds, _scale_with_threshold, _tpa_mutation


class AskTellCMAES(ConfigurableCMAES):
    def mutate(self):
        '''Method performing mutation and evaluation of a set of individuals. 
        Collects the output of the mutation generator.
        '''
        y, x, f = [], [], []
        n_offspring = self.parameters.lambda_
        if self.parameters.step_size_adaptation == 'tpa' and self.parameters.old_population:
            n_offspring -= 2
            _tpa_mutation(self.fitness_func, self.parameters, x, y, f)

        for i in range(1, n_offspring + 1):
            
            zi = next(self.parameters.sampler)
            if self.parameters.threshold_convergence:
                zi = _scale_with_threshold(zi, self.parameters.threshold)

            yi = np.dot(self.parameters.B, self.parameters.D * zi)
            xi = self.parameters.m + (self.parameters.sigma * yi)
            xi, corrected = _correct_bounds(xi, self.parameters.ub, self.parameters.lb, self.parameters.bound_correction)
            self.parameters.corrections += corrected
                
            [a.append(v) for a, v in ((y, yi), (x, xi),)]
        return x, y
                
    def ask_lambda(self):
        x, y = self.mutate()
        self.x_asked = x
        self.y_asked = y
        return x
    
    def tell_lambda(self, X, fitness):
        if X != self.x_asked:
            return
        self.parameters.population = Population(
            np.hstack(self.x_asked),
            np.hstack(self.y_asked),
            np.array(fitness))
        self.parameters.used_budget += len(fitness)
        self.select()
        self.recombine()
        self.parameters.adapt()



if __name__ == "__main__":
    