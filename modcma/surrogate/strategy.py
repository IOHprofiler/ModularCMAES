
import numpy as np

from abc import ABCMeta, abstractmethod

from modcma.parameters import Parameters

from ..typing_utils import XType, YType, yType
import .data
import .model

from ..modularcmaes import ModularCMAES


class SurrogateStrategyBase(metaclass=ABCMeta):
    def __init__(self, modcma: ModularCMAES):
        self.parameters: Parameters = modcma.parameters
        if self.parameters.sequential:
            raise NotImplementedError("Cannot use surrogate model with sequential selection")

        self.data = data.SurrogateData_V1(self.parameters)

        self._model_class = model.str_to_model(self.parameters.surrogate_model)

    @property
    def model(self):
        ''' gets the model and trains it '''
        if self.data.X is None or self.data.F is None:
            return None
        model = self._model_class(self.parameters)
        model.fit(self.data.X, self.data.F)
        return model

    def fitness_func(self, x):
        ''' evaluate one sample using true objective function & saves the result in the archive '''
        f = self.modcma.fitness_func(x)
        self.data.push(x, f)
        return f

    @abstractmethod
    def __call__(self, X: XType) -> YType:
        ''' evaluates all samples using true objective function '''
        F = np.empty(len(X), yType)
        for i in range(len(X)):
            F[i] = self.fitness_func(X[i])
        return F


class Unsure_Strategy(SurrogateStrategyBase):
    StrategyName = 'Unsure'

    def __call__(self, X: XType) -> YType:
        return super().__call__(X)


class Random_Strategy(SurrogateStrategyBase):
    StrategyName = 'Random'

    def __init__(self, modcma: ModularCMAES):
        super().__init__(modcma)
        self.trueEval: float = self.parameters.surrogate_strategy_random_trueEval

    def __call__(self, X: XType) -> YType:
        sample = np.random.rand(len(X), 1) <= self.trueEval

        Xtrue = X[sample]
        Ftrue = super().__call__(Xtrue)

        Xfalse = X[np.logical_not(sample)]
        Ffalse = self.model.predict(Xfalse)

        F = np.empty(shape=(len(X), 1), dtype=yType)
        F[sample] = Ftrue
        F[np.logical_not(sample)] = Ffalse

        return F


class Kendall_Strategy(SurrogateStrategyBase):
    StrategyName = 'Kendall'

    def __init__(self, modcma: ModularCMAES):
        super().__init__(modcma)

    def __call__(self, X: XType) -> YType:
        F = np.tile(np.nan, reps=(len(X), 1))
        to_evaluate = max(
            self.parameters.surrogate_strategy_kendall_minimum_trueEval_absolute,
            int(self.parameters.surrogate_strategy_kendall_minimum_trueEval_relative * len(X))
        )

        while np.any(np.isnan(F)):
            pass # <-- eval

            to_evaluate += max(1,
                int((self.parameters.surrogate_strategy_kendall_multiplicative_increase_relative - 1.) *
                    to_evaluate)
            )

        self.model 
        return super().__call__(X)

    def _eval_sequence(self, number, X):
        """evaluate unevaluated entries until `number` of entries are evaluated *overall*.
        """
        F_model = self.model(X)

        for i in np.argsort(F_model):
            if self.evaluations >= number:
                break
            if not self.evaluated[i]:
                self.fvalues[i] = self.fitness_function(self.X[i])
                self.evaluated[i] = True

        assert self.evaluations == number or \
            self.evaluations == len(self.X) < number








'''
class SurrogateModelBase(metaclass=ABCMeta):
    def on_population_size_change(self, new_size) -> None:
        pass

    def sort(self, top=None) -> None:
        pass

    @abstractproperty
    def true_y(self) -> YType:
        return np.array(np.nan)

    @abstractproperty
    def true_x(self) -> XType:
        return np.array(np.nan)

    def __call__(self, X: XType) -> YType:
        return np.array(np.nan)


class PopulationRepresentation:
    """Manage incremental evaluation of a population of solutions.
    Evaluate solutions, add them to the model and keep track of which
    solutions were evaluated.
    """

    def __init__(self,
                 X: XType,
                 settings: LQSurrogateStrategySettings,
                 surrogate_model: SurrogateModelBase,
                 fitness_function: Callable
                 ):
        """all is based on the population (list of solutions) `X`"""
        self.X = X

        self.evaluated = np.zeros(len(X), dtype=np.bool_)
        self.fvalues = np.repeat(np.nan, len(X))
        self.surrogate: SurrogateModelBase = surrogate_model
        self.settings: LQSurrogateStrategySettings = settings
        self.fitness_function: Callable = fitness_function

    def _eval_sequence(self, number, X):
        """evaluate unevaluated entries until `number` of entries are evaluated *overall*.
        """
        F_model = self.surrogate(X)

        for i in np.argsort(F_model):
            if self.evaluations >= number:
                break
            if not self.evaluated[i]:
                self.fvalues[i] = self.fitness_function(self.X[i])
                self.evaluated[i] = True

        assert self.evaluations == number or \
            self.evaluations == len(self.X) < number

    def surrogate_values(self, true_values_if_all_available=True) -> YType:
        """return surrogate values """

        if true_values_if_all_available and self.evaluations == len(self.X):
            return self.fvalues

        F_model: YType = self.surrogate(self.X)
        m_offset = np.nanmin(F_model)
        f_offset = np.nanmin(self.fvalues)

        if np.isfinite(f_offset):
            return F_model - m_offset + f_offset
        else:
            return F_model

    @property
    def evaluations(self):
        return sum(self.evaluated)

    def __call__(self, X: XType) -> YType:
        """return population f-values.
        Evaluate at least one solution on the true fitness.
        The smallest returned value is never smaller than the smallest truly
        evaluated value.
        """

        number_evaluated = self.settings.number_of_evaluated

        while len(X) - sum(self.evaluated) > 0:  # all evaluated ?
            self._eval_sequence(number_evaluated, X)
            self.surrogate.sort(top=number_evaluated)


            tau, _ = kendalltau(
                self.surrogate.true_y,
                self.surrogate(self.surrogate.true_x)
            )
            if tau >= self.settings.tau_truth_threshold:
                break

            number_evaluated += int(np.ceil(
                number_evaluated
                * self.settings.increase_of_number_of_evaluated
            ))

        # popsi = len(x)
        # nevaluated = self.evaluations
        n_for_tau = lambda popsi, nevaluated:
            int(max((15, min((1.2 * nevaluated, 0.75 * popsi)))))
            max(
        int(max((15, min((1.2 * nevaluated, 0.75 * popsi)))))


        self.surrogate.sort(top=self.evaluations)
        return self.surrogate_values(
            self.settings.return_true_fitness_if_all_evaluated
        )


        # TODO: Zjistit jestli je kendall stejny
        # TODO: Zjistit jestli je linearni regerese stejna
        # TODO: Nfortau







class LQ_SurrogateStrategy(ModularCMAES):
    pass

'''
