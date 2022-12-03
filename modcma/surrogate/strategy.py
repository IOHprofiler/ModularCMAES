
import math
import numpy as np

from abc import ABCMeta, abstractmethod

from modcma.parameters import Parameters

from typing import Type, Optional
from ..typing_utils import XType, YType, yType
import .data
import .model

from scipy.stats import kendalltau

from ..modularcmaes import ModularCMAES


class SurrogateStrategyBase(metaclass=ABCMeta):
    StrategyName = "Base"

    def __init__(self, modcma: ModularCMAES):
        self.modcma = modcma
        self.parameters: Parameters = modcma.parameters
        if self.parameters.sequential:
            raise NotImplementedError("Cannot use surrogate model with sequential selection")

        self.data = data.SurrogateData_V1(self.parameters)

        self._model = self.get_model(self.parameters)
        self._load_strategy_parameters()

    def _load_strategy_parameters(self):
        all_settings = self.parameters.__dict__.keys()
        prefix = "surrogate_strategy_" + self.StrategyName + "_"

        def filter_and_take(name, iterable):
            return map(lambda s: s[len(name):],
                       filter(lambda s: s.startswith(name), iterable))

        settings_for_this_strategy = filter_and_take(prefix,  all_settings)

        for name in settings_for_this_strategy:
            setattr(self, name, getattr(self.parameters, prefix + name))

    @property
    def model(self) -> Type[model.SurrogateModelBase]:
        ''' gets the model and trains it '''
        X, F = self.data.X, self.data.F
        if X is None or F is None:
            return None
        self._model.fit(X, F)
        return self._model

    def fitness_func(self, x):
        ''' evaluate one sample using true objective function & saves the result in the archive '''
        f = self.modcma.fitness_func(x)
        self.data.push(x, f)
        return f

    def apply_sort(self, n):
        if self.parameters.surrogate_strategy_sort_type is None:
            return
        elif self.parameters.surrogate_strategy_sort_type == 'all':
            self.data.sort()
        elif self.parameters.surrogate_strategy_sort_type == 'evaluated':
            self.data.sort(n=n)
        else:
            raise NotImplementedError(
                "Cannot find an implementation for 'surrogate_strategy_sort_type'")

    @abstractmethod
    def __call__(self,
                 X: XType,
                 sort: Union[int, bool] = True,
                 prune: bool = False
                 ) -> YType:
        ''' evaluates all samples using true objective function '''
        F = np.empty(len(X), yType)
        for i in range(len(X)):
            F[i] = self.fitness_func(X[i])

        if isinstance(sort, bool):
            if sort:
                self.apply_sort(len(X))
        elif isinstance(sort, int):
            self.apply_sort(sort)

        if prune:
            self.data.prune()

        return F

    def signal_for_bigger_model(self):
        self.model.signal_for_bigger_model(self.data)


class Unsure_Strategy(SurrogateStrategyBase):
    ''' always calls true evaluation '''
    StrategyName = 'Unsure'

    def __call__(self, X: XType) -> YType:
        return super().__call__(X)


class Random_Strategy(SurrogateStrategyBase):
    ''' randomly use surrogate and true evaluation '''
    StrategyName = 'Random'

    def __init__(self, modcma: ModularCMAES):
        self.eval_relative: float
        super().__init__(modcma)

    def __call__(self, X: XType) -> YType:
        n = min(len(X), math.ceil(len(X) * self.eval_relative))
        sample = np.arange(len(X))
        np.random.shuffle(sample)
        sample, not_sample = sample[:n], sample[n:]
        sample.sort()
        not_sample.sort()

        # objective function
        Xtrue = X[sample]
        Ftrue = super().__call__(Xtrue)
        # surrogate
        Xfalse = X[not_sample]
        Ffalse = self.model.predict(Xfalse)

        # putting it altogether
        F = np.empty(shape=(len(X), 1), dtype=yType)
        F[sample] = Ftrue
        F[not_sample] = Ffalse
        return F


class Kendall_Strategy(SurrogateStrategyBase):
    ''' a '''
    StrategyName = 'Kendall'

    def __init__(self, modcma: ModularCMAES):
        self.minimum_eval_relative: float  # ok
        self.minimum_eval_absolute: int  # ok
        self.truncation_ratio: float

        self.iterative_increse: float

        self.tau_minimum_samples: int
        self.tau_training_size_minimum: int
        self.tau_training_size_relative: float
        self.tau_threshold: float
        self.tau_threashold_to_signal_for_bigger_model: float

        self.return_true_values_if_all_available: bool

        super().__init__(modcma)


    def _get_order_of_F_by_surrogate(self, X, top=None, mask=None):
        """ evaluates some of the X's rows by the true function & returns which one (index) are evaluated and their values
        """
        if mask is not None:
            masked_argange = np.arange(len(X))[mask]
            X = X[mask]

        # try to get actual model ...
        F_model = self.model(X)
        if F_model is None:
            F_model = np.random.rand(len(X), 1)
        order = np.argsort(F_model)[:top]

        if mask is not None:
            order = masked_argange[order]
        return order

    def _kendall_test(self) -> bool:
        n_for_tau = max(
            self.tau_training_size_minimum,
            self.tau_training_size_relative*len(X))

        if self.data.X is None or self.data.F is None:
            return False
        if len(self.data.F) < self.tau_minimum_samples:
            return False

        TestX = self.data.X[-n_for_tau:]
        TestF = self.data.F[-n_for_tau:]

        PredictF = self.model(TestX)

        tau = kendalltau(TestF, PredictF).correlation
        if np.isnan(tau):
            tau = 0
        return tau

    def __call__(self, X: XType) -> YType:
        NE = np.ones(shape=(len(X),), dtype=np.bool_)  # not evaluated
        F = np.tile(np.nan, reps=(len(X), 1))  # values

        evaluated = 0
        to_evaluate = int(math.ceil(max((
            self.minimum_eval_absolute,
            self.minimum_eval_relative * len(X)
        ))))

        while np.any(NE):
            order = _get_order_of_F_by_surrogate(X, top=to_evaluate, mask=NE)
            evaluated += len(order)
            F[order] = super().__call__(X[order], sort=evaluated, prune=True)
            NE[order] = False

            # Kendall test !!!!
            tau = self._kendall_test()
            if tau >= self.tau_threshold:
                break

            to_evaluate = int(math.ceil(evaluated * self.iterative_increse))

        if tau < self.tau_threashold_to_signal_for_bigger_model:
            self.signal_better_model()

        if self.return_true_values_if_all_available and len(X) == evaluated:
            return F
        else:
            f_offset = np.nanmin(F[np.logical_not(NE)])
            F_values = self.model(X)
            m_offset = np.nanmin(F_values)
            return F_values - m_offset + f_offset








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
        F = np.tile(np.nan, reps=(len(X), 1))
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
