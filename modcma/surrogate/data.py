import numpy as np
import math

from typing import Callable, List, Union, Optional, Any
from numpy.typing import NDArray
from scipy.stats import kendalltau
from abc import ABCMeta, abstractmethod, abstractproperty

from modcma.parameters import Parameters
from modcma.typing_utils import XType, YType, xType, yType


class SurrogateData_V1(metaclass=ABCMeta):
    FIELDS = ['_X', '_F']

    def __init__(self, settings: Parameters):
        self.settings = settings

        self._X: Optional[XType] = None
        self._F: Optional[YType] = None

    def push(self, x, f: Union[YType, float]):
        ''' push elements to the archive '''
        x = np.atleast_2d(np.ravel(x))
        f = np.atleast_2d(np.ravel(f))

        if self._X is None or self._F is None:
            self._X, self._F = x, f
        else:
            self._X = np.vstack([self._X, x])
            self._F = np.vstack([self._F, f])

    def push_many(self, X, F):
        ''' same as push but with arrays '''
        if self._X is None or self._F is None:
            self._X, self._F = X, F
        else:
            self._X = np.vstack([self._X, X])
            self._F = np.vstack([self._F, F])

    def pop(self, number: int = 1):
        ''' removes n elements from the beggining of the stack (default=1)
            and returns them
        '''
        if self._X is None or self._F is None:
            return None, None
        x = self._X[:number]
        f = self._F[:number]
        self._X = self._X[number:]
        self._F = self._F[number:]
        return x, f

    def _sort_selection(self, selection: slice):
        ''' implemnts the sorting algorithm; returns order indices '''
        if self._F is None or self._X is None:
            return

        s_type = self.settings.surrogate_data_sorting.lower()

        if s_type == 'lq':
            measure = self._F[selection]
        elif s_type == 'mahalanobis':
            inv_root_C = self.settings.inv_root_C
            p = inv_root_C @ (self._X[selection] - self.settings.m.T)
            measure = (p.T @ p)
        elif s_type == 'time':
            raise RuntimeError('This should not happen')
        else:
            raise NotImplementedError('Unknown sorting method')

        # smallest last
        measure = -measure.ravel()
        order = np.argsort(measure)
        return order


    def sort(self, n: Optional[int] = None) -> None:
        ''' sorts top n elements default: sorts all elements '''

        if (n is not None and n <= 1) \
            or len(self) <= 1 \
            or self.settings.surrogate_data_sorting == 'time':
            return

        if n is None:
            select: slice = slice(None)
            other: slice = slice(0, 0)
        else:
            n = min(len(self), n)
            select: slice = slice(-n, None)
            other: slice = slice(None, -n)

        order = self._sort_selection(select)

        for name in self.FIELDS:
            data = getattr(self, name)
            new_data = [data[other], data[select, :][order, :]]
            setattr(self, name, np.vstack(new_data))

    def prune(self) -> None:
        ''' removes unwanted elements '''

        # MAX_SIZE
        if len(self) > self.model_size:
            self.pop(number=len(self) - self.model_size)

    def __len__(self) -> int:
        ''' number of saved samples (not nessesary for trainign purposes) '''
        if self._F is None:
            return 0
        return self._F.shape[0]

    @property
    def model_size(self) -> int:
        ''' number of samples selected for training a surrogate model '''
        size = len(self)

        # absolute max
        if self.settings.surrogate_data_max_size is not None:
            size = min(size, self.settings.surrogate_data_max_size)

        # relative max
        if self.settings.surrogate_data_max_relative_size is not None:
            if self.settings.surrogate_model_instance is not None:
                if self.settings.surrogate_model_instance.dof > 0:
                    size = min(size,
                               int(math.ceil(
                                   self.settings.surrogate_data_max_relative_size
                                   * self.settings.surrogate_model_instance.dof)))
        return size

        # truncation ratio
        #if self.settings.surrogate_data_truncation_ratio is not None:
        #    size = int(math.ceil(size * self.settings.surrogate_data_truncation_ratio))
        #return size


    def __getitem__(self, items):
        return
        pass

    # MODEL BUILDING BUSINESS

    @property
    def X(self) -> Optional[XType]:  # Covariates
        # TODO: return mahalanobis
        if self._X is None:
            return None

        X = self._X[-self.model_size:]
        if self.settings.surrogate_data_mahalanobis_space:
            X = self.settings.inv_root_C @ (X - self.settings.m.T)
        return X

    @property
    def F(self) -> Optional[YType]:  # Target Values
        if self._F is None:
            return None
        return self._F[-self.model_size:]

    @property
    def W(self):  # Weight
        if self.settings.surrogate_data_weighting == 'linear':
            return np.linspace(self.settings.surrogate_data_min_weight,
                               self.settings.surrogate_data_max_weight,
                               num=self.model_size)
        else:
            raise NotImplementedError("Couldnt interpret the weight_function")


'''
#####################
# Population Storage Management

class FilterUnique(Filter):
    def __call__(self, pop: PopHistory) -> PopHistory:
        _, ind = np.unique(pop.x, axis=1, return_index=True)
        return pop[ind]


class FilterDistance(Filter):
    def __init__(self, parameters: Parameters, distance: float):
        self.max_distance = distance
        self.parameters = parameters

    @abstractmethod
    def _compute_distance(self, pop: PopHistory) -> npt.NDArray[np.float32]:
        pass

    def _get_mask(self, pop: PopHistory) -> npt.NDArray[np.bool_]:
        distance = self._compute_distance(pop)
        return distance <= self.max_distance

    def __call__(self, pop: PopHistory) -> PopHistory:
        mask = self._get_mask(pop)
        return pop[mask]


class FilterDistanceMahalanobis(FilterDistance):
    def __init__(self, parameters: Parameters, distance: float):
        super().__init__(parameters, distance)
        B = self.parameters.B
        sigma = self.parameters.sigma
        D = self.parameters.D

        self.transformation = np.linalg.inv(B) @ np.diag((1./sigma)*(1./D))

    def _compute_distance(self, pop: PopHistory) -> npt.NDArray[np.float32]:
        center_x = pop.x - self.parameters.m
        return np.sqrt(self.transformation @ center_x)


FILTER_TYPE = Union[
    FilterRealEvaluation,
    FilterUnique,
    FilterDistanceMahalanobis
]

'''
