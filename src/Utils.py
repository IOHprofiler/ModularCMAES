from datetime import datetime
from functools import wraps
from time import time
import numpy as np

from bbob import bbobbenchmarks, fgeneric
from Constants import DEFAULT_TARGET_DISTANCES, DISTANCE_TO_TARGET


def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        print("Time elapsed", time() - start)
        return res
    return inner


def ert(evals, budget):
    _ert = evals.sum() / (evals < budget).sum()
    return _ert, np.std(evals)


def bbobfunction(ffid, logging=False, label='', iinstance=1, d=5):
    func, target = bbobbenchmarks.instantiate(
        ffid, iinstance=iinstance)
    if not logging:
        return func, target
    label = 'D{}_{}_{}'.format(
        d, label, datetime.now().strftime("%m%d"))
    fitness_func = fgeneric.LoggingFunction(
        f"/home/jacob/Code/thesis/data/{label}", label)
    target = fitness_func.setfun(
        *(func, target)
    ).ftarget
    return fitness_func, target


@timeit
def evaluate(ffid, d, optimizer_class, *args, iterations=50, label='', logging=False, **kwargs):
    evals, fopts = np.array([]), np.array([])
    _, target = bbobfunction(ffid)
    print("Optimizing function {} in {}D for target {} + {}".format(ffid, d, target,
                                                                    DISTANCE_TO_TARGET[ffid - 1]))
    for i in range(iterations):
        fitness_func, target = bbobfunction(
            ffid, label=label, logging=logging, d=d)
        optimizer = optimizer_class(fitness_func, target, d, *args,
                                    rtol=DISTANCE_TO_TARGET[ffid - 1],
                                    ** kwargs)
        optimizer.run()
        evals = np.append(evals, optimizer.used_budget)
        fopts = np.append(fopts, optimizer.fopt)

    print("FCE:\t{:10.8f}\t{:10.4f}\nERT:\t{:10.4f}\t{:10.4f}".format(
        np.mean(fopts), np.std(fopts), *ert(evals, optimizer.budget)
    ))
    return evals, fopts


def plot_cumulative_target(fitness_over_time, abs_target, label=None, log=False):
    # Don't include the points that have hit a target and than decrease.
    fitness_over_time = to_matrix(fitness_over_time)
    bins = np.digitize(fitness_over_time - abs_target,
                       DEFAULT_TARGET_DISTANCES, right=True)

    bins = np.maximum.accumulate(bins, axis=1)
    line = [i.sum() / (len(DEFAULT_TARGET_DISTANCES) * len(i)) for i in bins.T]
    plt.semilogx(line, label=label)
    plt.ylabel("Proportion of function+target pairs")
    plt.xlabel("Function Evaluations")
    plt.legend()


def to_matrix(array):
    max_ = len(max(array, key=len))
    return np.array([
        row + [row[-1]] * (max_ - len(row)) for row in array])
