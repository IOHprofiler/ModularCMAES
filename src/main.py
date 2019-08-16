from time import time
import numpy as np
import matplotlib.pyplot as plt

from bbob import bbobbenchmarks
from ConfigurableCMA import ConfigurableCMA
from argparse import ArgumentParser
from Parameters import Parameters

POWERS = [round(2 - ((p - 1) * .2), 2) for p in range(1, 51)]

DEFAULT_TARGET_DISTANCES = list(map(lambda x: pow(10, x), POWERS))


def to_matrix(array):
    max_ = len(max(array, key=len))
    return np.array([
        row + [row[-1]] * (max_ - len(row)) for row in array])


def plot_cumulative_target(fitness_over_time, abs_target, label=None, log=False):
    # Don't include the points that have hit a target and than decrease.
    fitness_over_time = to_matrix(fitness_over_time)
    bins = np.digitize(fitness_over_time - abs_target,
                       DEFAULT_TARGET_DISTANCES, right=True)

    # breakpoint()
    bins = np.maximum.accumulate(bins, axis=1)
    line = [i.sum() / (len(DEFAULT_TARGET_DISTANCES) * len(i)) for i in bins.T]
    plt.semilogx(line, label=label)
    plt.ylabel("Proportion of function+target pairs")
    plt.xlabel("Function Evaluations")
    plt.legend()


def evaluate(functionid, dim, iterations):
    start = time()
    ets, fce, fs = [], [], []
    for i in range(iterations):
        cma, target = ConfigurableCMA.make(functionid, d=dim)
        cma.run()
        ets.append(cma.parameters.used_budget)
        fce.append(cma.parameters.fmin)
        fs.append(cma.parameters.fitness_over_time)

    ets = np.array(ets)
    print("FCE:\t{:10.8f}\t{:10.4f}\nERT:\t{:10.4f}\t{:10.4f}".format(
        np.mean(fce),
        np.std(fce),
        ets.sum() / (ets != cma.parameters.budget).sum(),
        np.std(ets)
    ))

    print("Time:\t", time() - start)
    # plot_cumulative_target(fs, (target + cma.parameters.rtol), label='rewrite')
    # plt.show()


def main():
    parser = ArgumentParser(
        description='Run single function MAB exp iid 1')
    parser.add_argument(
        '-f', "--functionid", type=int,
        help="bbob function id", required=False, default=5
    )
    parser.add_argument(
        '-d', "--dim", type=int,
        help="dimension", required=False, default=5
    )
    parser.add_argument(
        '-i', "--iterations", type=int,
        help="number of iterations per agent",
        required=False, default=50
    )
    args = vars(parser.parse_args())
    np.random.seed(42)
    print("rewrite")
    evaluate(**args)

    import subprocess
    print("old")
    subprocess.run(
        "/home/jacob/Documents/thesis/.env/bin/python "
        f"/home/jacob/Documents/thesis/OnlineCMA-ES/src/main.py "
        f"-f {args['functionid']} -d {args['dim']} -i {args['iterations']} --clear", shell=True
    )
    # plt.show()


if __name__ == "__main__":
    '''
    There is a slight performance difference still between old and new code,
        old code seems more efficient on functions:
            1, 3, 5, 6
    Checks:
        ~ Parameters are exactly the same
        ~ Using same random seed
        ~ Mutation function is correct
        ~ Recombination is correct
        ~ Selection is correct
        ~ Did stepwize check by replacing every line in adapt method for old code,
            no difference in performance observed
    Differences:
        ~ Indivudual object:
            - implemets mutatation
        ~ Populations object:
            - More advanced data holder
            - iterable implementation of its indivuals
        ~ Parameters objects:
            - New holds more parameters
            - Restart is different -> new restart works better
        ~ Order of generation loop:
            - old code is incorrect.
    '''
    main()
