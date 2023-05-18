import sys
import numpy as np

from modcma import ModularCMAES

np.set_printoptions(precision=3)

evals = 0


def quad(x):
    global evals
    evals += 1
    res = np.sum(np.power(x, 2))
    print(f' {x} -> {res:0.2f}')
    return res


obj_func = quad
dim = 5
maxfun = 1000

cma = ModularCMAES(obj_func, dim, budget=maxfun).run()
res = cma.parameters.xopt, cma.parameters.fopt, cma.parameters.used_budget
print(res)

print("evals:", evals)

# cmaes = ModularCMAES(sum, d=2)
# cmaes.fitness_func(np.array([1,2]))
