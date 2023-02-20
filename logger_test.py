import ioh
import numpy as np
import modcma

def main():
    f = ioh.get_problem(fid=3, instance=1, dimension=10, problem_type=ioh.ProblemType.BBOB)
    l = ioh.logger.Analyzer(root="data", folder_name="run", algorithm_name="CMAES_E", algorithm_info="test of IOHexperimenter in python")
    f.attach_logger(l)

    # params = modcma.Parameters(d=10, budget=999, lambda_=50)
    # CMA_0 = modcma.ModularCMAES(fitness_func=f, parameters=params)
    # params = modcma.Parameters(d=10, budget=999, lambda_=100)
    # CMA_1 = modcma.ModularCMAES(fitness_func=f, parameters=params)

    # CMAs: list[modcma.ModularCMAES] = [CMA_0, CMA_1] 
    
    # for i in range(2):
    #     CMAs[i].run()

    # for j in range(len(CMAs)):
    #     print(CMAs[i]._fitness_func.state)

    CMAs: list[modcma.ModularCMAES] = []
    lambdas = list(range(10, 101, 10))
    mus = list(range(10, 101, 10))
    print(lambdas)
    for _ in range(10):
        params = modcma.Parameters(d=10, lambda_=lambdas[_], mu=mus[_])
        cma = modcma.ModularCMAES(fitness_func=f, parameters=params)
        CMAs.append(cma)
    
    for j in range(1000):
        for i in range(len(CMAs)):
            CMAs[i].step()
    
    # for j in range(len(CMAs)):
    #     print(CMAs[j]._fitness_func.state)

    print(f.state)

if __name__ == "__main__":
    main()