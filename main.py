from modcma import modularcmaes, Parameters, Population
import ioh

from argparse import ArgumentParser
import numpy as np


def initialize_parameters(problem: ioh.ProblemType, budget: int, lambda_: int, mu_: int,
                        n_generations: int = None, x0: np.ndarray = None, sigma0: float = 0.5, a_tpa: float = 0.5, b_tpa: float = 0., cs: float = None,
                        cmu: float = None, c1: float = None, seq_cutoff_factor: int = 1, init_threshold: float = 0.2,
                        decay_factor: float = 0.995, max_resamples: int = 100, active: bool = False, elitist: bool = False,
                        sequential: bool = False, threshold_convergence: bool = False, bound_correction: str = None,
                        orthogonal: bool = False, local_restart: str = None, base_sampler: str = "gaussian",
                        mirrored: str = "mirrored", weights_option: str = "default", step_size_adaptation: str = "csa",
                        population: Population = None, old_population: Population = None, ipop_factor: int = 2,
                        ps_factor: float = 1., sample_sigma: bool = False):        
    CMAES_params = Parameters(d=problem.meta_data.n_variables, lambda_=lambda_, mu=mu_, x0=x0, #ub=problem.bounds.ub, lb=problem.bounds.lb, 
                            budget=budget, target=problem.optimum.y, n_generations=n_generations, sigma0=sigma0, a_tpa=a_tpa, b_tpa=b_tpa, cs=cs, cmu=cmu, c1=c1, 
                            seq_cutoff_factor=seq_cutoff_factor, init_threshold=init_threshold, decay_factor=decay_factor, 
                            max_resamples=max_resamples, active=active, elitist=elitist, sequential=sequential, 
                            threshold_convergence=threshold_convergence, bound_correction=bound_correction, orthogonal=orthogonal, 
                            local_restart=local_restart, base_sampler=base_sampler, mirrored=mirrored, weights_option=weights_option, 
                            step_size_adaptation=step_size_adaptation, population=population, old_population=old_population, 
                            ipop_factor=ipop_factor, ps_factor=ps_factor, sample_sigma=sample_sigma)
    return CMAES_params

def initialize(problem_id: int, problem_instance: int, dimension: int, budget: int, iterations: int, lambda_: list[int], mu_: list[int], sharing_point: int, logger_info: dict):
    problem = ioh.get_problem(fid=problem_id, instance=problem_instance, dimension=dimension, problem_type=ioh.ProblemType.BBOB) # TODO replace dimension
    # print(problem)

    logger = ioh.logger.Analyzer(root="data", folder_name="run", algorithm_name=logger_info['name'], algorithm_info="test of IOHexperimenter in python")
    problem.attach_logger(logger)

    # initialize (subpop) number of populations
    subpop_n = len(lambda_)
    cmaes = []
    for i in range(subpop_n):
        params = initialize_parameters(problem=problem, budget=budget, lambda_=lambda_[i], mu_=mu_[i])
        cmaes.append(modularcmaes.ModularCMAES(fitness_func=problem, parameters=params))

    return run_cma(CMAES=cmaes, iterations=iterations, sharing_point=sharing_point)

def run_cma(CMAES: list[modularcmaes.ModularCMAES], iterations: int, sharing_point: int):
    for _ in range(iterations):
        for i in range(len(CMAES)):
            break_condition = CMAES[i].step()
            if break_condition: break
            # print(flag)
    return CMAES

def main(problem_id: int, problem_instance: int, dimension: int, iterations: int, budget: int, lambda_: int, mu_: int, sharing_point: int, logger_info: dict):
    # initialize
    cmaes = initialize(problem_id=problem_id, problem_instance=problem_instance, dimension=dimension, budget=budget, iterations=iterations, lambda_=lambda_, mu_=mu_, sharing_point=sharing_point, logger_info=logger_info)
    
    # TODO fix run algo
    # cmaes = run_cma(CMAES=cmaes, iterations=iterations, sharing_point=sharing_point)

    # display results
    for cma in cmaes:
        print(f"CMA: {cma} f:{cma.parameters.fopt}, {cma._fitness_func.optimum.y}, {cma.parameters.population}")

    
def normal_cma_benchmark(problem_id: int, dimension: int, iterations: int, budget: int, lambda_: int, mu_: int):
    problem = ioh.get_problem(fid=problem_id, instance=1, dimension=dimension, problem_type=ioh.ProblemType.BBOB)

    logger = ioh.logger.Analyzer(root='data', folder_name="run", algorithm_name="ModCMAES", algorithm_info="default version of ModCMAES benchmarked with default parameters.")
    problem.attach_logger(logger)

    params = initialize_parameters(problem=problem, budget=budget, lambda_=lambda_, mu_=mu_)

    cmaes = modularcmaes.ModularCMAES(fitness_func=problem, parameters=params)
    for _ in range(iterations):
        cmaes.step()

if __name__ == "__main__":
    # TODO: seed
    parser = ArgumentParser()
    parser.add_argument("-pid", "--problem_id", help="decide which BBOB problem to run CMAES on", choices=[3, 4], default=3, type=int) # only those two for now
    parser.add_argument("-iid", "--problem_instance", help="decide which instance of the problem to run CMAES on", choices=list(range(1,16)), default=1, type=int) # instances 1-15
    parser.add_argument("-d", "--dimension", help="the dimension of the problem we are solving", default=5, type=int) # only those two for now
    parser.add_argument("-i", "--iterations", help="how many iterations to run the algorithm for", default=10000, type=int) # only those two for now
    parser.add_argument("-b", "--budget", help="decides the budget of the problem, defaults to 1e4*dimension", nargs='?', default=1e4, type=float)
    parser.add_argument("-pt", "--subpop_type", help="sizes of subpopulations", choices=[1,2], default=1, type=int)
    parser.add_argument("-n", "--algo_name", help="algorithm name", default='ModCMA', type=str)
    # parser.add_argument("-l", "--lambda_", help="size of offsprings to generate", nargs='*', default=100, type=int)
    # parser.add_argument("-m", "--mu_", help="size of parents to use", nargs='*', default=100, type=int)
    # parser.add_argument("-p", "--subpop", help="size of subpopulation parents and offspring\neg: [100, 50, 20, 10, 5]", nargs='+', default=None, type=int)
    # parser.add_argument("-sn", "--sub_size", help="number of subpopulation", nargs='?', default=10, type=int)
    # parser.add_argument("-sp", "--info_sharing_point", help="when to share info between subpops", nargs='?', default=0, type=int)
    args = parser.parse_args()
    
    # TODO update lambda_ and mu_ arguments for subpopulation
    if args.subpop_type == 1:
        # no subpopulations, hard-coded size (hard-coded for now)
        lambda_ = [100]
        mu_ = [100]
    elif args.subpop_type == 2:
        # multiple subpopulations, same sizes (hard-coded for now)
        lambda_ = [20, 20, 20, 20, 20]
        mu_ = [20, 20, 20, 20, 20]

    print("Subpopulation CMA-ES: start")
    main(problem_id=args.problem_id, 
         problem_instance=args.problem_instance,
         dimension=args.dimension, 
         iterations=args.iterations, 
         budget=int(args.budget*args.dimension), 
         lambda_=lambda_,
         mu_=mu_, 
         sharing_point=None,
         logger_info={'name':args.algo_name, 'description': ''})
    print("Subpopulation CMA-ES: complete")
    # python main.py -pt