from modcma import modularcmaes, Parameters, Population
import ioh

from argparse import ArgumentParser

def initialize_parameters(problem: ioh.ProblemType, budget: int, lambda_: int, mu_: int,
                        sigma0: float = 0.5, a_tpa: float = 0.5, b_tpa: float = 0., cs: float = None,
                        cmu: float = None, c1: float = None, seq_cutoff_factor: int = 1, init_threshold: float = 0.2,
                        decay_factor: float = 0.995, max_resamples: int = 100, active: bool = False, elitist: bool = False,
                        sequential: bool = False, threshold_convergence: bool = False, bound_correction: str = None,
                        orthogonal: bool = False, local_restart: str = None, base_sampler: str = "gaussian",
                        mirrored: str = "mirrored", weights_option: str = "default", step_size_adaptation: str = "csa",
                        population: Population = None, old_population: Population = None, ipop_factor: int = 2,
                        ps_factor: float = 1., sample_sigma: bool = False):        
    CMAES_params = Parameters(d=problem.meta_data.n_variables, lambda_=lambda_, mu=mu_, #ub=problem.bounds.ub, lb=problem.bounds.lb, 
                            budget=budget, target=problem.optimum.y, sigma0=sigma0, a_tpa=a_tpa, b_tpa=b_tpa, cs=cs, cmu=cmu, c1=c1, 
                            seq_cutoff_factor=seq_cutoff_factor, init_threshold=init_threshold, decay_factor=decay_factor, 
                            max_resamples=max_resamples, active=active, elitist=elitist, sequential=sequential, 
                            threshold_convergence=threshold_convergence, bound_correction=bound_correction, orthogonal=orthogonal, 
                            local_restart=local_restart, base_sampler=base_sampler, mirrored=mirrored, weights_option=weights_option, 
                            step_size_adaptation=step_size_adaptation, population=population, old_population=old_population, 
                            ipop_factor=ipop_factor, ps_factor=ps_factor, sample_sigma=sample_sigma)
    return CMAES_params

def initialize(problem_id: int, dimension: int, budget: int, lambda_: list[int], mu_: list[int]):
    problem = ioh.get_problem(fid=problem_id, instance=1, dimension=dimension, problem_type=ioh.ProblemType.BBOB) # TODO replace dimension
    # print(problem)

    # params = initiate_parameters(problem=problem, budget=budget, lambda_=(lambda_//subpop_n), mu_=(mu_//subpop_n))
    # print(params)

    # TODO fix logger issue
    # log = ioh.logger.Analyzer(root="data", folder_name="run", algorithm_name=f"CMAES_E", algorithm_info="test of IOHexperimenter in python")
    # problem.attach_logger(log)

    # initialize (subpop) number of populations
    subpop_n = len(lambda_)
    print(subpop_n)
    cmaes = []
    for i in range(subpop_n):
        params = initialize_parameters(problem=problem, budget=budget, lambda_=lambda_[i], mu_=mu_[i])
        cmaes.append(modularcmaes.ModularCMAES(fitness_func=problem, parameters=params))

    return cmaes

def run_cma(CMAES: list[modularcmaes.ModularCMAES], iterations: int, sharing_point: int):
    for _ in range(iterations):
        for i in range(len(CMAES)):
            flag = CMAES[i].step()
            # print(flag)
    return CMAES

def main(problem_id: int, dimension: int, iterations: int, budget: int, lambda_: int, mu_: int, sharing_point: int):
    # initialize
    cmaes = initialize(problem_id=problem_id, dimension=dimension, budget=budget, lambda_=lambda_, mu_=mu_)
    # run algo
    cmaes = run_cma(CMAES=cmaes, iterations=iterations, sharing_point=sharing_point)
    # display results
    for cma in cmaes:
        print(f"CMA: {cma} f:{cma.parameters.fopt}, {cma._fitness_func.optimum.y}")

    

if __name__ == "__main__":
    # TODO: seed
    parser = ArgumentParser()
    parser.add_argument("-id", "--problem_id", help="decide which BBOB problem to run CMAES on", choices=[3, 4], default=3, type=int) # only those two for now
    parser.add_argument("-d", "--dimension", help="the dimension of the problem we are solving", default=5, type=int) # only those two for now
    parser.add_argument("-i", "--iterations", help="how many iterations to run the algorithm for", default=1000, type=int) # only those two for now
    parser.add_argument("-b", "--budget", help="how many iterations to run CMAES for", nargs='?', default=1000, type=int)
    parser.add_argument("-pt", "--subpop_type", help="sizes of subpopulations", choices=[1,2,3], default=1, type=int)
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
    else:
        # multiple subpopulations, different sizes (hard-coded for now)
        lambda_ = [5, 10, 20, 30, 50]
        mu_ = [5, 10, 20, 30, 50]
    
    main(problem_id=args.problem_id, 
            dimension=args.dimension, 
            iterations=args.iterations, 
            budget=args.budget, 
            lambda_=lambda_,
            mu_=mu_,
            sharing_point=None)


    # python main.py -