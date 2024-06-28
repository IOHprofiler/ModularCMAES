import argparse
import os
import pickle

import ioh
import modcma.c_maes as c_cmaes

def get_problem(fid, instance, dim):
    if fid < 25:
        return ioh.get_problem(fid, instance, dim)
    problem = ioh.get_problem(fid, problem_class=ioh.ProblemClass.CEC2013)
    problem.set_optimum(instance % problem.n_optima)
    problem.invert()
    problem.set_instance(instance)
    return problem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=int, default=1)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n_instances", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=50)
    parser.add_argument("--elitist", action="store_true")
    parser.add_argument("--logged", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--coverage", type=float, default=10.0)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--strat", type=int, choices=(0, 1, 2))
    args = parser.parse_args()

    root = os.path.realpath(os.path.dirname(__file__))
    
    modules = c_cmaes.parameters.Modules()
    if args.strat == 0:
        modules.restart_strategy = c_cmaes.options.RESTART
    elif args.strat == 1:
        modules.restart_strategy = c_cmaes.options.IPOP
    else:
        modules.restart_strategy = c_cmaes.options.BIPOP


    modules.center_placement = c_cmaes.options.UNIFORM
    modules.bound_correction = c_cmaes.options.SATURATE
    modules.repelling_restart = args.coverage != 0
    modules.elitist = args.elitist
    
    algorithm_name=f"CMA-ES-{str(modules.restart_strategy).split('.')[-1]}"
    if args.coverage != 0:
        algorithm_name += f"-repelling-c{args.coverage}"
    if args.elitist:
        algorithm_name += "-elitist"

    if args.logged:
        logger = ioh.logger.Analyzer(
            root=os.path.join(root, "data/ioh"), 
            algorithm_name=algorithm_name, 
            folder_name=algorithm_name
        )
    centers = []
    for instance in range(1, args.n_instances + 1):
        problem = get_problem(args.fid, instance, args.dim)
        if args.logged:
            problem.attach_logger(logger)

        for run in range(args.n_runs):
            c_cmaes.utils.set_seed(42 * run)
            settings = c_cmaes.parameters.Settings(
                problem.meta_data.n_variables,
                modules,
                sigma0=(problem.bounds.ub[0] - problem.bounds.lb[0]) *.2,
                budget=problem.meta_data.n_variables * args.budget,
                target=problem.optimum.y + 1e-8,
            )

            parameters = c_cmaes.Parameters(settings)
            parameters.repelling.coverage = args.coverage
            cma = c_cmaes.ModularCMAES(parameters)
            cma.run(problem)
            centers.append((instance, run, [(sol.x, sol.y, sol.t, sol.e) for sol in cma.p.stats.centers]))
            problem.reset()

    if args.logged:
        with open(os.path.join(root, f"data/pkl/{algorithm_name}_fid{args.fid}_dim{args.dim}.pkl"), "wb+") as f:
            pickle.dump(centers, f)    



