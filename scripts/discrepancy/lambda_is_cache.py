import time
import argparse

import ioh
import numpy as np
from scipy.stats.qmc import discrepancy
from scipy.stats import norm

from modcma.c_maes import ModularCMAES, parameters, options, constants, utils, sampling
      

class Stats:
    cache_discrepancy = float("inf")
    population_discrepancy = float("inf")
    selected_discrepancy = float("inf")
    
    

if __name__ == "__main__":
    """
    data/pointsets/OptFib_2_16.txt
    data/pointsets/Sub_Sobol_10_16.txt
    data/pointsets/Sub_Sobol_20_16.txt
    data/pointsets/Sub_Sobol_40_16.txt
    data/pointsets/Sub_Sobol_5_16.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--fid", type=int, default=0)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--logged", action='store_true')
    parser.add_argument("--sampler", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--lamb", type=int, default=16)
    
    args = parser.parse_args()
    
    if args.fid != 0:
        fids = [args.fid]
    else:
        fids = list(range(1, 25))
        
    modules = parameters.Modules()
    if args.path: 
        points = []
        with open(args.path) as f:
            next(f)
            for line in f:
                points.append(list(map(float, line.strip().split())))
                
        points = np.array(points)
        cache_size, dim = points.shape
        
        constants.cache_max_doubles = 0
        constants.cache_min_samples = cache_size
        constants.cache_samples = True
        algorithm_name = f"CMA-ES-OPT-cache-{cache_size}-lambda{args.lamb}"
    else:
        dim = args.dim
        sampler = options.BaseSampler(args.sampler)
        algorithm_name = f"CMA-ES-{sampler.name}-lambda{args.lamb}"
        modules.sampler = sampler
    
    
    if args.logged:
        logger = ioh.logger.Analyzer(
            root="data",
            folder_name=algorithm_name,
            algorithm_name=algorithm_name
        )
        logger.add_run_attributes(Stats, ["cache_discrepancy"])   
            
    
    for fid in fids:
        for iid in range(1, 101):
            utils.set_seed(fid * dim * iid)
            np.random.seed(fid * dim * iid)
            problem = ioh.get_problem(fid, iid, dim)
            if args.logged:
                problem.attach_logger(logger)
                
            start = time.perf_counter()
            settings = parameters.Settings(
                problem.meta_data.n_variables,
                x0=np.random.uniform(-4, 4, size=dim),
                sigma0=(problem.bounds.ub[0] - problem.bounds.lb[0]) *.2,
                budget=problem.meta_data.n_variables * args.budget,
                target=problem.optimum.y + 1e-8,
                lambda0=args.lamb,
                modules=modules
            )
        
            cma = ModularCMAES(settings)
            
            if args.path:
                cma.p.sampler = sampling.CachedSampler(points[np.random.permutation(len(points))], True)
                cached_sample = norm.cdf(np.vstack([cma.p.sampler() for _ in range(cache_size)]))
                Stats.cache_discrepancy = discrepancy(cached_sample, method="L2-star")
                print("cache discrepancy: ", Stats.cache_discrepancy)
          
            cma.run(problem)
                
            print(
                problem.meta_data, 
                problem.state.evaluations, 
                problem.state.final_target_found, 
                problem.state.current_best_internal .y,
                time.perf_counter() - start                        
            )
            problem.reset()