#!/usr/bin/env python3
# 
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import ioh
from ioh import logger
import sys

from ConfigSpace import Configuration, ConfigurationSpace, ForbiddenGreaterThanRelation
from ConfigSpace import NormalFloatHyperparameter
from ConfigSpace import Normal, Integer
from ConfigSpace.hyperparameters import MultiConditionalNormalFloatHyperparameter
# from IPython.display import display
from modcma.c_maes import (
    ModularCMAES,
    Parameters,
    options,
    parameters,
    utils,
    constants
)


from itertools import product
from functools import partial
from multiprocessing import Pool

import numpy as np
import time

import argparse

def create_search_space(DIM, use_learning_rates):
    # print(DIM, use_learning_rates)
    BUDGET = DIM * 10_000
    default_lambda =  (4 + np.floor(3 * np.log(DIM))).astype(int)

    def mueff(lambda_, mu, weight_option):
        modules = parameters.Modules()
        weights_mapping = {
            "default": options.RecombinationWeights.DEFAULT,
            "equal": options.RecombinationWeights.EQUAL,
            "1/2^lambda": options.RecombinationWeights.EXPONENTIAL,
        }
        modules.weights = weights_mapping[weight_option] 
        settings = parameters.Settings(DIM, modules, budget=BUDGET, lambda0=lambda_, mu0=mu)
        return Parameters(settings).weights.mueff

    def c1(lambda_, mu, weight_option):
        _mueff = mueff(lambda_, mu, weight_option)
        return 2 / (_mueff + pow(DIM + 1.3, 2))

    def cmu(lambda_, mu, weight_option, c1):
        _mueff = mueff(lambda_, mu, weight_option)
        return min(1-c1, 2.0 * ((_mueff - 2.0 + (1.0 / _mueff)) / (pow(DIM + 2.0, 2) + (2.0 * _mueff / 2))))

    def cc(lambda_, mu, weight_option):
        _mueff = mueff(lambda_, mu, weight_option)
        return (4 + (_mueff / DIM)) / (DIM + 4 + (2 * _mueff / DIM))


    cma_cs = ConfigurationSpace(
        {
            "covariance": ["covariance", "matrix", "seperable", "none", "cholesky", "cmsa", "natural_gradient"],
            "active": [False, True],
            "elitist": [False, True],
            "orthogonal": [False, True],
            "sequential": [False, True],
            "threshold": [False, True],
            "base_sampler": ["halton", "sobol", "gaussian"],
            "weights_option": ["default", "equal", "1/2^lambda"],
            "mirrored": ["nan", "mirrored", "mirrored_pairwise"],
            "step_size_adaptation": ["csa", "psr", "tpa", "msr", "mxnes", "sr", "sa"],
            "local_restart": ["nan", "IPOP", "BIPOP", "Restart"],
            # "bound_correction": ["nan",  "saturate"],# "cotn", "mirror", "toroidal", "uniform"],
            "sample_transform": ["Gaussian", "Cauchy", "Uniform", "dWeibull"], #, "Laplace", "Logistic", "dWeibull",
            # "repelling_restart": [False, True],
            "lambda_": Integer(
                'lambda_', bounds=(1,20*DIM), distribution = Normal(default_lambda, 10), log=True
            ),
            "mu": Integer(
                'mu', bounds=(1,10*DIM), distribution = Normal(default_lambda // 2, 10), log=True
            ),
        }
    ) 


    forbidden_clause = ForbiddenGreaterThanRelation(cma_cs['mu'], cma_cs['lambda_'])
    cma_cs.add(forbidden_clause)
    # cond = NotEqualsCondition(cma_cs['repelling_restart'], cma_cs['local_restart'], 'nan')
    # cma_cs.add(cond)

    if use_learning_rates:
        cs_param = NormalFloatHyperparameter(
            "cs", lower=0.0, upper=1.5, default_value=0.3, mu=0.3, sigma=0.2)
        
        # Define the conditional normal float hyperparameter
        c1_param = MultiConditionalNormalFloatHyperparameter("c1", ["lambda_", "mu", "weights_option"], c1, lambda lambda_, mu, weight_option: 0.1, lower=0.0001, upper=1.0, default_value=c1(default_lambda, default_lambda // 2, "default"))
        cc_param = MultiConditionalNormalFloatHyperparameter("cc", ["lambda_", "mu", "weights_option"], cc, lambda lambda_, mu, weight_option: 0.1, lower=0.0001, upper=1.0, default_value=cc(default_lambda, default_lambda // 2, "default"))
        cmu_param = MultiConditionalNormalFloatHyperparameter("cmu", ["lambda_", "mu", "weights_option", "c1"], cmu, lambda lambda_, mu, weight_option, c1: 0.1, lower=0.0001, upper=1.0, default_value=cmu(default_lambda, default_lambda // 2, "default", c1(default_lambda, default_lambda // 2, "default")))

        cma_cs.add([cs_param, c1_param, cc_param, cmu_param])
    return cma_cs

def configspace_to_irace_params_txt(cs: ConfigurationSpace, filename: str, dim = 5, use_learning_rates = False, fid = 1):
    """
    Generate an irace parameters txt file from a ConfigSpace object.

    :param cs: ConfigurationSpace object
    :param filename: Output file path
    """
    with open(filename, "w") as f:
        f.write(f"fid \"--fid \" c ({fid})\n")
        f.write(f"dim \"--dim \" c ({dim})\n")
        f.write(f"use_learning_rates \"--use_learning_rates \" c ({use_learning_rates})\n")
        for hp in list(cs.values()):
            # breakpoint()
            name = hp.name
            if hasattr(hp, "choices"):
                # Categorical
                values = ", ".join(str(v) for v in hp.choices)
                f.write(f"{name} \"--{name} \" c ({values})\n")
            elif hasattr(hp, "lower"):
                # Integer or Float
                lower, upper = hp.lower, hp.upper
                if "Integer" in type(hp).__name__:
                    type_str = "i"
                else:
                    type_str = "r"
                if hasattr(hp, "log") and hp.log:
                    type_str = f"{type_str},log"
                
                f.write(f"{name} \"--{name} \" {type_str} ({lower},{upper})\n")

            else:
                # Fallback
                print(f"# {name} type not recognized\n")
        # Forbidden clauses
        if cs.forbidden_clauses:
            f.write("\n[forbidden]\n")
        for fc in cs.forbidden_clauses:
            if isinstance(fc, ForbiddenGreaterThanRelation):
                f.write(f"{fc.left.name} > {fc.right.name}\n")

def config_to_cma_parameters(config, dim, budget, target):
    # modules first
    modules = parameters.Modules()
    active = bool(config.get("active"))
    if config.get("active") == "True":
        active = True
    if config.get("active") == "False":
        active = False
    modules.active = active

    elitist = bool(config.get("elitist"))
    if config.get("elitist") == "True":
        elitist = True
    if config.get("elitist") == "False":
        elitist = False
    modules.elitist = elitist

    if "orthogonal" in config.keys():
        orthogonal = bool(config.get("orthogonal"))
        if config.get("orthogonal") == "True":
            orthogonal = True
        if config.get("orthogonal") == "False":
            orthogonal = False
        modules.orthogonal = orthogonal

    if "sigma" in config.keys():
        sigma = bool(config.get("sigma"))
        if config.get("sigma") == "True":
            sigma = True
        if config.get("sigma") == "False":
            sigma = False
        modules.sample_sigma = sigma

    if "sequential" in config.keys():
        sequential = bool(config.get("sequential"))
        if config.get("sequential") == "True":
            sequential = True
        if config.get("sequential") == "False":
            sequential = False
        modules.sequential_selection = sequential

    if "threshold" in config.keys():
        threshold = bool(config.get("threshold"))
        if config.get("threshold") == "True":
            threshold = True
        if config.get("threshold") == "False":
            threshold = False
        modules.threshold_convergence = threshold

    if "repelling_restart" in config.keys():
        repelling_restart = bool(config.get("repelling_restart"))
        if config.get("repelling_restart") == "True":
            repelling_restart = True
        if config.get("repelling_restart") == "False":
            repelling_restart = False
        modules.repelling_restart = repelling_restart

    if "bound_correction" in config.keys():
        correction_mapping = {
            "cotn": options.CorrectionMethod.COTN,
            "mirror": options.CorrectionMethod.MIRROR,
            "nan": options.CorrectionMethod.NONE,
            "saturate": options.CorrectionMethod.SATURATE,
            "toroidal": options.CorrectionMethod.TOROIDAL,
            "uniform": options.CorrectionMethod.UNIFORM_RESAMPLE,
        }
        modules.bound_correction = correction_mapping[config.get("bound_correction")]

    mirrored_mapping = {
        "mirrored": options.Mirror.MIRRORED,
        "nan": options.Mirror.NONE,
        "mirrored_pairwise": options.Mirror.PAIRWISE,
    }
    modules.mirrored = mirrored_mapping[config.get("mirrored")]

    restart_strategy_mapping = {
        "IPOP": options.RestartStrategy.IPOP,
        "nan": options.RestartStrategy.NONE,
        "BIPOP": options.RestartStrategy.BIPOP,
        "Restart": options.RestartStrategy.RESTART,
    }
    modules.restart_strategy = restart_strategy_mapping[config.get("local_restart")]

    sampler_mapping = {
        "sobol": options.BaseSampler.SOBOL,
        "gaussian": options.BaseSampler.UNIFORM,
        "halton": options.BaseSampler.HALTON,
    }
    modules.sampler = sampler_mapping[config.get("base_sampler")]

    sample_transform_mapping = {
        "Gaussian": options.SampleTranformerType.GAUSSIAN,
        "Cauchy": options.SampleTranformerType.CAUCHY,
        "dWeibull": options.SampleTranformerType.DOUBLE_WEIBULL,
        "Laplace": options.SampleTranformerType.LAPLACE,
        "Logistic": options.SampleTranformerType.LOGISTIC,
        "Uniform": options.SampleTranformerType.SCALED_UNIFORM,
    }
    modules.sample_transformation = sample_transform_mapping[config.get("sample_transform")]

    ssa_mapping = {
        "csa": options.StepSizeAdaptation.CSA,
        "psr": options.StepSizeAdaptation.PSR,
        "lpxnes": options.StepSizeAdaptation.LPXNES,
        "msr": options.StepSizeAdaptation.MSR,
        "mxnes": options.StepSizeAdaptation.MXNES,
        "tpa": options.StepSizeAdaptation.TPA,
        "xnes": options.StepSizeAdaptation.XNES,
        "sr": options.StepSizeAdaptation.SR,
        "sa": options.StepSizeAdaptation.SA,

    }

    modules.ssa = ssa_mapping[config.get("step_size_adaptation")]

    weights_mapping = {
        "default": options.RecombinationWeights.DEFAULT,
        "equal": options.RecombinationWeights.EQUAL,
        "1/2^lambda": options.RecombinationWeights.EXPONENTIAL,
    }
    modules.weights = weights_mapping[config.get("weights_option")]


    covariance_mapping = {
        "covariance": options.MatrixAdaptationType.COVARIANCE,
        "seperable": options.MatrixAdaptationType.SEPARABLE,
        "matrix": options.MatrixAdaptationType.MATRIX,
        "none": options.MatrixAdaptationType.NONE,
        "cholesky": options.MatrixAdaptationType.CHOLESKY,
        "cmsa": options.MatrixAdaptationType.CMSA,
        "natural_gradient": options.MatrixAdaptationType.NATURAL_GRADIENT,
        "covariance_no_eigv": options.MatrixAdaptationType.COVARIANCE_NO_EIGV,
    }
    modules.matrix_adaptation = covariance_mapping[config.get("covariance")]
    #TODO: BOUNDS
    settings = parameters.Settings(dim, modules, budget=budget, lambda0=int(config.get("lambda_")), mu0=int(config.get("mu")), sigma0=2, #config.get("sigma0"),
                                   c1=config.get("c1"), cc=config.get("cc"), cmu=config.get("cmu"), cs=config.get("cs"), target=target,
                                   lb= -5 * np.ones(dim), ub=5 * np.ones(dim), verbose=False)
    return Parameters(settings)



def calc_aoc(problem, logger, budget, fid, iid, DIM):
    data = logger.data()
    data1 = data['None'][fid][DIM][iid][0]
    fvals = [x['raw_y_best'] for x in data1.values()]
    fvals = np.array(fvals)
    if np.isnan(fvals).any():
        np.nan_to_num(fvals, copy=False, nan=1e8)
    if len(fvals) < budget:
        fvals = np.concatenate([fvals, (budget-len(fvals))*[np.min(fvals)]])
    parts = np.log10(np.clip(fvals[:budget], 1e-8, 1e2))+8
    return np.mean(parts)/10


def get_bbob_performance(config : Configuration, instance: str, seed: int = 0, DIM = 5):
        # print(instance, seed, DIM)
        # print(config.get_dictionary())
        constants.use_box_muller = False
        fid, iid = instance.split(",")
        fid = int(fid[1:])
        iid = int(iid[:-1])
        np.random.seed(seed + iid)
        utils.set_seed(seed + iid)
        BUDGET = DIM * 10_000
        l3 = logger.Store(triggers=[ioh.logger.trigger.ALWAYS], properties=[ioh.logger.property.RAWYBEST])
        problem = ioh.get_problem(fid, iid, DIM)
        problem.attach_logger(l3)
        par = config_to_cma_parameters(config, DIM, int(BUDGET), problem.optimum.y + 1e-9)
        par.bounds.lb = problem.bounds.lb
        par.bounds.ub = problem.bounds.ub
        par.settings.lb = problem.bounds.lb
        par.settings.ub = problem.bounds.ub
        if par == False:
            print("Wrong mu/lambda")
            return [np.inf]
        cma = ModularCMAES(par)
        try:
            cma.run(problem)
        except Exception as e:
            print(
                f"Found target {problem.state.current_best.y} target, but exception ({e}), so run failed"
            )
            print(config)
            return [np.inf]
        auc = calc_aoc(problem, l3, BUDGET, fid, iid, DIM)
        return [auc] #minimizing


# def run_smac(arg):
#     from smac import Scenario
#     from smac.acquisition.function import PriorAcquisitionFunction
#     from smac import AlgorithmConfigurationFacade, HyperparameterOptimizationFacade

#     fid, DIM, USE_LEARNING_RATES = arg
#     print(f"Running SMAC with fid={fid}, USE_LEARNING_RATES={USE_LEARNING_RATES} and DIM={DIM}")
#     cma_cs = create_search_space(DIM, USE_LEARNING_RATES)
#     iids = range(1, 51)
#     if fid == 'all':
#         fids = range(1, 25)
#         min_budget = 24
#         max_budget = 240
#     else: 
#         fids = [fid]
#         min_budget = 3
#         max_budget = 50
#     args = list(product(fids, iids))
#     np.random.shuffle(args)
#     inst_feats = {str(arg): [arg[0]] for idx, arg in enumerate(args)}
#     scenario = Scenario(
#         cma_cs,
#         name=str(int(time.time())) + "-" + "CMA",
#         deterministic=False,
#         n_trials=100_000,
#         instances=args,
#         instance_features=inst_feats,
#         output_directory=f"/local/vermettendl/TuningCMA/BBOB/F{fid}_{DIM}D_LR{USE_LEARNING_RATES}", 
#         n_workers=1
#     )
#     pf = PriorAcquisitionFunction(
#         acquisition_function=AlgorithmConfigurationFacade.get_acquisition_function(scenario),
#         decay_beta=scenario.n_trials / 10,  
#     )
#     eval_func = partial(get_bbob_performance, DIM=DIM)
#     intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=max_budget)
#     smac = HyperparameterOptimizationFacade(scenario, eval_func, acquisition_function = pf, 
#                                             intensifier=intensifier)
#     smac.optimize()

    
def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    

    arguments = list(arguments)
    p = Pool(min(200, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results

def parse_paramters(configuration_paramters):
    """
    Function to parse parameters into usable dictionaries

    Parameters
    ----------
    configuration_paramters:
        The command line arguments, without the executables name
    Notes
    -----
    Returns two dicts: one for the problem-specific settings and one for the algorithm settings.
    This algorithm settings contains the splitpoint, C1 as dict and C2 as dict
    """
    parser = argparse.ArgumentParser(description="Run basic single switch CMA-ES configuration",
                                     argument_default=argparse.SUPPRESS)

    
    # First, extract dim and include_lr from configuration_paramters
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--dim', dest='dim', type=int, required=True)
    temp_parser.add_argument('--use_learning_rates', dest='use_learning_rates', type=bool, required=True, default=False)
    temp_args, _ = temp_parser.parse_known_args(configuration_paramters)
    dim = temp_args.dim
    include_lr = temp_args.use_learning_rates
    cs = create_search_space(dim, include_lr)

    # Positional paramters
    parser.add_argument('configuration_id', type=str)
    parser.add_argument('instance_name', type=str)
    parser.add_argument('seed', type=int)
    parser.add_argument('iid', type=int)
    parser.add_argument('budget', type=float)

    # 'global' parameters
    parser.add_argument('--fid', dest='fid', type=int, required=True)
    parser.add_argument('--dim', dest='dim', type=int, required=True)
    parser.add_argument('--use_learning_rates', dest='use_learning_rates', type=bool, required=True, default=False)

    param_names = []
    for hp in list(cs.values()):
        if hasattr(hp, "choices"):
            parser.add_argument(f"--{hp.name}", dest=f"{hp.name}", type=str)
        elif hasattr(hp, "lower"):
            if "Integer" in type(hp).__name__:
                parser.add_argument(f"--{hp.name}", dest=f"{hp.name}", type=int)
            else:
                parser.add_argument(f"--{hp.name}", dest=f"{hp.name}", type=float)
        param_names.append(hp.name)

    # Process into dicts
    argsdict = parser.parse_args(configuration_paramters).__dict__
    alg_config = {}
    for k, v in argsdict.items():
        if v in ["False", "True", "None"]:
            v = eval(v)
        if k in param_names:
            alg_config[k] = v
    problem_config = {k: argsdict[k] for k in ('fid', 'dim', 'iid', 'seed', 'budget')}
    return [problem_config, alg_config]

if __name__ == "__main__":
    if sys.argv[1] == "--generate_parameters":
        use_LR = sys.argv[3]=='1'
        cs = create_search_space(int(sys.argv[2]), use_LR)
        configspace_to_irace_params_txt(cs, sys.argv[5], int(sys.argv[2]), use_LR, int(sys.argv[4]))
        sys.exit(0)
    else:
        problem_config, alg_config = parse_paramters(sys.argv[1:])
        
        result = get_bbob_performance(alg_config, f"({problem_config['fid']}, {problem_config['iid']})", problem_config['seed'], DIM=problem_config['dim'])
        print(result[0])
    if False:
        fids = list(range(1, 25))
        fids.append('all')
        dims = [2,3,5,10]
        lrs = [True, False]
        args = list(product(fids, dims, lrs))
        np.random.shuffle(args)
        runParallelFunction(run_smac, args)
    