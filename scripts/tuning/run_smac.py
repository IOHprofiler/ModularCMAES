import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import time
from functools import partial
from pathlib import Path

import ioh
import numpy as np
from ConfigSpace import (
    AndConjunction,
    Configuration,
    ConfigurationSpace,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenEqualsRelation,
    ForbiddenGreaterThanRelation,
    ForbiddenInClause,
    GreaterThanCondition,
    InCondition,
)
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac import AlgorithmConfigurationFacade, Scenario
from smac.main.config_selector import ConfigSelector

from modcma import c_maes

DATA_DIR = Path(__file__).resolve().parent / "data"


def calc_aoc(problem: ioh.ProblemType, logger: ioh.logger.Store, budget: int) -> float:
    """
    Compute the Area Over the Curve (AOC) for an optimization run.

    The AOC summarizes optimization performance over time by averaging
    the log-scaled best-so-far objective values across a fixed evaluation
    budget. Lower values indicate better and faster convergence.

    Steps:
    - Extract best-so-far objective values ("raw_y_best") from the logger.
    - Replace NaNs with a large penalty value (1e8).
    - Pad the trajectory to the full budget using the best observed value.
    - Clip values to [1e-8, 1e2] and apply log10 scaling.
    - Shift values to [0, 10] and normalize to [0, 1].
    - Return the mean over the budget (the AOC score).

    Parameters
    ----------
    problem : ioh.ProblemType
        The evaluated problem
    logger : ioh.logger.Store
        IOH logger containing experiment data.
    budget : int
        Maximum number of function evaluations to consider.
    Returns
    -------
    float
        AOC score in [0, 1], where lower values indicate better performance.
    """

    data = logger.data()
    data1 = data["None"][problem.meta_data.problem_id][problem.meta_data.n_variables][
        problem.meta_data.instance
    ][0]
    fvals = [x["raw_y_best"] for x in data1.values()]
    fvals = np.array(fvals)
    if np.isnan(fvals).any():
        np.nan_to_num(fvals, copy=False, nan=1e8)
    if len(fvals) < budget:
        fvals = np.concatenate([fvals, (budget - len(fvals)) * [np.min(fvals)]])
    parts = np.log10(np.clip(fvals[:budget], 1e-8, 1e2)) + 8
    return np.mean(parts) / 10


def get_bbob_performance(
    config: Configuration, seed: int = 0, fid: int = 0, dim: int = 5
):
    iid = 1 + (seed % 10)
    np.random.seed(seed + iid)
    c_maes.utils.set_seed(seed + iid)
    BUDGET = dim * 10_000

    problem = ioh.get_problem(fid, iid, dim)
    logger = ioh.logger.Store(
        triggers=[ioh.logger.trigger.ALWAYS], properties=[ioh.logger.property.RAWYBEST]
    )
    problem.attach_logger(logger)
    settings = c_maes.settings_from_config(
        dim,
        config,
        budget=BUDGET,
        target=problem.optimum.y + 1e-8,
        ub=problem.bounds.ub,
        lb=problem.bounds.lb,
    )
    settings.modules.center_placement = c_maes.options.CenterPlacement.UNIFORM
    par = c_maes.Parameters(settings)

    try:
        cma = c_maes.ModularCMAES(par)
        cma.run(problem)
        aoc = calc_aoc(problem, logger, BUDGET)
    # Convert any optimizer failure to a penalized SMAC observation so one
    # invalid configuration cannot abort the entire tuning campaign.
    except Exception as error:  # noqa: BLE001
        print(f"Optimizer evaluation failed at {problem.state.current_best.y}: {error}")
        aoc = np.inf

    extra = {
        "fid": fid,
        "iid": iid,
        "dim": dim,
        "target": float(problem.optimum.y + 1e-8),
        "final_y": float(problem.state.current_best.y),
        "evals": int(problem.state.evaluations),
        "hit_target": bool(problem.state.current_best.y <= problem.optimum.y + 1e-8),
        "precision": float(abs(problem.state.current_best.y - problem.optimum.y)),
    }
    return aoc, extra


def make_new(hp: CategoricalHyperparameter, excluded: list[str]):
    new_choices = [choice for choice in hp.choices if choice not in excluded]
    return CategoricalHyperparameter(
        name=hp.name, choices=new_choices, default_value=hp.default_value
    )


# def get_configspace(dim, use_learning_rates, add_popsize, add_sigma):
#     cma_cs = c_maes.get_configspace(
#         dim,
#         add_learning_rates=use_learning_rates,
#         add_popsize=add_popsize,
#         add_sigma=add_sigma,
#     )
#     cs = ConfigurationSpace()
#     for hp in cma_cs.values():
#         if hp.name in ("sample_sigma", "bound_correction", "center_placement"):
#             continue

#         if hp.name == "matrix_adaptation":
#             cs.add(make_new(hp, ("COVARIANCE_NO_EIGV",)))
#             continue

#         if hp.name == "repelling_type":
#             cs.add(make_new(hp, ("ADAPTIVE",)))
#             continue

#         if hp.name == "restart_strategy":
#             cs.add(make_new(hp, ("STOP",)))
#             continue

#         if hp.name == "sample_transformation":
#             cs.add(make_new(hp, ("NONE",)))
#             continue

#         cs.add(hp)

#     if add_popsize:
#         cs.add(ForbiddenGreaterThanRelation(cs["mu0"], cs["lambda0"]))
#     return cs


def get_configspace(
    dim: int,
    use_learning_rates: bool,
    add_popsize: bool,
    add_sigma: bool,
) -> ConfigurationSpace:
    """
    Construct the SMAC configuration space while removing configurations
    that are invalid, silently rewritten, or algorithmically redundant.
    """
    cma_cs = c_maes.get_configspace(
        dim,
        add_learning_rates=use_learning_rates,
        add_popsize=add_popsize,
        add_sigma=add_sigma,
    )

    cs = ConfigurationSpace()

    # ------------------------------------------------------------------
    # Add the hyperparameters included in this specific SMAC experiment.
    # ------------------------------------------------------------------
    for hp in cma_cs.values():
        if hp.name in (
            "sample_sigma",
            "bound_correction",
            "center_placement",
        ):
            continue

        if hp.name == "matrix_adaptation":
            cs.add(
                make_new(
                    hp,
                    ["COVARIANCE_NO_EIGV"],
                )
            )
            continue

        if hp.name == "repelling_type":
            cs.add(
                make_new(
                    hp,
                    ["ADAPTIVE"],
                )
            )
            continue

        if hp.name == "restart_strategy":
            cs.add(
                make_new(
                    hp,
                    ["STOP"],
                )
            )
            continue

        if hp.name == "sample_transformation":
            cs.add(
                make_new(
                    hp,
                    ["NONE"],
                )
            )
            continue

        if hp.name == "sampler":
            # TESTER produces deterministic incrementing vectors and is
            # intended for unit testing, not algorithm configuration.
            cs.add(
                make_new(
                    hp,
                    ["TESTER"],
                )
            )
            continue

        cs.add(hp)

    # ------------------------------------------------------------------
    # Conditions: parameters that only exist in part of the search space.
    # ------------------------------------------------------------------
    conditions = []

    # Repelling only has meaning if a restart can actually be performed.
    conditions.append(
        InCondition(
            child=cs["repelling_type"],
            parent=cs["restart_strategy"],
            values=[
                "RESTART",
                "IPOP",
                "BIPOP",
            ],
        )
    )

    # Active adaptation uses the negative recombination weights. It is
    # ignored by NONE and NATURAL_GRADIENT.
    active_matrix_adaptations = [
        "COVARIANCE",
        "MATRIX",
        "SEPARABLE",
        "CHOLESKY",
        "CMSA",
        "NATURAL_GRADIENT",
    ]

    if add_popsize:
        conditions.append(
            AndConjunction(
                InCondition(
                    child=cs["active"],
                    parent=cs["matrix_adaptation"],
                    values=active_matrix_adaptations,
                ),
                GreaterThanCondition(
                    child=cs["active"],
                    parent=cs["lambda0"],
                    value=1,
                ),
            )
        )
    else:
        conditions.append(
            InCondition(
                child=cs["active"],
                parent=cs["matrix_adaptation"],
                values=active_matrix_adaptations,
            )
        )

    if add_popsize:
        # Settings overrides these when lambda0 == 1:
        #
        # mu0                  = 1
        # elitist              = True
        # sequential_selection = False
        # weights              = EQUAL
        # ssa                  = SR
        #
        # They should consequently not be separate SMAC decisions in the
        # one-plus-one branch.
        one_plus_one_inactive = [
            "mu0",
            "elitist",
            "sequential_selection",
            "weights",
            "ssa",
        ]

        for name in one_plus_one_inactive:
            conditions.append(
                GreaterThanCondition(
                    child=cs[name],
                    parent=cs["lambda0"],
                    value=1,
                )
            )

    if use_learning_rates:
        # damps is used by CSA and SR in the current mutation code.
        conditions.append(
            InCondition(
                child=cs["damps"],
                parent=cs["ssa"],
                values=[
                    "CSA",
                    "SR",
                ],
            )
        )

        # c1 and cmu control the matrix update for these adaptations.
        c1_cmu_adaptations = [
            "COVARIANCE",
            "MATRIX",
            "SEPARABLE",
            "CHOLESKY",
        ]

        conditions.extend(
            [
                InCondition(
                    child=cs["c1"],
                    parent=cs["matrix_adaptation"],
                    values=c1_cmu_adaptations,
                ),
                InCondition(
                    child=cs["cmu"],
                    parent=cs["matrix_adaptation"],
                    values=c1_cmu_adaptations,
                ),
            ]
        )

        # cc controls covariance/path updates for these adaptations.
        cc_adaptations = [
            "COVARIANCE",
            "SEPARABLE",
            "CHOLESKY",
            "NATURAL_GRADIENT",
        ]

        conditions.append(
            InCondition(
                child=cs["cc"],
                parent=cs["matrix_adaptation"],
                values=cc_adaptations,
            )
        )

        # cs is intentionally left unconditional. It is used by many SSA
        # rules and by several matrix-adaptation evolution paths.

    cs.add(conditions)

    # ------------------------------------------------------------------
    # Forbidden clauses: configurations that should never be sampled.
    # ------------------------------------------------------------------
    forbiddens = []

    if add_popsize:
        # Basic feasibility.
        forbiddens.append(
            ForbiddenGreaterThanRelation(
                cs["mu0"],
                cs["lambda0"],
            )
        )

        # Settings changes every odd pairwise population size to the next
        # even value, so lambda=k and lambda=k+1 otherwise represent the
        # same realized algorithm.
        lambda_hp = cs["lambda0"]

        lower = int(lambda_hp.lower)
        upper = int(lambda_hp.upper)

        first_odd = lower if lower % 2 == 1 else lower + 1

        for lambda_value in range(first_odd, upper + 1, 2):
            forbiddens.append(
                ForbiddenAndConjunction(
                    ForbiddenEqualsClause(
                        cs["mirrored"],
                        "PAIRWISE",
                    ),
                    ForbiddenEqualsClause(
                        lambda_hp,
                        lambda_value,
                    ),
                )
            )

        # IPOP and BIPOP are silently changed to RESTART in the
        # one-plus-one case.
        forbiddens.append(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(
                    cs["lambda0"],
                    1,
                ),
                ForbiddenInClause(
                    cs["restart_strategy"],
                    [
                        "IPOP",
                        "BIPOP",
                    ],
                ),
            )
        )

        # With mu == lambda there are no unselected offspring and therefore
        # no negative weights for active adaptation to use.
        forbiddens.append(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(
                    cs["active"],
                    True,
                ),
                ForbiddenEqualsRelation(
                    cs["mu0"],
                    cs["lambda0"],
                ),
            )
        )

        # Sequential selection can only terminate early after at least mu
        # evaluations. With mu == lambda, it cannot save evaluations.
        forbiddens.append(
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(
                    cs["sequential_selection"],
                    True,
                ),
                ForbiddenEqualsRelation(
                    cs["mu0"],
                    cs["lambda0"],
                ),
            )
        )

    # COVARIANCE_NO_EIGV was removed from this search space. However,
    # Settings silently maps:
    #
    #   matrix_adaptation=COVARIANCE, ssa!=CSA
    #
    # to COVARIANCE_NO_EIGV unless always_compute_eigv=True.
    #
    # Without this clause, the supposedly removed option remains present
    # indirectly.
    non_csa_ssa = [
        "TPA",
        "MSR",
        "XNES",
        "MXNES",
        "LPXNES",
        "PSR",
        "SR",
        "SA",
    ]

    forbiddens.append(
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(
                cs["matrix_adaptation"],
                "COVARIANCE",
            ),
            ForbiddenInClause(
                cs["ssa"],
                non_csa_ssa,
            ),
        )
    )

    cs.add(forbiddens)

    return cs


def run_smac(
    fid,
    dim,
    use_learning_rates,
    add_popsize,
    add_sigma,
    n_workers,
    n_trials,
    max_config_calls,
    seed,
    output_root,
):
    print(f"Running SMAC with fid={fid}, lr={use_learning_rates} and d={dim}")
    cs = get_configspace(dim, use_learning_rates, add_popsize, add_sigma)
    scenario = Scenario(
        cs,
        name=str(int(time.time())) + "-" + "CMA",
        deterministic=False,
        n_trials=n_trials,
        output_directory=os.path.join(
            output_root,
            f"BBOB_F{fid}_{dim}D_LR{use_learning_rates}{add_popsize}",
        ),
        n_workers=n_workers,
        seed=seed,
    )

    eval_func = partial(get_bbob_performance, fid=fid, dim=dim)
    config_selector = ConfigSelector(
        scenario,
        retrain_after=100,
        min_trials=100,
        max_new_config_tries=16,
    )

    smac = AlgorithmConfigurationFacade(
        scenario,
        eval_func,
        intensifier=AlgorithmConfigurationFacade.get_intensifier(
            scenario, max_config_calls=max_config_calls
        ),
        config_selector=config_selector,
        initial_design=AlgorithmConfigurationFacade.get_initial_design(scenario),
        # model = AlgorithmConfigurationFacade.get_model(
        #     scenario,
        #     n_trees=7,
        #     ratio_features=0.5,
        #     min_samples_split=10,
        #     min_samples_leaf=5,
        #     max_depth=7,
        #     bootstrapping=True,
        #     pca_components=15
        # ),
        # acquisition_maximizer=LocalAndSortedRandomSearch(
        #     scenario.configspace,
        #     seed=scenario.seed,
        #     challengers=500
        # )
    )
    smac.optimize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=int, default=1)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--use_learning_rates", action="store_true")
    parser.add_argument("--add_popsize", action="store_true")
    parser.add_argument("--add_sigma", action="store_true")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=50_000)
    parser.add_argument(
        "--max_config_calls",
        type=int,
        default=25,
        help="Maximum evaluations of one configuration; 25 matches the paper.",
    )
    parser.add_argument("--seed", type=int, default=1993 + 69)
    parser.add_argument("--output_root", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if args.n_workers < 1:
        parser.error("--n_workers must be positive")
    if args.n_trials < 1:
        parser.error("--n_trials must be positive")
    if args.max_config_calls < 1:
        parser.error("--max_config_calls must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)

    run_smac(
        args.fid,
        args.dim,
        args.use_learning_rates,
        args.add_popsize,
        args.add_sigma,
        args.n_workers,
        args.n_trials,
        args.max_config_calls,
        args.seed,
        args.output_root,
    )
