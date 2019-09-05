from argparse import ArgumentParser
import numpy as np

from ConfigurableCMA import ConfigurableCMA
from CannonicalCMA import CannonicalCMA
from Utils import evaluate


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
    args = parser.parse_args()
    np.random.seed(42)
    # print("Modular: rewrite (old order)")
    # evaluate(args.functionid, args.dim, ConfigurableCMA,
    #          iterations=args.iterations, old_order=True,
    #          label="new_old_order", logging=True
    #          )

    print("Modular: rewrite (correct order)")
    evaluate(args.functionid, args.dim, ConfigurableCMA,
             iterations=args.iterations,
             label="new_corect_order", logging=True
             )

    # import subprocess
    # print("\nModular: old")
    # subprocess.run(
    #     "/home/jacob/Documents/thesis/.env/bin/python "
    #     f"/home/jacob/Documents/thesis/OnlineCMA-ES/src/main.py "
    #     f"-f {args.functionid} -d {args.dim} -i {args.iterations} --clear", shell=True
    # )
    # print("\nCannonical CMA")
    # evaluate(args.functionid, args.dim, CannonicalCMA,
    #          iterations=args.iterations,
    #          label="cma_no_eigendecomp", logging=True
    #          )
