from argparse import ArgumentParser

from .configurablecmaes import ConfigurableCMAES
from .utils import evaluate


parser = ArgumentParser(
    description='Run single function MAB exp iid 1')
parser.add_argument(
    '-f', "--fid", type=int,
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
parser.add_argument(
    '-l', '--logging', required=False,
    action='store_true', default=False
)
parser.add_argument(
    '-L', '--label', type=str, required=False,
    default=""
)
parser.add_argument(
    "-s", "--seed", type=int, required=False,
    default=42
)

evaluate(ConfigurableCMAES, **vars(parser.parse_args()))
