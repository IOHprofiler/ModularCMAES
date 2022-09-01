"""Script to re-initialize the expected function values in expected.json."""
import os
import json

import ioh
import numpy as np


from modcma import ModularCMAES, parameters, utils


def run_bbob_function(module, value, fid):
    """Runs the specified version of ModularCMAES on the bbob-function."""
    np.random.seed(42)
    function = ioh.get_problem(fid, dimension=2, instance=1)
    p = parameters.Parameters(2, budget=20, **{module: value})
    ModularCMAES(function, parameters=p).run()
    return function.state.current_best_internal.y


def create_expected_dict():
    """Creates the dictionary containing the expected final function values."""
    bbob_2d_per_module = dict()
    for module in parameters.Parameters.__modules__:
        m = getattr(parameters.Parameters, module)
        if type(m) == utils.AnyOf:
            for o in filter(None, m.options):
                bbob_2d_per_module[f"{module}_{o}"] = [0] * 24
                for fid in range(1, 25):
                    bbob_2d_per_module[f"{module}_{o}"][
                        fid - 1
                    ] = run_bbob_function(module, o, fid)

        elif type(m) == utils.InstanceOf:
            bbob_2d_per_module[f"{module}_{True}"] = [0] * 24
            for fid in range(1, 25):
                bbob_2d_per_module[f"{module}_{True}"][
                    fid - 1
                ] = run_bbob_function(module, True, fid)
    return bbob_2d_per_module


if __name__ == "__main__":
    directory = os.path.realpath(os.path.dirname(__file__))
    data = create_expected_dict()
    
    with open(os.path.join(directory, "expected.json"), "w") as f:
        json.dump(data, f)
