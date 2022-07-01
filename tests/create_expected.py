"""Script to re-initialize the expected function values in expected.json."""

from modcma import ModularCMAES, parameters, utils
import ioh
import numpy as np
import json

def run_bbob_function(module, value, fid):
        """Runs the specified version of ModularCMAES on the bbob-function and returns its output."""
        np.random.seed(42)
        dim = 2
        budget = 20
        iid = 1
        f = ioh.get_problem(fid, dimension=dim, instance=iid)
        p = parameters.Parameters(
            dim, budget=budget, **{module: value}
        )
        ModularCMAES(f, parameters=p).run()
        return f.state.current_best_internal.y
        
def create_expected_dict():
    """Creates the dictionary containing the expected final function values."""
    BBOB_2D_PER_MODULE_20_ITER = dict()
    for module in parameters.Parameters.__modules__:
        m = getattr(parameters.Parameters, module)
        if type(m) == utils.AnyOf:
            for o in filter(None, m.options):
                BBOB_2D_PER_MODULE_20_ITER[f"{module}_{o}"] = np.zeros(24)
                for fid in range(1, 25):
                    BBOB_2D_PER_MODULE_20_ITER[f"{module}_{o}"][fid - 1] = run_bbob_function(module, o, fid)

        elif type(m) == utils.InstanceOf:
            BBOB_2D_PER_MODULE_20_ITER[f"{module}_{True}"] = np.zeros(24)
            for fid in range(1, 25):
                BBOB_2D_PER_MODULE_20_ITER[f"{module}_{True}"][fid - 1] = run_bbob_function(module, True, fid)
    return BBOB_2D_PER_MODULE_20_ITER

if __name__ == "__main__":
    BBOB_2D_PER_MODULE_20_ITER = create_expected_dict()
    with open("expected.json", "w") as f:
        json.dump(BBOB_2D_PER_MODULE_20_ITER, f)