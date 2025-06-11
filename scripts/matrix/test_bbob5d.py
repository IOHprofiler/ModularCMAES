from time import perf_counter
import ioh
import modcma.c_maes as modcma 
import iohinspector as ins
import matplotlib.colors as mcolors
import numpy as np


def timeit(f):
    def inner(*args, **kwargs):
        start = perf_counter()
        result = f(*args, **kwargs)
        stop = perf_counter()
        elapsed = stop - start
        return elapsed
    return inner


@timeit
def run_modma(f: ioh.ProblemType, dim: int, n_evaluations):
    modules = modcma.parameters.Modules()
    # modules.restart_strategy = modcma.options.RestartStrategy.IPOP
    # modules.active = True
    settings = modcma.Settings(
        dim, 
        budget=n_evaluations, 
        target=f.optimum.y + 1e-8,
        lb=f.bounds.lb,
        ub=f.bounds.ub,
        sigma0=2.0, 
        modules=modules, 
        verbose=False
    )
    cma = modcma.ModularCMAES(settings)
    cma.run(f)
    return cma

def fix_legend_labels(ax, n_split, algs, groupby_word = None, reorder=False):
    colors = dict(zip(algs, mcolors.TABLEAU_COLORS))
    lines = ax.get_lines()[::]
    if reorder:
        lines = lines[::2] + lines[1::2]
        
    for line, line_label in zip(lines[:len(lines)//2], lines[len(lines)//2:]):
        if (lab:=line_label.get_label()) in colors:
            for l in (line, line_label):
                l.set_color(colors[lab])
                l.set_linewidth(3)  
                if groupby_word is not None and groupby_word in lab:
                    l.set_linestyle('dashed')  
                else:
                    l.set_linestyle('solid')  
            
    handles, labels = ax.get_legend_handles_labels()
    labels = [l[n_split:] for l in labels[:]]
    idx = np.argsort(labels)
    ax.legend(np.array(handles)[idx], np.array(labels)[idx], fancybox=True, shadow=True, fontsize=13)
    return handles, labels

def place_legend_below(ax, handles, labels, show = True, legend_nrow = 1, start_legend = 3, loc_y = -.11):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
            box.width, box.height * 0.9])
    
    ax.legend().remove()
    if show:
        ax.legend(np.array(handles), np.array(labels), loc='upper center',
            fontsize=13, bbox_to_anchor=(start_legend, loc_y), fancybox=True, shadow=True, ncol=np.ceil(len(labels) / legend_nrow), 
        )
        

if __name__ == "__main__":
    # modcma.utils.set_seed(43)
    # modcma.constants.calc_eigv = True
    # name = f"CMA-ES eig={modcma.constants.calc_eigv}"

    # logger = ioh.logger.Analyzer(
    #     folder_name=name, 
    #     algorithm_name=name, 
    #     root="data"
    # )

    # dim = 5
    # n_rep = 5
    # n_instances = 5

    # budget = 50_000 * dim
    # for i in range(1, 25):
    #     for ii in range(1, n_instances + 1):
    #         problem = ioh.get_problem(i, ii, dim)
    #         problem.attach_logger(logger)
    #         for r in range(n_rep):
    #             run_modma(problem, dim, budget)
    #             print(problem.state.evaluations, problem.state.current_best_internal.y)
    #             problem.reset()

    import os
    manager = ins.DataManager()
    algs = []
    for folder in os.listdir("data"):
        algs.append(folder)
        manager.add_folder(f"data/{folder}")



    import matplotlib.pyplot as plt

    f, axes = plt.subplots(5, 5, figsize=(25, 13), sharex=True, sharey=True)

    x_values = ins.get_sequence(1, 50_000 * 5, 50, True, True)
    for fid, ax in enumerate(axes.ravel(), 1):
        if fid > 24:
            break
        dt = manager.select(function_ids=[fid]).load(True, True, x_values=x_values)
        ins.plot.single_function_fixedbudget(data=dt, ax=ax)
        h,l = fix_legend_labels(ax, 1, algs, None)
        place_legend_below(ax, h, l, fid == 24, 1)

    plt.tight_layout()
    plt.show()