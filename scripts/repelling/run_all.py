import os
import time
import subprocess
from itertools import product

ELITISM = [True, False]
STRATS = [0, 1, 2]
COVERAGE = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

cec_parameters = list(product(list(range(1104, 1121)), [1], ELITISM, COVERAGE, STRATS))
bbob_parameters = list(
    product(
        list(range(1, 25)), [2, 3, 4, 5, 6, 7, 8, 9, 10, 20], ELITISM, COVERAGE, STRATS
    )
)
parameters = cec_parameters

n_proc = len(parameters)
print(f"Spawning {n_proc} processess 2s to stop")
time.sleep(2)

for i, (fid, dim, el, cov, strat) in enumerate(parameters):
    if i > 0  and i % 10 == 0:
        time.sleep(2)
        load1, load5, load15 = os.getloadavg()
        while load1 > 225:
            print(" load too high, sleeping 10 seconds...", end="\r", flush=True)
            time.sleep(10)
            load1, load5, load15 = os.getloadavg()
            print(f" proc {i}/{n_proc}, load (1, 5, 15): ({load1, load5, load15})", end=" ")

    command = [
            "python",
            "repelling/collect_data.py",
            "--fid",
            str(fid),
            "--dim",
            str(dim),
            "--logged",
            "--coverage",
            str(cov),
            "--strat",
            str(strat)
    ]
    if el:
        command += ["--elitist"]
    subprocess.Popen(command,
        start_new_session=True,
    )
    print(f"proc {i}/{n_proc}: " + " ".join(command))
