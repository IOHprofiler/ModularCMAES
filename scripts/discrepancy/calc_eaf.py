import os
import glob
import sqlite3
import warnings
from functools import partial
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = os.path.join(ROOT, "data") 
N_CPU = 8
EAF_MIN_LOG_TGT = -8
SQL = True
BUDGET = 10_000

def run_parallel_function(function, arguments):
    arguments = list(arguments)
    p = Pool(min(N_CPU, len(arguments)))
    results = p.map(function, arguments)
    p.close()
    return results

def get_eaf_table(fname, tgt):
    print(fname)
    dt = pd.read_csv(fname, sep=' ', decimal=',')
    dt = dt[dt['raw_y'] != 'raw_y'].astype(float) + 1e-20
    if dt['raw_y'][0] < 0:
        dt['raw_y'] *= -1
        
    dt['eaf'] = (2 - np.clip(np.log10(dt['raw_y'][:-1]), EAF_MIN_LOG_TGT, 2)) / 10
    dt = dt[~pd.isna(dt['eaf'])] 
    
    dt['run_id'] = np.cumsum(dt['evaluations'] == 1)
    dt['fid'] = fid = int(fname.split('/')[-1].split('_')[1][1:])
    dt['dim'] = dim = int(fname.split('/')[-1].split('_')[-1][3:-4])
    dt['version'] = version = fname.split('/')[-3].split('_')[-1]
    dt['evaluations'] = dt['evaluations'].astype(int)
    
    dt['sampler'] = version.split("-")[2]
    dt['orthogonal'] = 'orthogonal' in version
    dt['mirrored'] = 'mirror' in version
    dt['cache_size'] = int(version.split("cache-")[-1].split('-')[0]) if "cache" in version else 0
    
    max_budget = BUDGET * dim
    budget_eaf = []
    budgets = np.array(sorted(list(set((10 ** np.arange(0, (np.log(max_budget) / np.log(10)) + 0.1, 0.1)).astype(int))))[1:])
    for budget in budgets:
        dt_temp = (
            dt
            .query(f"evaluations < {budget}")
            .groupby(["run_id", "fid", "dim", "version", "sampler", "orthogonal", "mirrored", "cache_size"])
            .agg("max")["eaf"]
            .reset_index()
        )
        dt_temp['budget'] = budget
        budget_eaf.append(dt_temp)
        
    budget_eaf = pd.concat(budget_eaf)
    
    if not SQL:
        filename = f"{version}_{fid}_{dim}.csv"
        budget_eaf.to_csv(f"{tgt}/{filename}")
        return 
    
    con = sqlite3.connect(f"{tgt}/eaf.db", timeout=20)
    budget_eaf.to_sql(name='eaf', con=con, if_exists='append')


def find_files(src):
    files = glob.glob(f"{src}/*OPT-cache-64*/*/IOHprofiler_f*5.dat")
    return files

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder-name', type=str, default='eaf')
    parser.add_argument('--root_folder', type=str, default=DATA_PATH)
    args = parser.parse_args()
    
    tgt = os.path.join(DATA_PATH, args.folder_name)
    src = args.root_folder

    assert os.path.exists(src), f"{src} does not exist"
    os.makedirs(tgt, exist_ok=True)
    
    # if os.path.exists(f"{tgt}/eaf.db") and SQL == True:
    #     os.remove(f"{tgt}/eaf.db")
        
    files = find_files(src)
    auc_func = partial(get_eaf_table, tgt=tgt)
    results = run_parallel_function(auc_func, files)
        