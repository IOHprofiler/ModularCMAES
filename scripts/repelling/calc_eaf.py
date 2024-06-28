import numpy as np
import pandas as pd
import glob
from functools import partial
from multiprocessing import Pool, cpu_count

import warnings
import os

root = os.path.realpath(os.path.dirname(__file__))
src = os.path.join(root, "data/repelling/ioh_fixed")
tgt = os.path.join(root, "data/auc_repelling")

def run_parallel_function(runFunction, arguments):
    arguments = list(arguments)
    p = Pool(min(32, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results

def get_auc_table(fname):
    print(fname)
    dt = pd.read_csv(fname, sep=' ', decimal=',')
    dt = dt[dt['raw_y'] != 'raw_y'].astype(float)
    if dt['raw_y'][0] < 0:
        dt['raw_y'] *= -1
    dt['run_id'] = np.cumsum(dt['evaluations'] == 1)
    dt['ecdf'] = (2 - np.clip(np.log10(dt['raw_y'][:-1]), -8, 2))/10
    
    dt['version'] = fname.split('/')[-3].split('_')[-1]
    dt['fid'] = int(fname.split('/')[-1].split('_')[1][1:])
    dt['dim'] = int(fname.split('/')[-1].split('_')[-1][3:-4])
    
    dt.to_csv(f"{tgt}/{fname.split('/')[-3].split('_')[-1]}_{fname.split('/')[-1].split('_')[1][1:]}_{fname.split('/')[-1].split('_')[-1][3:-4]}.csv")


def find_files():
    files_orig = glob.glob(f"{src}/*/*/IOHprofiler_f*.dat")
    return files_orig

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not os.path.isdir(tgt):
        os.makedirs(tgt)
        
    files = find_files()
    results = run_parallel_function(get_auc_table, files)
        
        
        
        
        
        
        