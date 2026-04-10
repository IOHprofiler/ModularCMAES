import os
import shutil
import json
from pprint import pprint
from time import sleep
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np


def min_rt(config_rt):
    amin = float("inf")
    cid = None
    for k, values in config_rt.items():
        if len(values) < 10: continue
        if (kmin:= np.mean(values)) < amin:
            amin = kmin
            cid = k
    return cid, amin, config_rt[cid]
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fid", default=1, type=int)
    parser.add_argument("--dim", default=5, type=int)
    parser.add_argument("--root", default="data", type=str) 
    parser.add_argument("--show_all_feasible", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    folders = os.listdir(args.root)
    arg_folder = f"BBOB_F{args.fid}_{args.dim}D_LRFalse"
    
    for folder in sorted(os.listdir(args.root)):
        if not args.all and not folder.endswith(arg_folder): continue
        algf, *_ = os.listdir(os.path.join(args.root, folder))
        filename = f"./{args.root}/{folder}/{algf}/0/runhistory.json"
        print(folder, end=' - ')
        for i in range(10):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                break
            except Exception as err:
                print(err)
                sleep(.1)
        else:
            print("cannot load data")
            continue
        
        config_costs = defaultdict(list)
        config_rt = defaultdict(list)
        records = defaultdict(list)

        for config in data['data']:
            cost, cid = config['cost'], str(config['config_id'])
            records[cid].append(config)
            if cost != 1_000_000 and np.isfinite(cost):
                config_costs[cid].append(float(cost))
                rt = config['additional_info']['evals']
                solved = config['additional_info']['hit_target']
                config_rt[cid].append(rt if solved else 50_000)

        amin = float('inf')
        cmin = None
        for cid, values in config_costs.items():
            mvalue = np.mean(values)
            if args.show_all_feasible:
                print(cid, mvalue, len(values), data['configs'][cid])     
            if len(values) > 10 and mvalue < amin:
                amin = mvalue
                cmin = cid


        print(f"{len(data['data'])} configs evaluated")
        if cmin is None:
            print("No best solutions yet")
        else:
            print(f"lowest avg. cost ({cmin}): {amin: .6f}", end = ' - ')
            print(f"avg. rt: {np.mean(config_rt[cmin]): .2f}")
            pprint(data['configs'][cmin])

        # cid, m_rt, rts = min_rt(config_rt)
        # print(cid, m_rt, np.mean(config_costs[cid]))
        # pprint(records[cid])
        print()        
