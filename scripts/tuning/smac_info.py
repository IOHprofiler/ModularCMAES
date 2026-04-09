import os
import shutil
import json
from time import sleep
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fid", default=1, type=int)
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--root", default="smac", type=str)
    parser.add_argument("--show_all_feasible", action="store_true")
    args = parser.parse_args()
    
    folders = os.listdir(args.root)
    arg_folder = f"BBOB_f{args.fid}_{args.dim}D_LRFalse"
    for folder in sorted(os.listdir(args.root)):
        if not args.all and not folder.endswith(arg_folder): continue
        filename = f"./{args.root}/{folder}/0/runhistory.json"
        print(folder, end=' - ')
        for i in range(10):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                break
            except:
                sleep(.1)
        else:
            print("cannot load data")
            continue
        
        for key, value in data['config_origins'].items():
            if value == "Initial Design: Default configuration":
                def_id = key 
                break
        else:
            print("no default id")
            continue

        configs = defaultdict(list)
        
        for config in data['data']:
            cost, cid = config['cost'], str(config['config_id'])
            if cost != 1_000_000 or cid == def_id and np.isfinite(cost):
                configs[cid].append(float(cost))

        amin = float('inf')
        cmin = None
        for cid, values in configs.items():
            mvalue = round(np.mean(values), 1)
            if args.show_all_feasible:
                print(cid, mvalue, len(values), data['configs'][cid])     
            if len(values) > 20 and mvalue < amin:
                amin = mvalue
                cmin = cid
        print(f"{len(data['data'])} configs evaluated")
        # if len(data['data']) != 10_000:
        print(f"default solution ({def_id}):",
            np.mean(configs[def_id]), 
            data['configs'][def_id], 
            len(configs[def_id])
        )
        if cmin is None:
            print("No best solutions yet")
        else:
            print(f"lowest avg. cost ({cmin}):", amin, data['configs'][cmin])     
        print()        
