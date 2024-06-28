Scripts:
    - collect_data.py
        Runs a single experiment (combination of fid, dim, strategy)
    - run_all.py 
        Runs all experiments.
    - merge_ioh_data.py
        Fixes the file structure of the ioh logs after run_all has been called.    
    - calc_auc.py 
        Calculates empirical attainment for each run. requires run_all.py, merge_ioh_data.py to be executed first.
    - repelling.py
        Collection of methods and functions used in project

Notebooks:
    - eaf.ipynb
        Creates EAF visualizations. Requires calc_auc.py to be executed first.
    - potential.ipynb
        Creates restart potential visualizations. requires run_all.py to be excecuted first.

Folders:
    - data/auc_repelling
        contains intermediate files used for EAF visualization
    - data/ioh
        contains intermediate ioh logs
    - data/ioh_fixed
        contains ioh logs
    - data/pkl
        contains restart data   


To install the modcma package:
    pip install -e . 