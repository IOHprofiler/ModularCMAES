# Reproducing the paper experiments

This document describes the maintained command-line workflow for the experiments
in *The Modular CMA-ES: A Framework for Modern Evolution Strategies*.

The notebooks in `scripts/matrix/` and `scripts/tuning/` are retained as
analysis records. The commands below are the supported reproduction entry
points and do not depend on notebook kernel state or machine-specific paths.

## Environment

A Linux environment, a C++17 compiler, and Python 3.10 or newer are
recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -r requirements-experiments.txt
```

The original environment was not fully locked. `requirements-experiments.txt`
records compatible dependency ranges and pins Pycma to the version stated in
the paper.

The timing scripts set common BLAS/OpenMP thread variables to one unless those
variables have already been set by the caller.

## Data layout

The maintained scripts use paths relative to the repository, independent of the
current working directory:

```text
data/                                      # matrix BBOB IOH data
scripts/matrix/time_stats.csv              # ModCMA timing data
scripts/matrix/time_stats_pycma.csv        # Pycma timing data
scripts/tuning/data/                       # SMAC run histories
scripts/tuning/ERT_BBOB.csv                # derived BBOB-2009 ERT reference
```

Each tuning function directory must contain exactly one run history:

```text
scripts/tuning/data/
└── BBOB_F<fid>_5D_LRFalseTrue/
    └── <scenario>/
        └── <seed>/
            └── runhistory.json
```

Use a separate `--output_root` for each independent SMAC reproduction. The
analysis command rejects ambiguous directories containing multiple run
histories for one function.

For reference, a `data.zip` is included in the `/scripts/tuning/`folder, which contains the data to generate the plots from the paper.

## Matrix-adaptation runtime experiment

Generate both ModCMA and Pycma timing data:

```bash
python scripts/matrix/test_timing_modules.py
```

This performs 15 repetitions of 1,000 generations on BBOB `f2` for all
matrix-adaptation methods and the Pycma baseline.

To use the dimensions written in the paper, including 50 rather than the
historical CSV's 40:

```bash
python scripts/matrix/test_timing_modules.py \
    --dims 2 3 5 10 20 50 100 200 500 1000
```

Create the runtime figure:

```bash
python scripts/matrix/plot_results.py --figure runtime
```

The plot reports the median over 15 runs. Error bars show the interquartile
range. The right panel reports the median time per adaptation update.

## Matrix-adaptation BBOB experiment

The archived generator defaults reproduce the protocol encoded by the
historical code: dimensions 2, 3, 5, 10, 20, and 40; 100 repetitions; target
precision `1e-8`; and a budget of `100000 * d`.

```bash
python scripts/matrix/get_data.py --workers 6
```

The paper instead states a target precision of `1e-9` and a budget of
`10000 * d`. To run that written protocol explicitly:

```bash
python scripts/matrix/get_data.py \
    --target 1e-9 \
    --budget-per-dimension 10000 \
    --workers 6
```

Use `--implementation modcma` or `--implementation pycma` to run only one
part of the experiment. Existing IOH output directories should be moved or
removed before a fresh run so that repetitions from different protocols are
not mixed.

Create the 24-function ERT figure from a complete archived `data/` directory:

```bash
python scripts/matrix/plot_results.py --figure bbob
```

The plotting target must match the stopping target used to generate the data.
Use `--target 1e-9` only for data generated with the written paper protocol.

## SMAC configuration experiment

Run SMAC independently for all 24 BBOB functions:

```bash
for fid in $(seq 1 24); do
    python scripts/tuning/run_smac.py \
        --fid "$fid" \
        --dim 5 \
        --add_popsize \
        --n_workers 1
done
```

The defaults match the tuning setup described in the paper:

- 50,000 SMAC target-function evaluations per BBOB function;
- at most 25 evaluations of one configuration;
- population sizes enabled;
- learning rates and initial sigma excluded;
- seed 2062;
- one worker.

The main resource controls are exposed as `--n_trials`,
`--max_config_calls`, `--n_workers`, `--seed`, and `--output_root`.
Increasing the worker count can change the exact order in which SMAC observes
results.

## Validation, ERT, and SHAP analysis

After all 24 SMAC run histories are present, run:

```bash
python scripts/tuning/analyze.py
```

The command:

1. selects configurations evaluated at least 25 times;
2. validates the five best-AOCC configurations per function on 50 held-out
   BBOB instances (IDs 100--149);
3. selects the candidate with the lowest validation ERT;
4. evaluates the default ModCMA configuration on the same instances;
5. creates the main and appendix ERT figures;
6. fits a CatBoost model for each function and computes TreeSHAP values; and
7. creates the global and per-module SHAP figures.

Use `--skip-shap` to create only the configuration and ERT outputs. Validation
and SHAP are computationally expensive.

The generated files are written to `scripts/tuning/` by default:

```text
configs_selected.csv
configs_all.csv
ert_heatmap_small.pdf
ert_heatmap_all.pdf
median_shap.pdf
<module>_shap_heat.pdf
```

Use `--output-dir` to keep newly generated artifacts separate from archived
ones.

`ERT_BBOB.csv` contains the derived ERT values for the 30 BBOB-2009 reference
algorithms. The repository does not contain the original COCO download and
conversion pipeline, so this CSV is the reproduction input for the reference
rows.

## Quick smoke tests

Small runs can verify the installation and command wiring without launching the
full experiments:

```bash
python scripts/matrix/test_timing_modules.py \
    --implementation modcma --dims 2 --repeats 1 --generations 2 \
    --output-dir /tmp/modcma-timing-smoke

python scripts/matrix/get_data.py \
    --implementation modcma --functions 1 --dims 2 --repeats 1 \
    --budget-per-dimension 20 --root /tmp/modcma-bbob-smoke

python scripts/tuning/run_smac.py --help
python scripts/tuning/analyze.py --help
```

## Historical artifacts

- `scripts/matrix/plots.ipynb` contains the original interactive plotting
  workflow.
- `scripts/tuning/analysis.ipynb` contains the original final tuning and SHAP
  analysis.
- `scripts/tuning/plotting.ipynb` is an older irace-based analysis and is not
  part of the final SMAC pipeline.
- `scripts/matrix/test_timing.py`, `scripts/matrix/test_bbob5d.py`,
  `scripts/tuning/test_config.py`, `scripts/tuning/smac_info.py`, and
  `scripts/tuning/cmd.sh` are exploratory or diagnostic helpers.

The standalone commands above supersede these artifacts for clean
reproduction, while the notebooks remain available for provenance.
