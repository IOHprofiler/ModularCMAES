<!-- # ModularCMAES  -->
<p align="center">
  <img src="banner.png" alt="Modular CMA-ES Banner"/>
</p>

<hr>
 
![Unittest](https://github.com/IOHprofiler/ModularCMAES/workflows/Unittest/badge.svg) 
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e25b2d338c194d67954fc9e138ca69cc)](https://app.codacy.com/gh/IOHprofiler/ModularCMAES?utm_source=github.com&utm_medium=referral&utm_content=IOHprofiler/ModularCMAES&utm_campaign=Badge_Grade) 
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/73720e228a89480585bdde05d3806661)](https://www.codacy.com/gh/IOHprofiler/ModularCMAES/dashboard?utm_source=github.com&utm_medium=referral&utm_content=IOHprofiler/ModularCMAES&utm_campaign=Badge_Coverage)
![PyPI - Version](https://img.shields.io/pypi/v/modcma)
![PyPI - Downloads](https://img.shields.io/pypi/dm/modcma)

The **Modular CMA-ES** is a Python and C++ package that provides a modular implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm. This package allows you to create various algorithmic variants of CMA-ES by enabling or disabling different modules, offering flexibility and customization in evolutionary optimization. In addition to the CMA-ES, the library includes an implementation of the Matrix Adaptation Evolution Strategy (MA-ES) algorithm, which has similar emprical performance on most problems, but signifanctly lower runtime. All modules implemented are compatible with both the CMA-ES and MA-ES.

This implementation is based on the algorithm introduced in the paper "[Evolving the Structure of Evolution Strategies. (2016)](https://ieeexplore.ieee.org/document/7850138)" by Sander van Rijn et. al. If you would like to cite this work in your research, please cite the paper: "[Tuning as a Means of Assessing the Benefits of New Ideas in Interplay with Existing Algorithmic Modules (2021)](https://doi.org/10.1145/3449726.3463167)" by Jacob de Nobel, Diederick Vermetten, Hao Wang, Carola Doerr and Thomas Bäck.

This README provides a high level overview of the implemented modules, and provides some usage examples for both the Python-only and the C++-based versions of the framework. 

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation ](#installation-)
  - [Python Installation](#python-installation)
  - [Installation from source](#installation-from-source)
- [Usage ](#usage-)
  - [C++ Backend ](#c-backend-)
    - [High Level interface ](#high-level-interface-)
    - [Low Level interface ](#low-level-interface-)
  - [Tuning ](#tuning-)
    - [Configuration Space Generation](#configuration-space-generation)
    - [Creating Settings from a Configuration](#creating-settings-from-a-configuration)
  - [Integer Variables](#integer-variables)
  - [Python-only (Legacy) ](#python-only-legacy-)
    - [Ask–Tell Interface ](#asktell-interface-)
- [Modules ](#modules-)
  - [Matrix Adaptation ](#matrix-adaptation-)
  - [Active Update ](#active-update-)
  - [Elitism ](#elitism-)
  - [Orthogonal Sampling ](#orthogonal-sampling-)
  - [Sequential Selection ](#sequential-selection-)
  - [Threshold Convergence ](#threshold-convergence-)
  - [Sample Sigma ](#sample-sigma-)
  - [Quasi-Gaussian Sampling ](#quasi-gaussian-sampling-)
  - [Recombination Weights ](#recombination-weights-)
  - [Mirrored Sampling ](#mirrored-sampling-)
  - [Step size adaptation](#step-size-adaptation)
  - [Restart Strategy](#restart-strategy)
  - [Bound correction](#bound-correction)
  - [Sample Transformer ](#sample-transformer-)
  - [Repelling Restart ](#repelling-restart-)
  - [Center Placement ](#center-placement-)
- [Citation ](#citation-)
- [License ](#license-)

## Installation <a name="installation"></a>

You can install the Modular CMA-ES package using `pip`.

### Python Installation

```bash
pip install modcma
```

### Installation from source

If you want to work on a development version of the library, you should follow the following steps. A C++ compiler is required, and the following is valid for g++ (v11.1.0):

1. Clone the repository:

   ```bash
   git clone git@github.com:IOHprofiler/ModularCMAES.git
   cd ModularCMAES
   ```

2. Install dependencies (in a virtual environment)

   ```bash
   python3 -m venv env
   source ./env/bin/activate
   pip install -r requirements.txt   
   ```

3. Compile the library, we can optionally install the package globally:

   ```bash
   python setup.py install
   ```

   or install in develop mode, which is recommended if one would like to actively develop the library:

   ```bash
   python setup.py develop 
   ```

4. Ensure all functionality works as expected by running the unittests:

   ```bash
   python -m unittest discover   
   ```

## Usage <a name="usage"></a>

> **Note:**  
> The **C++ backend** is the primary and recommended interface for ModularCMAES.  
> The **pure Python implementation** remains available for reference and educational purposes,  
> but is **no longer actively developed** and may not include all new module options or performance improvements.

### C++ Backend <a name="c-backend"></a>

For performance, completeness, and modularity, the **C++ backend** is the **recommended interface** for ModularCMAES. It exposes all available modules and options through a direct Python binding.

#### High Level interface <a name="high-level-interface"></a>

In addition to the fully specified method described below, we can run an optimization via a friendly `fmin` interface:

```python
x0 = [0, 1, 2, 3] # Location to start the search from
sigma0 = 0.234    # Initial estimate of the stepsize, try 0.3 * (ub - lb) if you're unsure
budget = 100      # Total number of function evaluations
xopt, fopt, evals, cma = c_maes.fmin(
    func, x0, sigma0, budget,
    # We can specify modules and setting values as keyword arguments
    active=True,
    target=10.0,
    cc=0.8  
    matrix_adaptation='NONE'
)
```
Note that the `func`, `x0`, `sigma0` and `budget` arguments are **required**. Modules and settings can be specified via keyword arguments by they corresponding names in the `Modules` and `Settings` objects. Module options, such as `matrix_adaptation` in the above example, can be specified by their name as `str`. 

#### Low Level interface <a name="low-level-interface"></a>

To run an optimization, first create a `Modules` object specifying which modules and options to enable. Then construct a `Settings` object (which defines problem dimension and strategy parameters), and pass it through a `Parameters` object to the optimizer:

```python
from modcma import c_maes

# Instantiate module configuration
modules = c_maes.parameters.Modules()
modules.active = True
modules.matrix_adaptation = c_maes.parameters.MatrixAdaptationType.MATRIX

# Create Settings and Parameters objects
settings = c_maes.parameters.Settings(dim=10, modules=modules, sigma0=2.5)
parameters = c_maes.Parameters(settings)

# Instantiate the optimizer
cma = c_maes.ModularCMAES(parameters)

# Define objective function
def func(x):
    return sum(x**2)

# Run the optimization
cma.run(func)
```

The API provides fine-grained control over the optimization process. Instead of calling `run`, you can explicitly step through the algorithm:

```python
while not cma.break_conditions():
    cma.step(func)
```

Or execute the internal components of each iteration separately:

```python
while not cma.break_conditions():
    cma.mutate(func)
    cma.select()
    cma.recombine()
    cma.adapt()
```

This modularity allows experimentation with specific parts of the evolution strategy, 
such as custom selection, recombination, or adaptation routines.




---

### Tuning <a name="tuning"></a>

To facilitate **automated hyperparameter tuning**, ModularCMAES now provides functionality to **create and manipulate configuration spaces** directly compatible with popular optimization and AutoML tools, such as **SMAC**, **BOHB**. This functionality allows users to systematically explore both **algorithmic module combinations** and **numerical hyperparameters** (e.g., population size, learning rates, damping coefficients).

#### Configuration Space Generation

The configuration space is automatically derived from the available modules and tunable parameters via the function:

```python
from modcma.cmaescpp import get_configspace
```

Usage:

```python
from modcma.cmaescpp import get_configspace

# Create a configuration space for a 10-dimensional problem
cs = get_configspace(dim=10)
```

This function returns a `ConfigSpace.ConfigurationSpace` object containing:

- **Categorical parameters** for all available modules (e.g., mirrored sampling, restart strategy, bound correction).  
- **Numeric parameters** for key internal strategy settings such as `lambda0`, `mu0`, `sigma0`, `cs`, `cc`, `cmu`, `c1`, and `damps`.  
- A built-in constraint ensuring `mu0 ≤ lambda0`.  
- Optionally, the configuration space can include **only module-level options** by setting `add_popsize=False, add_sigma=False, add_learning_rates=False`.

Example:

```python
# Get only the module configuration space (no numeric parameters)
cs_modules = get_configspace(add_popsize=False, add_sigma=False, add_learning_rates=False)
```

#### Creating Settings from a Configuration

Once a configuration has been selected—either manually or from a tuner—the library provides a simple interface to construct a corresponding `Settings` object:

```python
from modcma.cmaescpp import settings_from_config
from ConfigSpace import Configuration

# Sample or load a configuration
config = cs.sample_configuration()

# Or for defaults
default = cs.default_configuration()

# The config can be edited
config['sampler'] = 'HALTON'

# Convert the configuration to a Settings object
# Note that keyword arguments like lb in the next example, can be passed to settings like so
settings = settings_from_config(dim=10, config=config, lb=np.ones(10))
```

The resulting `Settings` object can then be passed directly to the C++ backend:

```python
from modcma import c_maes

parameters = c_maes.Parameters(settings)
cma = c_maes.ModularCMAES(parameters)
cma.run(func)
```

### Integer Variables
A rudimentary mechanism for dealing with integer variables is implemented, which applies a lower bound on `sigma` for integer coordinates and rounds them to the nearest integer before evaluation, inspired by http://www.cmap.polytechnique.fr/~nikolaus.hansen/marty2024lb+ic-cma.pdf. 

An option can be provided to the `Settings` object or `settings_from_dict`/`settings_from_config` functions:

```python
dim = 5
settings = Settings(
    dim,
    integer_variables=[1, 2] # A list or numpy array of integer indices
)

settings = settings_from_dict(
    dim, 
    integer_variables=list(range(dim)) # All variables are integer
)
```


---

### Python-only (Legacy) <a name="python-only"></a>

> **Legacy notice:**  
> The Python-only implementation is **no longer actively developed** and does not include all features of the C++ version.  
> It remains available for experimentation and teaching purposes.
> A complete API documentation can be found [here](https://modularcmaes.readthedocs.io/) (under construction).

The Python interface provides a simple API and includes a convenience `fmin` function for optimizing a single objective function in one call:

```python
from modcma import fmin
xopt, fopt, used_budget = fmin(func=sum, x0=[1, 2, 3, 4], budget=1000, active=True, sigma0=2.5)
```

The main class is `ModularCMAES`, which mimics the structure of the C++ version but runs entirely in Python:

```python
from modcma import ModularCMAES
import numpy as np

def func(x: np.ndarray):
    return sum(x)

dim = 10
budget = 10_000

# Instantiate and run
cma = ModularCMAES(func, dim, budget=budget)
cma = cma.run()
```

You can also run the algorithm step by step:

```python
while not cma.break_conditions():
    cma.step()
```

Or explicitly call each internal phase — analogous to the C++ interface:

```python
while not cma.break_conditions():
    cma.mutate()
    cma.select()
    cma.recombine()
    cma.adapt()
```

---

#### Ask–Tell Interface <a name="ask-tell"></a>

The **Ask–Tell interface** is only available in the **Python implementation**.  It provides an alternative interaction model where function evaluations are managed externally.  This is particularly useful for **parallel**, **asynchronous**, or **expensive** objective evaluations,  
where you want to control when and how points are evaluated.

The optimizer generates candidate solutions via `ask()`, and their objective values are later supplied with `tell()`.

```python
from modcma import AskTellCMAES

def func(x):
    return sum(x**2)

# Instantiate an ask-tell CMA-ES optimizer
# Note: the objective function is not passed at construction
cma = AskTellCMAES(dim=10, budget=10_000, active=True)

while not cma.break_conditions():
    # Get a candidate solution
    xi = cma.ask()
    # Evaluate externally
    fi = func(xi)
    # Report the result back
    cma.tell(xi, fi)
```

This design provides flexibility when the evaluation process is nontrivial,  
such as when objectives are computed through simulations, APIs, or distributed systems.  
However, note that this interface is **not available** in the C++ backend.

## Modules <a name="modules"></a>

The CMA-ES Modular package provides various modules, grouped into 13 categories. For each of these categories a given option can be selected, which can be arbitrarly combined. The following table lists the categories and the available options. Not all modules are available in both versions (i.e. some are only implemented in C++), an overview is given in the table. By default, the first option in the table is selected for a given category. Boolean modules, i.e. modules that only can be turned on or off are turned off by default.

<table>
<tr ><td style="border: none!important;">
| Category | Option | Python | C++ |
|---|---|---|---|
| [Matrix Adaptation](#matrix-adaptation) | COVARIANCE | :green_circle: | :green_circle: |
|  | MATRIX | :red_circle: | :green_circle: |
|  | SEPARABLE | :red_circle: | :green_circle: |
|  | NONE | :red_circle: | :green_circle: |
|  | CHOLESKY | :red_circle: | :green_circle: |
|  | CMSA | :red_circle: | :green_circle: |
|  | NATURAL_GRADIENT | :red_circle: | :green_circle: |
| [Active Update](#active-update) | Off/On | :green_circle: | :green_circle: |
| [Elitism](#elitism) | Off/On | :green_circle: | :green_circle: |
| [Orthogonal Sampling](#orthogonal-sampling) | Off/On | :green_circle: | :green_circle: |
| [Sequential Selection](#sequential-selection) | Off/On | :green_circle: | :green_circle: |
| [Threshold Convergence](#threshold-convergence) | Off/On | :green_circle: | :green_circle: |
| [Sample Sigma](#sample-sigma) | Off/On | :green_circle: | :green_circle: |
| [Base Sampler](#base-sampler) | GAUSSIAN | :green_circle: | :red_circle: *(use Sample Transformer = GAUSSIAN)* |
|  | SOBOL | :green_circle: | :green_circle: |
|  | HALTON | :green_circle: | :green_circle: |
|  | UNIFORM | :red_circle: | :green_circle: |
| [Sample Transformer](#sample-transformer) | GAUSSIAN | :red_circle: | :green_circle: |
|  | SCALED_UNIFORM | :red_circle: | :green_circle: |
|  | LAPLACE | :red_circle: | :green_circle: |
|  | LOGISTIC | :red_circle: | :green_circle: |
|  | CAUCHY | :red_circle: | :green_circle: |
|  | DOUBLE_WEIBULL | :red_circle: | :green_circle: |
| [Recombination Weights](#recombination-weights) | DEFAULT | :green_circle: | :green_circle: |
|  | EQUAL | :green_circle: | :green_circle: |
|  | 1/2^λ | :green_circle: | :red_circle: *(use EXPONENTIAL instead)* |
|  | EXPONENTIAL | :red_circle: | :green_circle: |
| [Mirrored Sampling](#mirrored-sampling) | NONE | :green_circle: | :green_circle: |
|  | MIRRORED | :green_circle: | :green_circle: |
|  | PAIRWISE | :green_circle: | :green_circle: |
| [Step size adaptation](#step-size-adaptation) | CSA | :green_circle: | :green_circle: |
|  | TPA | :green_circle: | :green_circle: |
|  | MSR | :green_circle: | :green_circle: |
|  | XNES | :green_circle: | :green_circle: |
|  | MXNES | :green_circle: | :green_circle: |
|  | LPXNES | :green_circle: | :green_circle: |
|  | PSR | :green_circle: | :green_circle: |
|  | SR | :red_circle: | :green_circle: |
|  | SA | :red_circle: | :green_circle: |
| [Restart Strategy](#restart-strategy) | NONE | :green_circle: | :green_circle: |
|  | RESTART | :green_circle: | :green_circle: |
|  | IPOP | :green_circle: | :green_circle: |
|  | BIPOP | :green_circle: | :green_circle: |
|  | STOP | :red_circle: | :green_circle: |
| [Bound correction](#bound-correction) | NONE | :green_circle: | :green_circle: |
|  | SATURATE | :green_circle: | :green_circle: |
|  | MIRROR | :green_circle: | :green_circle: |
|  | COTN | :green_circle: | :green_circle: |
|  | TOROIDAL | :green_circle: | :green_circle: |
|  | UNIFORM_RESAMPLE | :green_circle: | :green_circle: |
|  | RESAMPLE | :red_circle: | :green_circle: |
| [Repelling Restart](#repelling-restart) | Off/On | :red_circle: | :green_circle: |
| [Center Placement](#center-placement) | X0 | :red_circle: | :green_circle: |
|  | ZERO | :red_circle: | :green_circle: |
|  | UNIFORM | :red_circle: | :green_circle: |
|  | CENTER | :red_circle: | :green_circle: |

</td></tr> </table>


**Notes**  
- In C++, `BaseSampler` generates uniform points in \[0,1)^d; the actual search-space distribution is controlled by `SampleTranformerType` (e.g., GAUSSIAN, CAUCHY).  
- Python’s “1/2^λ” recombination weights correspond to C++’s `EXPONENTIAL`.  
- New C++-only modules: `repelling_restart` (bool), `center_placement` (enum), and extended `ssa`, `bound_correction`, and sampling options.

### Matrix Adaptation <a name="matrix-adaptation"></a>

The ModularCMAES can be turned into an implementation of the (fast)-MA-ES algortihm by changing the `matrix_adaptation` option from `COVARIANCE` to `MATRIX` in the `Modules` object. This is currently only available in the C++ version of the framework. An example of specifying this, using the required `MatrixAdaptationType` enum:

```python
...
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.COVARIANCE
# or for MA-ES
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.MATRIX
# We can also only perform step-size-adaptation
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.NONE
# Or use the seperable CMA-ES
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.SEPARABLE
# Other variants:
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.CHOLESKY
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.CMSA
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.COVARIANCE_NO_EIGV
modules.matrix_adaptation = c_maes.options.MatrixAdaptationType.NATURAL_GRADIENT
```

### Active Update <a name="active-update"></a>

In the standard update of the covariance matrix C in the CMA-ES algorithm, only the most successful mutations are considered. However, the Active Update, introduced by Jastrebski et al., offers an alternative approach. This module adapts the covariance matrix by incorporating the least successful individuals with negative weights in the update process.

For the Python only version, this can be enabled by passing the option `active=True`:

```python
cma = ModularCMAES(func, dim, active=True)
```

For the C++ version, this can be done by setting the appropriate value in the `Modules` object:

```python
...
modules.active = True
```

### Elitism <a name="elitism"></a>

When this option is selected, (𝜇 + 𝜆)-selection instead of (𝜇, 𝜆)-selection is enabled. This can be usefull to speed up convergence on unimodal problems, but can have a negative impact on population diversity.  


For the C++ version, this can be done by setting the appropriate value in the `Modules` object:

```python
...
modules.elitist = True
```

For the Python only version, this can be enabled by passing the option `elitist=True`:

```python
cma = ModularCMAES(func, dim, elitist=True)
```

### Orthogonal Sampling <a name="orthogonal-sampling"></a>

Orthogonal Sampling was introduced by Wang et al. as an extension of Mirrored Sampling. This method improves sampling by ensuring that the newly sampled points in the population are orthonormalized using a Gram-Schmidt procedure.

For the Python only version, this can be enabled by passing the option `orthogonal=True`:

```python
cma = ModularCMAES(func, dim, orthogonal=True)
```

And for C++:

```python
...
modules.orthogonal = True
```

### Sequential Selection <a name="sequential-selection"></a>

Sequential Selection option offers an alternative approach to selection, originally proposed by Brockhoff et al., which optimizes the use of objective function evaluations by immediately ranking and comparing candidate solutions with the current best solution. Then, whenever more than $\mu$ individuals have been sampled that improve on the current best found solution, no more additional function evaluations are performed.

For the Python only version, this can be enabled by passing the option `sequential=True`:

```python
cma = ModularCMAES(func, dim, sequential=True)
```

And for C++:

```python
...
modules.sequential_selection = True
```

### Threshold Convergence <a name="threshold-convergence"></a>

In evolutionary strategies (ES), balancing exploration and exploitation is a critical challenge. The Threshold Convergence option, proposed by Piad et al. [25], provides a method to address this issue. It aims to prolong the exploration phase of evolution by requiring mutation vectors to reach a specific length threshold. This threshold gradually decreases over successive generations to transition into local search.

For the Python only version, this can be enabled by passing the option `threshold_convergence=True`:

```python
cma = ModularCMAES(func, dim, threshold_convergence=True)
```

And for C++:

```python
...
modules.threshold_convergence = True
```

### Sample Sigma <a name="sample-sigma"></a>

A method based on self-adaptation by co-evolution as seen in classical evolution strategies, where for each candidate solution the step size is sampled seperately from a lognormal distribution based on the global step size $\sigma$.  

For the Python only version, this can be enabled by passing the option `sample_sigma=True`:

```python
cma = ModularCMAES(func, dim, sample_sigma=True)
```

And for C++:

```python
...
modules.sample_sigma = True
```

### Quasi-Gaussian Sampling <a name="base-sampler"></a>

Instead of performing the simple random sampling from the multivariate Gaussian, new solutions can alternatively be drawn from quasi-random sequences (a.k.a. low-discrepancy sequences). We implemented two options for this module, the Halton and Sobol sequences.

This can be selected by setting the `base_sampler="sobol"` or `base_sampler="halton"` in the Python only version:

```python
cma = ModularCMAES(func, dim, base_sampler="gaussian")
# or 
cma = ModularCMAES(func, dim, base_sampler="sobol")
# or 
cma = ModularCMAES(func, dim, base_sampler="halton")
```

For C++, the `BaseSampler` enum should be provided to the `sampler` member of the `Modules` object:

```python
...
modules.sampler = c_maes.options.BaseSampler.UNIFORM
# or
modules.sampler = c_maes.options.BaseSampler.SOBOL
# or
modules.sampler = c_maes.options.BaseSampler.HALTON
```

Here, this works slightly different. The base sampler only defined the method for generating uniform sampling in a $[0,1)^d$ hyperbox. In order to define that a, for example, Gaussian distribution (this is default) should be used when running the algorithm, we need to specify the `sample_transformation` type:

```python
modules.sample_transformation = c_maes.options.SampleTranformerType.GAUSSIAN
# or 
modules.sample_transformation = c_maes.options.SampleTranformerType.SCALED_UNIFORM
# or 
modules.sample_transformation = c_maes.options.SampleTranformerType.LAPLACE
# or 
modules.sample_transformation = c_maes.options.SampleTranformerType.LOGISTIC
# or 
modules.sample_transformation = c_maes.options.SampleTranformerType.CAUCHY
# or 
modules.sample_transformation = c_maes.options.SampleTranformerType.DOUBLE_WEIBULL
```

### Recombination Weights <a name="recombination-weights"></a>

We implemented three different variants of the recombination weights used in the update of the strategy parameters, default, equal and $1/2\lambda$.

This can be selected by setting the `weights_option="sobol"` or `weights_option="halton"` in the Python only version:

```python
cma = ModularCMAES(func, dim, weights_option="default")
# or 
cma = ModularCMAES(func, dim, weights_option="equal")
# or 
cma = ModularCMAES(func, dim, weights_option="1/2^lambda")
```

For C++, the `RecombinationWeights` enum should be provided to the `weights` member of the `Modules` object:

```python
...
modules.weights = c_maes.options.RecombinationWeights.DEFAULT
# or
modules.weights = c_maes.options.RecombinationWeights.EQUAL
# or
modules.weights = c_maes.options.RecombinationWeights.EXPONENTIAL
```

### Mirrored Sampling <a name="mirrored-sampling"></a>

Mirrored Sampling, introduced by Brockhoff et al., aims to create a more evenly spaced sample of the search space. In this technique, half of the mutation vectors are drawn from the normal distribution, while the other half are the mirror image of the preceding random vectors. When using Pairwise Selection in combination with Mirrored Sampling, only the best point from each mirrored pair is selected for recombination. This approach ensures that the mirrored points do not cancel each other out during recombination. This module has three options, off, on and on + pairwise.

For Python, we can add the option `mirrored="mirrored"` or `mirrored="mirrored pairwise"`.

```python
cma = ModularCMAES(func, dim, mirrored=None)
# or
cma = ModularCMAES(func, dim, mirrored="mirrored")
# or 
cma = ModularCMAES(func, dim, mirrored="pairwise")
```

For C++ this can be configured using the `c_maes.options.Mirror` enum:

```python
...
modules.mirrored = c_maes.options.Mirror.NONE
# or 
modules.mirrored = c_maes.options.Mirror.MIRRORED
# or 
modules.mirrored = c_maes.options.Mirror.PAIRWISE
```

### Step size adaptation

Several methods for performing step size adaptation have been implemented in the framework. For more details on the implemented methods, we refer the interested reader to our 2021 paper.

The availble options for  `step_size_adaptation` for the Python only interface are: `{"csa", "tpa", "msr", "xnes", "m-xnes", "lp-xnes", "psr"}`, for which one can be selected and pased to the algortihms als active option, for example:

```python
cma = ModularCMAES(func, dim, step_size_adaptation="csa")
# or
cma = ModularCMAES(func, dim, step_size_adaptation="msr")
```

The same options are available for the C++ version, but should be passed via the `StepSizeAdaptation` enum, which has the following values available: `{CSA, TPA, MSR, XNES, MXNEs, LPXNES, PSR}` and can be configured via the `ssa` option:

```python
...
modules.ssa = c_maes.options.StepSizeAdaptation.CSA
# or 
modules.ssa = c_maes.options.StepSizeAdaptation.MSR
```

### Restart Strategy

Restarting an optimization algorithm, like CMA-ES, can be an effective way to overcome stagnation in the optimization process. The Modular CMA-ES package offers three restart strategies to help in such scenarios. The first restart option just restarts the algorithm. When IPOP is enabled, the algorithm employs a restart strategy that increases the size of the population after every restart. BIPOP on the other hand, not only changes the size of the population after a restart but alternates between larger and smaller population sizes.

For the Python only interface, this option can be configured with 4 values `{None, "restart", "IPOP", "BIPOP"}`:

```python
cma = ModularCMAES(func, dim, local_restart=None)
# or
cma = ModularCMAES(func, dim, local_restart="IPOP")
```

For the C++ version these should be passed via the `RestartStrategy` enum, which has the following values available: `{NONE, RESTART, IPOP, BIPOP, STOP}` and can be configured via the `restart_strategy` option:

```python
...
modules.restart_strategy = c_maes.options.RestartStrategy.NONE
# or 
modules.restart_strategy = c_maes.options.RestartStrategy.IPOP
```

Note that the C++ version has an addtional option here, `STOP`, which forces the algortihm to stop whenever a restart condition is met (not to be confused with a break condition). The C++ version also offers fine control over when a restart happens. The `Parameters` object has an `criteria` member, of which the `items` member defines a list of `Criterion` objects, which are triggers for when a restart should happen. This list can be freely changed, and default items can be deleted and modified if desired. For example, `cma.p.criteria.items = []`, clears this list entirely, and no restarts will happen. Several of the currently defined `Criterion` object can also be modified, often this can be controlled by setting the `tolerance` parameter. For example, for the `MinSigma` criterion, you can set `c_maes.restart.MinSigma.tolerance = 1` to ensure a parameter value of sigma below 1 triggers a restart. Note that this is a static varaible, so modifying it in this fashion sets this for all instances of `c_maes.restart.MinSigma`. Alternatively, it is possible to define a custom `Criterion` like so:

```python
class MyCriterion(modcma.restart.Criterion):
    def __init__(self):
        super().__init__("MyCriterionName")
    
    def on_reset(self, par: modcma.Parameters):
        """Called when a restart happens (also at the start)"""
    
    def update(self, par: modcma.Parameters):
        """Called after each iteration, needs to modify self.met"""
        self.met = True

# Python needs a reference to this object, 
# so you CANNOT create it in place, i.e. 
# cma.p.criteria.items = [MyCriterion()] 
# will produce an error
c = MyCriterion() 
cma.p.criteria.items = [c]
```

### Bound correction

Several methods for performing bound correction have been implemented in the framework. For more details on the implemented methods, we refer the interested reader to our 2021 paper.

The availble options for  `bound_correction` for the Python only interface are: `{None, "saturate", "unif_resample", "COTN", "toroidal", "mirror"}`, for which one can be selected and pased to the algortihms als active option, for example:

```python
cma = ModularCMAES(func, dim, bound_correction=None)
# or
cma = ModularCMAES(func, dim, bound_correction="saturate")
```

The same options are available for the C++ version, but should be passed via the `CorrectionMethod` enum, which has the following values available `{NONE, SATURATE, UNIFORM_RESAMPLE, COTN, TOROIDAL MIRROR}` and can be configure via the `bound_correction` option:

```python
...
modules.bound_correction = c_maes.options.CorrectionMethod.NONE
# or 
modules.bound_correction = c_maes.options.CorrectionMethod.SATURATE
```
 
### Sample Transformer <a name="sample-transformer"></a>

Controls the transformation from a base uniform sampler in \[0,1)^d to a target search-space distribution (e.g., `GAUSSIAN, CAUCHY, LAPLACE`, …) in the C++ backend. See the [paper](https://dl.acm.org/doi/10.1145/3712256.3726479). By default, this performs a transform to a standard Gaussian. 

### Repelling Restart <a name="repelling-restart"></a>

C++-only boolean module to bias restarts away from previously explored regions (helpful in multimodal settings). See the [paper](https://link.springer.com/chapter/10.1007/978-3-031-70068-2_17)

### Center Placement <a name="center-placement"></a>

C++-only enum controlling the initial center of mass of the sampling distribution (`X0, ZERO, UNIFORM, CENTER`) on restart, iniatiated by a `restart_strategy`. The options are:
   -  `X0` sets the intial center to x0 on every restart
   -  `ZERO` set the intial center to the all-zero vector
   -  `UNIFORM` picks a uniform random location inside the search space
   -  `CENTER` sets the center to the precise center of the search space.
   -  

## Citation <a name="citation"></a>

The following BibTex entry can be used for the citation.

```
@inproceedings{denobel2021,
   author = {de Nobel, Jacob and Vermetten, Diederick and Wang, Hao and Doerr, Carola and B\"{a}ck, Thomas},
   title = {Tuning as a Means of Assessing the Benefits of New Ideas in Interplay with Existing Algorithmic Modules},
   year = {2021},
   isbn = {9781450383516},
   publisher = {Association for Computing Machinery},
   address = {New York, NY, USA},
   url = {https://doi.org/10.1145/3449726.3463167},
   doi = {10.1145/3449726.3463167},
   booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
   pages = {1375–1384},
   numpages = {10},
   location = {Lille, France},
   series = {GECCO '21}
}
```

## License <a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
