# ModularCMAES ![Unittest](https://github.com/IOHprofiler/ModularCMAES/workflows/Unittest/badge.svg) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/e25b2d338c194d67954fc9e138ca69cc)](https://app.codacy.com/gh/IOHprofiler/ModularCMAES?utm_source=github.com&utm_medium=referral&utm_content=IOHprofiler/ModularCMAES&utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/73720e228a89480585bdde05d3806661)](https://www.codacy.com/gh/IOHprofiler/ModularCMAES/dashboard?utm_source=github.com&utm_medium=referral&utm_content=IOHprofiler/ModularCMAES&utm_campaign=Badge_Coverage)

The **Modular CMA-ES** is a Python and C++ package that provides a modular implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm. This package allows you to create various algorithmic variants of CMA-ES by enabling or disabling different modules, offering flexibility and customization in evolutionary optimization. In addition to the CMA-ES, the library includes an implementation of the Matrix Adaptation Evolution Strategy (MA-ES) algorithm, which has similar emprical performance on most problems, but signifanctly lower runtime. All modules implemented are compatible with both the CMA-ES and MA-ES.

This implementation is based on the algorithm introduced in the paper "[Evolving the Structure of Evolution Strategies. (2016)](https://ieeexplore.ieee.org/document/7850138)" by Sander van Rijn et. al. If you would like to cite this work in your research, please cite the paper: "[Tuning as a Means of Assessing the Benefits of New Ideas in Interplay with Existing Algorithmic Modules (2021)](https://doi.org/10.1145/3449726.3463167)" by Jacob de Nobel, Diederick Vermetten, Hao Wang, Carola Doerr and Thomas Bäck.

This README provides a high level overview of the implemented modules, and provides some usage examples for both the Python-only and the C++-based versions of the framework. A complete API documentation can be found [here](https://modularcmaes.readthedocs.io/) (under construction).

## Table of Contents

- [ModularCMAES   ](#modularcmaes---)
  - [Table of Contents](#table-of-contents)
  - [Installation ](#installation-)
    - [Python Installation](#python-installation)
    - [Installation from source](#installation-from-source)
  - [Usage ](#usage-)
    - [Python-only ](#python-only-)
    - [Ask-Tell Interface ](#ask-tell-interface-)
    - [C++ Backend ](#c-backend-)
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

To optimize a single function, we provide a basic fmin interface, which requires two parameters: `func`, which is the function to be minimized, and `x0`, the initial estimate for the function. Optionally, any parameter that is valid for the `ModularCMAES` class, is valid for this function as keyword argument. For example, to minimize the value of the sum function, in 4D with a budget of 1000 function evaluations, using an active CMA-ES with an intial stepsize $\sigma$ of 2.5, we could use the following:

```python
from modcma import fmin
xopt, fopt, used_budget = fmin(func=sum, x0=[1, 2, 3, 4], budget=1000, active=True, sigma0=2.5)
```

### Python-only <a name="python-only"></a>

The Python-only implentation revolves around the `ModularCMAES` class. The class has a `run` method, which runs the specified algorithm until any break conditions arise.

```python
from modcma import ModularCMAES

def func(x: np.ndarray):
   return sum(x)

dim = 10
budget = 10_000

# Create an instance of the CMA-ES (no modules active)
cma = ModularCMAES(func, dim, budget=budget)

# Run until break conditions are met
cma = cma.run()
```

Alternatively, we could also iteratively run the `step` method, for a more fine grained control on how the algorithm is executed.  

```python
cma = ModularCMAES(func, dim, budget=budget)

while not cma.break_conditions():
   cma.step()
```

At an even lower level, we could run all methods ran by the `step` methods seperately, which are (in order) `mutate`, `select`, `recombine` and `adapt`. The following snippet shows an example of all three methods.

```python
cma = ModularCMAES(func, dim, budget=budget)

while not cma.break_conditions():
   cma.mutate()
   cma.select()
   cma.recombine()
   cma.adapt()
```

### Ask-Tell Interface <a name="ask-tell"></a>

Often, it can be usefull consider the algorithm in an Ask-Tell fashion, such that we can sequentally evaluate points while having outside control of the objective function. For this purpose, we provide the `AskTellCMAES` interface, which can be used as follows:

```python
from modcma import AskTellCMAES

# Instantiate an ask-tell cmaes. Note that the objective function argument is omitted here. 
# All other parameters, e.g. the active modules can be passed by keyword, similar to ModularCMAES
cma = AskTellCMAES(dim, budget=budget, active=True)

while not cma.break_conditions():
   # Retrieve a single new candidate solution
   xi = cma.ask()
   # Evaluate the objective function
   fi = func(xi)
   # Update the algorithm with the objective function value
   cma.tell(xi, fi)
```

### C++ Backend <a name="c-backend"></a>

For obvious performance reasons, we've also implemented the algorithm in C++, with an interface to Python. The algorithm can be accessed similarly in Python, but calling it is slightly more verbose. The `ModularCMAES` class in C++ accepts a single argument, which is an `Parameters` object. This object must be instantiated with a `Settings` object, which in turn is built from the problem dimension and a `Modules` object, which can be used to specify certain module options. A boilerplate code example for this process is given in the following:

```python
# import the c++ subpackage
from modcma import c_maes
# Instantate a modules object
modules = c_maes.parameters.Modules()
# Create a settings object, here also optional parameters such as sigma0 can be specified
settings = c_maes.parameters.Settings(dim, modules, sigma0 = 2.5)
# Create a parameters object
parameters = c_maes.Parameters(settings)
# Pass the parameters object to the ModularCMAES optimizer class
cma = c_maes.ModularCMAES(parameters)
```

Then, the API for both the Python-only and C++ interface is mostly similar, and a single run of the algorithm can be performed by using the `run` function. A difference is that now the objective function is a parameter of the run function, and not pass when the class is instantiated.

```python
cma.run(func)
```

Similarly, the `step` function is also directly exposed:

```python
while not cma.break_conditions():
   cma.step(func)
```

Or by calling the function in the `step` seperately:

```python
while not cma.break_conditions():
   cma.mutate(func)
   cma.select()
   cma.recombine()
   cma.adapt()
```

## Modules <a name="modules"></a>

The CMA-ES Modular package provides various modules, grouped into 13 categories. For each of these categories a given option can be selected, which can be arbitrarly combined. The following table lists the categories and the available options. Not all modules are available in both versions (i.e. some are only implemented in C++), an overview is given in the table. By default, the first option in the table is selected for a given category. Boolean modules, i.e. modules that only can be turned on or off are turned off by default.

<table>
<tr ><td style="border: none!important;">

| Category                                         | Option    | Python         | C++            |
| --------                                         | ------    | ------         | ----           |
| [Matrix Adaptation](#matrix-adaptation)          | Covariance       | :green_circle: | :green_circle: |
|                                                  |  Matrix          | :red_circle: | :green_circle: |
|                                                  |  Separable       | :red_circle: | :green_circle: |
|                                                  |  None            | :red_circle: | :green_circle: |
| [Active Update](#active-update)   | Off/On       | :green_circle:   | :green_circle: |
| [Elitism](#elitism)              | Off/On        | :green_circle:    | :green_circle: |
| [Orthogonal Sampling](#orthogonal-sampling)      | Off/On    | :green_circle: | :green_circle: |
| [Sequential Selection](#sequential-selection)    | Off/On    | :green_circle: | :green_circle: |
| [Threshold Convergence](#threshold-convergence) | Off/On    | :green_circle: | :green_circle: |
| [Sample Sigma](#sample-sigma)          | Off/On  | :green_circle: | :green_circle: |
| [Base Sampler](#base-sampler)          | Gaussian  | :green_circle: | :green_circle: |
|                       | Sobol     | :green_circle: | :green_circle: |
|                       | Halton    | :green_circle: | :green_circle: |
| [Recombination Weights](#recombination-weights) | Default   | :green_circle: | :green_circle: |
|                       | Equal     | :green_circle: | :green_circle: |
|                       | $1/2^\lambda$   | :green_circle: | :green_circle: |
| [Mirrored Sampling](#mirrored-sampling)     | Off       | :green_circle: | :green_circle: |
|                       | On        | :green_circle: | :green_circle: |
|                       | Pairwise  | :green_circle: | :green_circle: |

</td><td style="border: none!important;">

| Category              | Option             | Python         | C++            |
| --------              | ------             | ------         | ----           |
| [Step size adaptation](#step-size-adaptation)  | CSA                | :green_circle: | :green_circle: |
|                       | TPA                | :green_circle: | :green_circle: |
|                       | MSR                | :green_circle: | :green_circle: |
|                       | PSR                | :green_circle: | :green_circle: |
|                       | XNES               | :green_circle: | :green_circle: |
|                       | MXNES              | :green_circle: | :green_circle: |
|                       | MPXNES             | :green_circle: | :green_circle: |
| [Restart Strategy](#restart-strategy)      | Off                | :green_circle: | :green_circle: |
|                       | Restart            | :green_circle: | :green_circle: |
|                       | IPOP               | :green_circle: | :green_circle: |
|                       | BIPOP              | :green_circle: | :green_circle: |
| [Bound Correction](#bound-correction)      | Off                | :green_circle: | :green_circle: |
|                       | Saturate           | :green_circle: | :green_circle: |
|                       | Mirror             | :green_circle: | :green_circle: |
|                       | COTN               | :green_circle: | :green_circle: |
|                       | Toroidal           | :green_circle: | :green_circle: |
|                       | Uniform resample   | :green_circle: | :green_circle: |

</td></tr> </table>

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

For the Python only version, this can be enabled by passing the option `elitist=True`:

```python
cma = ModularCMAES(func, dim, elitist=True)
```

For the C++ version, this can be done by setting the appropriate value in the `Modules` object:

```python
...
modules.elitist = True
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
modules.weights = c_maes.options.RecombinationWeights.HALF_POWER_LAMBDA
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
