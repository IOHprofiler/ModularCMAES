# ConfigurableCMAES ![Unittest](https://github.com/IOHprofiler/ModularCMAES/workflows/Unittest/badge.svg)

This is a restructured version of the Modular CMA-ES framework, first introduced in ...
Its modular design allows for the creation of thousands of variants of the CMA-ES algorithm.

## Installation
Installation can be done via pip, using:
`$ pip install modcma`

## Usage
To optimize a single function, we provide a basic fmin interface, which can be used as follows:
''
from ccmaes import configurablecmaes
configurablecmaes.fmin(func=sum, dim=5, maxfun=100)
''

[Documentation](https://ccmaes.readthedocs.io/)

Running tests
`$ python3 -m unittest discover`

Running optimizer
`$ python3 -m ccmaes  [-h] [-f FID] [-d DIM] [-i ITERATIONS] [-l] [-c] [-L LABEL]
                   [-s SEED] [-a ARGUMENTS [ARGUMENTS ...]]`


[Documentation](https://ccmaes.readthedocs.io/)
