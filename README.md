# ModularCMAES ![Unittest](https://github.com/IOHprofiler/ModularCMAES/workflows/Unittest/badge.svg)

This is a restructured version of the Modular CMA-ES framework, first introduced in ...
Its modular design allows for the creation of thousands of variants of the CMA-ES algorithm.

## Installation
Installation can be done via pip, using:
`$ pip install modcma`

## Usage
To optimize a single function, we provide a basic fmin interface, which can be used as follows:
''
from modcma import modularcmaes
modularcmaes.fmin(func=sum, dim=5, maxfun=100)
''

[Documentation](https://modularcmaes.readthedocs.io/)

Running tests
`$ python3 -m unittest discover`

Running optimizer
`$ python3 -m modcma  [-h] [-f FID] [-d DIM] [-i ITERATIONS] [-l] [-c] [-L LABEL]
                   [-s SEED] [-a ARGUMENTS [ARGUMENTS ...]]`

