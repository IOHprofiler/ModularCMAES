# ModularCMAES ![Unittest](https://github.com/IOHprofiler/ModularCMAES/workflows/Unittest/badge.svg) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/e25b2d338c194d67954fc9e138ca69cc)](https://app.codacy.com/gh/IOHprofiler/ModularCMAES?utm_source=github.com&utm_medium=referral&utm_content=IOHprofiler/ModularCMAES&utm_campaign=Badge_Grade) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/73720e228a89480585bdde05d3806661)](https://www.codacy.com/gh/IOHprofiler/ModularCMAES/dashboard?utm_source=github.com&utm_medium=referral&utm_content=IOHprofiler/ModularCMAES&utm_campaign=Badge_Coverage)

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
`$ python3 -m modcma  [-h] [-f FID] [-d DIM] [-i ITERATIONS] [-l] [-c] [-L LABEL] [-s SEED] [-a ARGUMENTS [ARGUMENTS ...]]`

