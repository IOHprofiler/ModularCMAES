# ConfigurableCMAES

Running tests
`$ python3 -m unittest discover`

Running optimizer
`$ python3 -m src  [-h] [-f FID] [-d DIM] [-i ITERATIONS] [-l] [-c] [-L LABEL]
                   [-s SEED] [-a ARGUMENTS [ARGUMENTS ...]]`


Generating documentation
`cd docs`
`make html`

Running simple webserver in order to view documentation (they should be generated first)
`cd docs/build/html`
`python3 -m http.server`
The docs should now be accesible at port 8000 on your local machine.