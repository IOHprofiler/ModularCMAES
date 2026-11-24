# Deep dives

The Modular CMA-ES combines mechanisms that were published separately, so a module is rarely
exactly the algorithm it is named after: it is that algorithm's idea, fitted into one framework
where every part can be combined with every other. These pages say, per module, what modcma does,
where it differs from the published method, and which combination of modules comes closest to the
original. They are for readers who compare results with the literature, or with pycma, `cmaes` and
other implementations.

* [Matrix adaptations](matrix-adaptations.md): CMA-ES, sep-CMA-ES, MA-ES, Cholesky-CMA-ES,
  CMSA-ES, xNES, and the variants without a published counterpart.
* [Step-size adaptation](step-size-adaptation.md): CSA, TPA, MSR, PSR, the xNES rules, the success
  rule and self-adaptation.
* [Selection, sampling and restarts](selection-sampling-restarts.md): the recombination weights,
  the active update per matrix adaptation, elitism, sequential and pairwise selection, the samplers,
  IPOP, BIPOP and the restart criteria compared with pycma's termination criteria.

Some of the differences are deliberate generalizations (the same step-size rule for every matrix
adaptation); others are recorded for fixing (TASKS.md, WP23; marked *known issue* with the
identifier). Which combinations interact, and which to avoid, is on the
[compatibility](../guide/compatibility.md) page.

## A general difference: one framework for every module

Two choices hold for every module and explain many of the differences below:

* **One generation, one order.** Each generation samples, evaluates, selects, recombines the mean
  (the weighted average of the µ best, with learning rate 1), updates the evolution paths, then the
  step size, then the matrix. A published algorithm with another order or another mean update is
  reproduced as far as this order allows.
* **One step-size path for every matrix adaptation.** CSA needs C^-1/2 Δm. Only `COVARIANCE` has
  the eigendecomposition for it; every other matrix adaptation uses the weighted sum of the selected
  standard normal samples z, which is the same vector when the points are sampled as m + σAz
  (the framework paper, "CSA without C^-1/2").
