# Deep dives

The Modular CMA-ES combines mechanisms that were published separately, so a module is rarely
exactly the algorithm it is named after: it is that algorithm's idea, fitted into one framework
where every part can be combined with every other. These pages say, per module, what modcma does,
where it differs from the published method, and which combination of modules comes closest to the
original. They are for readers who compare results with the literature, or with pycma, `cmaes` and
other implementations.

* [Matrix adaptations](matrix-adaptations.md): CMA-ES, sep-CMA-ES, MA-ES, Cholesky-CMA-ES,
  CMSA-ES, xNES, and the variants without a published counterpart.
* [Step-size adaptation](step-size-adaptation.md): CSA, TPA, TPA_INPLACE, MSR, PSR, the xNES
  rules, the success rule and self-adaptation.
* [Selection, sampling and restarts](selection-sampling-restarts.md): the recombination weights,
  the active update per matrix adaptation, elitism, sequential and pairwise selection, the samplers,
  IPOP, BIPOP and the restart criteria compared with pycma's termination criteria.

Some of the differences are deliberate generalizations (the same step-size rule for every matrix
adaptation); the open questions are GitHub issues, linked where they come up. Which combinations
interact, and which to avoid, is on the [compatibility](../guide/compatibility.md) page.

## A general difference: one framework for every module

Two choices hold for every module and explain many of the differences below:

* **One generation, one order.** Every generation runs the same steps, in the order below; the
  modules act within them. The mean is recombined with learning rate 1: the weighted average of
  the µ best (of all λ with `weights=UTILITY`). A published algorithm with another order or another
  mean update is reproduced as far as this order allows.

    ```mermaid
    flowchart TD
        R{"a restart criterion met?"} -- yes --> N["end the run: evaluate its center of mass;<br>start a new one (restart strategy, center placement, repelling)"]
        R -- no --> S
        N --> S["sample z, y = A z, x = m + σ y<br>(sampler, mirrored, orthogonal, threshold convergence, bound correction)"]
        S --> E["evaluate the points<br>(TPA's two points with the first batch; sequential selection can stop early)"]
        E --> Sel["select (elitism, pairwise)"]
        Sel --> M["recombine the mean (weights)"]
        M --> P["update the evolution paths p_σ, p_c"]
        P --> SS["adapt σ (ssa)"]
        SS --> C["adapt the matrix (matrix_adaptation, active)"]
        C --> U["update the restart criteria"]
        U --> R
    ```

* **One step-size path for every matrix adaptation.** CSA needs C^-1/2 Δm. Only `COVARIANCE` has
  the eigendecomposition for it. Every other matrix adaptation uses the weighted sum of the selected
  standard normal samples z (the framework paper, "CSA without C^-1/2"). With x = m + σAz this sum
  is A⁻¹Δm: it has the length of C^-1/2 Δm, but is rotated against it unless A is the symmetric
  root of C. The rotation changes when A does, so the path is not exactly CMA-ES's; for `NONE` and
  `SEPARABLE` (A diagonal) it is exact.
