# Citing

If you use modcma in your research, please cite the paper that describes the framework (under
review; until it is published, its arXiv version):

> Jacob de Nobel, Diederick Vermetten, Anna V. Kononova, Carola Doerr and Thomas Bäck. 2026. *The
> Modular CMA-ES: A Framework for Modern Evolution Strategies.* arXiv:2610.09699.
> [doi:10.48550/arXiv.2610.09699](https://doi.org/10.48550/arXiv.2610.09699)

```bibtex
@misc{denobel2026modcma,
  author        = {de Nobel, Jacob and Vermetten, Diederick and Kononova, Anna V. and Doerr, Carola and B\"{a}ck, Thomas},
  title         = {The Modular {CMA-ES}: A Framework for Modern Evolution Strategies},
  year          = {2026},
  eprint        = {2610.09699},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2610.09699},
  url           = {https://arxiv.org/abs/2610.09699},
}
```

The earlier version of the framework and the study of its modules with algorithm configuration:

> Jacob de Nobel, Diederick Vermetten, Hao Wang, Carola Doerr and Thomas Bäck. 2021. *Tuning as a
> Means of Assessing the Benefits of New Ideas in Interplay with Existing Algorithmic Modules.* In
> Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '21),
> 1375–1384. [doi:10.1145/3449726.3463167](https://doi.org/10.1145/3449726.3463167)

```bibtex
@inproceedings{denobel2021,
  author    = {de Nobel, Jacob and Vermetten, Diederick and Wang, Hao and Doerr, Carola and B\"{a}ck, Thomas},
  title     = {Tuning as a Means of Assessing the Benefits of New Ideas in Interplay with Existing Algorithmic Modules},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  series    = {GECCO '21},
  year      = {2021},
  pages     = {1375--1384},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3449726.3463167},
}
```

The experiments of the framework paper can be [reproduced](reproducibility.md). The modular design goes
back to van Rijn et al., [*Evolving the Structure of Evolution
Strategies*](https://ieeexplore.ieee.org/document/7850138) (IEEE SSCI 2016).

GitHub's "Cite this repository" button uses
[`CITATION.cff`](https://github.com/IOHprofiler/ModularCMAES/blob/master/CITATION.cff).

## The methods behind the modules

The docstrings of the [options](reference/options.md) cite the paper behind each module. The main
ones:

* N. Hansen and A. Ostermeier (2001). Completely derandomized self-adaptation in evolution
  strategies. *Evolutionary Computation* 9(2).
* N. Hansen (2016). The CMA evolution strategy: a tutorial. arXiv:1604.00772.
* G. A. Jastrebski and D. V. Arnold (2006). Improving evolution strategies through active
  covariance matrix adaptation. *IEEE CEC*.
* A. Auger and N. Hansen (2005). A restart CMA evolution strategy with increasing population size.
  *IEEE CEC*.
* N. Hansen (2009). Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed.
  *GECCO Companion*.
* R. Ros and N. Hansen (2008). A simple modification in CMA-ES achieving linear time and space
  complexity. *PPSN X*.
* H.-G. Beyer and B. Sendhoff (2017). Simplify your covariance matrix adaptation evolution strategy.
  *IEEE Transactions on Evolutionary Computation* 21(5).
* H.-G. Beyer and B. Sendhoff (2008). Covariance matrix adaptation revisited – the CMSA evolution
  strategy. *PPSN X*.
* T. Glasmachers, T. Schaul, Y. Sun, D. Wierstra and J. Schmidhuber (2010). Exponential natural
  evolution strategies. *GECCO*.
* D. Brockhoff, A. Auger, N. Hansen, D. V. Arnold and T. Hohm (2010). Mirrored sampling and
  sequential selection for evolution strategies. *PPSN XI*.
* A. Auger, D. Brockhoff and N. Hansen (2011). Mirrored sampling in evolution strategies with
  weighted recombination. *GECCO*.
* N. Hansen (2008). CMA-ES with two-point step-size adaptation. arXiv:0805.0231.
* O. Ait Elhara, A. Auger and N. Hansen (2013). A median success rule for non-elitist evolution
  strategies: study of feasibility. *GECCO*.
* I. Loshchilov (2014). A computationally efficient limited memory CMA-ES for large scale
  optimization. *GECCO*.
* O. Krause, T. Glasmachers and C. Igel (2017). Qualitative and quantitative assessment of step
  size adaptation rules. *FOGA*. (xNES, m-xNES and lp-xNES step-size adaptation)
* C. Igel, T. Suttorp and N. Hansen (2006). A computational efficient covariance matrix update and
  a (1+1)-CMA for evolution strategies. *GECCO*.
* O. Krause, D. R. Arbonès and C. Igel (2016). CMA-ES with optimal covariance update and storage
  complexity. *NeurIPS*.
* A. Ahrari, K. Deb and M. Preuss (2017). Multimodal optimization by covariance matrix
  self-adaptation evolution strategy with repelling subpopulations. *Evolutionary Computation*
  25(3).
* A. Piad-Morffis, S. Estévez-Velarde, A. Bolufé-Röhler, J. Montgomery and S. Chen (2015).
  Evolution strategies with thresheld convergence. *IEEE CEC*.
* F. Caraffini, A. V. Kononova and D. Corne (2019). Infeasibility and structural bias in
  differential evolution. *Information Sciences* 496. (the bound corrections)
* S. C. Maree, T. Alderliesten, D. Thierens and P. A. N. Bosman (2018). Real-valued evolutionary
  multi-modal optimization driven by hill-valley clustering. *GECCO*.
* J. de Nobel, D. Vermetten, A. V. Kononova, O. M. Shir and T. Bäck (2024). Avoiding redundant
  restarts in multimodal global optimization. *PPSN XVIII*. (repelling restarts)
* J. de Nobel, D. Vermetten, T. Bäck and A. V. Kononova (2024). Sampling in CMA-ES: low numbers of
  low discrepancy points. *ECTA*.
* J. de Nobel, D. Vermetten, H. Wang, A. V. Kononova, G. Rudolph and T. Bäck (2025). Abnormal
  mutations: evolution strategies don't require Gaussianity. arXiv:2502.03148.
* T. Marty, N. Hansen, A. Auger, Y. Semet and S. Héron (2024). LB+IC-CMA-ES: two simple
  modifications of CMA-ES to handle mixed-integer problems. *PPSN XVIII*.
* H. Wang, M. Emmerich and T. Bäck (2014). Mirrored orthogonal sampling with pairwise selection in
  evolution strategies. *ACM SAC*.
* H. Wang, M. Emmerich and T. Bäck (2019). Mirrored orthogonal sampling for covariance matrix
  adaptation evolution strategies. *Evolutionary Computation* 27(4).
