# Citing

If you use modcma in your research, please cite:

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

The framework and its current modules are described in *The Modular CMA-ES: A Framework for Modern
Evolution Strategies*; [reproducing its experiments](reproducibility.md). The modular design goes
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
* H. Wang, M. Emmerich and T. Bäck (2014). Mirrored orthogonal sampling with pairwise selection in
  evolution strategies. *ACM SAC*.
