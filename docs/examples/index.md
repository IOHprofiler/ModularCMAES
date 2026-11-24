# Examples

Complete scripts, in [`examples/`](https://github.com/IOHprofiler/ModularCMAES/tree/master/examples)
of the repository. Each runs in a few seconds, and all of them are run by the tests, so they work
with the current version.

| example | shows |
|---|---|
| [Multimodal problems](multimodal.md) | IPOP and BIPOP restarts to find the global optimum; repelling restarts to find several optima |
| [Parallel evaluation](parallel.md) | ask–tell with a process pool for an expensive objective |
| [Benchmarking on BBOB](bbob.md) | comparing configurations on the 24 BBOB functions with IOHexperimenter, logged for IOHanalyzer |
| [Tuning a configuration](tuning.md) | a random search over the configuration space of the modules |
| [Mixed-integer problems](mixed-integer.md) | integer and continuous variables with bounds |
| [Coming from pycma](from-pycma.md) | the modcma equivalents of pycma's calls and options |
| [The C++ library](../guide/cpp.md) | a C++ program with `run` and ask–tell |
