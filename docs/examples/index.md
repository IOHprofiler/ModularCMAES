# Examples

Complete scripts, in [`examples/`](https://github.com/IOHprofiler/ModularCMAES/tree/master/examples)
of the repository. Most run in a few seconds; the parallel example takes about 20 s, the tuning
example about 40 s. The tests
run all of them, in a shortened form (`MODCMA_EXAMPLE_QUICK=1`), so they work with the current
version. The outputs on these pages are from the full runs, on macOS; on another platform the
numbers can differ a little ([reproducibility](../guide/reproducibility.md)).

| example | shows |
|---|---|
| [Multimodal problems](multimodal.md) | IPOP and BIPOP restarts to find the global optimum; repelling restarts to find several optima |
| [Parallel evaluation](parallel.md) | ask–tell with a process pool for an expensive objective |
| [Benchmarking on BBOB](bbob.md) | comparing configurations on the 24 BBOB functions with IOHexperimenter, logged for IOHanalyzer |
| [Tuning a configuration](tuning.md) | a random search over the configuration space of the modules |
| [Mixed-integer problems](mixed-integer.md) | integer and continuous variables with bounds |
| [Warm starting](warm-start.md) | the initial mean, step size and covariance matrix from the solutions of a related task (WS-CMA-ES), and the paper's synthetic experiment |
| [Coming from pycma](from-pycma.md) | the modcma equivalents of pycma's calls and options |
| [Checkpointing a long run](../guide/checkpointing.md#checkpointing-a-long-run) | saving the optimizer with pickle every n generations, and resuming after an interruption, bit for bit |
| [The C++ library](../guide/cpp.md) | a C++ program with `run` and ask–tell |
