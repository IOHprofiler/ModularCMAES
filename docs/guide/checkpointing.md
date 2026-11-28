# Saving and resuming a run

An optimizer can be saved with `pickle` at any point of a run and loaded later, in the same process
or another one. The loaded optimizer continues with the same numbers as the run that was not
interrupted, bit for bit: the same points, the same step sizes, the same restarts.
`copy.deepcopy` (and `copy.copy`) give an independent optimizer the same way. Pickle protocol 2 or
higher is needed (the default is 5).

```python
import copy
import pickle

import numpy as np
import modcma

def sphere(x):
    return float(np.sum(x**2))

es = modcma.ModularCMAES(modcma.Settings(5, budget=2_000, seed=1))
for _ in range(10):
    es.step(sphere)

state = pickle.dumps(es)        # bytes: write them to a file, send them to another process
resumed = pickle.loads(state)
twin = copy.deepcopy(es)

es.run(sphere)
resumed.run(sphere)             # the objective is not saved: it is given again
twin.run(sphere)
assert es.p.stats.global_best.y == resumed.p.stats.global_best.y == twin.p.stats.global_best.y
```

The optimizer and everything in it can be pickled on its own: `Settings`, `Modules`, `Parameters`
(`es.p`), the statistics, the populations and every module object (`es.p.adaptation`,
`es.p.sampler`, `es.p.mutation`, the restart criteria, ...).

## Checkpointing a long run

To survive a crash, or the time limit of a cluster job, save the optimizer every n generations and
start from the last checkpoint when there is one. The script below writes a checkpoint every 100
generations, is interrupted after 250 generations, and resumes from the checkpoint of generation 200:
it repeats generations 201 to 250 and ends exactly as the run that was not interrupted.

```python
--8<-- "examples/checkpoint.py"
```

Output:

```text
interrupted after generation 250
resumed from the checkpoint of generation 200
checkpoint: 314866 bytes
resumed       best f = 0 after 99417 evaluations, 1324 generations, 7 runs
uninterrupted best f = 0 after 99417 evaluations, 1324 generations, 7 runs
```

* **Write the checkpoint to a new file and rename it** (`os.replace`): a run that is interrupted
  while it writes keeps the previous checkpoint.
* **The cost** of a checkpoint is small next to a generation of an expensive objective: the
  state of the default configuration in d = 10 is 17 kB and takes 14 µs to save and 20 µs to load
  with `pickle` (Apple M5); in d = 100, 400 kB and 0.2 ms each. The state grows with the
  population (IPOP, BIPOP) and with the repelling archive. Two modules make it larger or slower:
  `constants.cache_samples` saves its cache (up to `cache_max_doubles` numbers, 16 MB), and the
  `SOBOL` sampler computes its points again when it is loaded (about 1 ms in d = 5).
* **With ask–tell**, save at any point, also between `ask()` and `tell()`: the state holds the asked
  points, and `ask()` returns them again after loading.

```python
es = modcma.ModularCMAES(modcma.Settings(5, budget=2_000, seed=1))
X = es.ask()
es = pickle.loads(pickle.dumps(es))   # the process restarts while X is being evaluated
assert np.array_equal(es.ask(), X)
es.tell([sphere(x) for x in X])
```

## What is saved

The state is everything that changes during a run:

* the settings and the modules (also a setting changed during the run);
* the random generator, and the positions of the samplers: the Sobol and Halton sequences, the
  cache of `CachedSampler`, the second number of a Box–Muller pair, the xoshiro generator of
  `fast_sampling`;
* the mean, the step size, the evolution paths, the matrix of the matrix adaptation and what it
  derived from it (the eigendecomposition, the inverse factor of `MATRIX` and whether it is current,
  the cached `metric()`);
* the populations, the recombination weights, the statistics (`es.p.stats`, with the best
  solutions and the record of every run);
* the state of the step-size adaptation (TPA's last comparison and the step size of its test
  points, the rank statistic of `TPA_INPLACE`, ...), the restart criteria (their histories), the
  restart strategy (BIPOP's regimes), the repelling archive and the update of the archive during
  a restart;
* the generation in progress: the points of `ask()` that wait for `tell()`, the step that runs next,
  whether a gradient can be injected.

Not saved:

* **the objective and the gradient function**: they are given again (`run(f)`, `step(f)`, or the
  values for `tell()`). A Python function is often not picklable (a lambda, a closure, a C
  extension), and the run does not depend on it otherwise;
* **the process-wide settings**: `modcma.constants` (`cache_samples`, `use_box_muller`, ...) and the
  tolerances of the restart criteria (`restart.TolX.tolerance`, ...). A run continues with the
  values of the process that loads it, so set them before loading as before the run;
* **the global generator** (`modcma.set_seed`): an optimizer draws from its own generator, which
  is saved. An object created on its own (`sampling.Uniform(d)`) draws from the global one, also
  after loading.

Objects that share a generator, as the modules of one optimizer do, still share one after loading.
A module pickled on its own (`pickle.dumps(es.p.sampler)`) gets a copy of the generator.
A copy has the same generator state as the original, so two copies continue with the same random
numbers: for independent runs, create new optimizers with other seeds.

Objects defined in Python:

* **a restart criterion defined in Python** (a subclass of `restart.Criterion`) is saved with
  `pickle`: its class must be importable (defined at the top level of a module), and its attributes
  picklable;
* **a subclass of `ModularCMAES`** (or of `Settings`, ...) keeps its class and its attributes;
* **a `sampling.PySampler`** calls a Python function and cannot be saved: `pickle.dumps` raises
  `TypeError`.

## The format

`__getstate__` returns the bytes of the optimizer's C++ serialization, the same in C++ and Python:
a header (the magic number `MODCMAST`, the format version, the kind of object), the object, and a
CRC-32 checksum (that of `zlib.crc32`). Numbers are little-endian and floating point numbers keep
all their bits. The Mersenne Twister is stored in the representation of the C++ standard, so a state
loads with every C++ standard library: a checkpoint written on Linux loads on macOS. The runs
continue bit for bit on the same platform and toolchain; on another one they continue as a run of
that platform does ([reproducibility](reproducibility.md#how-far-identical-results-go)).

The format is versioned. A newer modcma reads the states of older versions (or says that it does
not); a state of a newer format raises `ValueError` ("written by a newer modcma, in format version
...: update modcma to load it"). A state that is truncated or corrupted fails the checksum and raises
`ValueError` too, as bytes that are not a modcma state, or a state of another kind of object.

As with every pickle, load only states that you trust: the checksum detects accidental damage, not
a state crafted to describe an inconsistent optimizer, and a pickle can run code when it is loaded.

## In C++

`include/serialization.hpp` has the same, for C++ programs:

```cpp
#include <fstream>
#include "serialization.hpp"

// Save: the bytes of the state
const std::string state = serialization::save(es);  // ModularCMAES es
std::ofstream("run.state", std::ios::binary) << state;

// Load: an optimizer that continues where es was
std::ifstream file("run.state", std::ios::binary);
const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
ModularCMAES resumed = serialization::load<ModularCMAES>(bytes);
```

`save<T>` and `load<T>` take the other objects too (`parameters::Settings`, `Population`, ...), and
a module by its base class (`load<std::shared_ptr<sampling::Sampler>>`). A state that cannot be
loaded raises `serialization::Error` (a `std::invalid_argument`). A restart criterion defined in
Python can only be saved and loaded through the Python package.
