# Matrix adaptation internals

The interface `matrix_adaptation::Adaptation` (`include/matrix_adaptation.hpp`) and what each of the
eight adaptations keeps. The cached factorizations: `factor_version`, the metric computed on
request (`metric()`, `metric_d`, `metric_B`, `metric_version`), who reads it (the restart
criteria, the record of a finished run), and when each is computed again. The Woodbury update of
M⁻¹ of MA-ES (#74, #159), its exact recomputation, and the limits in dimension. LAPACK and BLAS
from a threshold dimension on (`include/lapack.hpp`, the `constants`). The Mahalanobis distance in
the current distribution (`Adaptation::distance`, through `invert_y`), which repelling uses. What
the adaptations do is on the pages [Matrix adaptation](../guide/matrix-adaptation.md) and
[Matrix adaptations](../deep-dives/matrix-adaptations.md).

!!! note "Planned"
    This page is an outline. Its three parts are written in step 2 of
    [#186](https://github.com/IOHprofiler/ModularCMAES/issues/186).

## How it works

## How to change it

## When it breaks
