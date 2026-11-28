# The generation and ask–tell

The steps of `ModularCMAES::advance()` in detail: `queue`, `answer`, `dispatch` and the sources of
the queued points. The points besides the population: TPA's two points, the hill-valley tests of
repelling, the batches of sequential selection, and the center of mass that ends every run. The
budget: `Parameters::reserved_points()`, `restart_fits()` and `fit_lambda_to_budget()`, and TPA's
points, which are reserved also in a generation that does not evaluate them (#179). Why ask–tell
gives the results of `run()` bit for bit, and what breaks that. How to add a module that evaluates
the objective; this page will replace the section "A module that evaluates the objective" of
[Contributing](../contributing.md). What to do when a golden run differs in one driver only. What
`ask()` returns, seen from the user, is on the page [Ask–tell](../getting-started/ask-tell.md).

!!! note "Planned"
    This page is an outline. Its three parts are written in step 2 of
    [#186](https://github.com/IOHprofiler/ModularCMAES/issues/186).

## How it works

## How to change it

## When it breaks
