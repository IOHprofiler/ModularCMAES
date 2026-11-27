# Working on ModularCMAES

modcma is a modular CMA-ES:
* the C++ core is in `include/` and `src/`;
* the nanobind bindings are in `bindings/`;
* the Python package is `modcma/`, the C++ API under `modcma.c_maes`, also `import modcma`.

The rules for every change are in `CONTRIBUTING.md`; read it first. This file adds how the work is
organized and the habits that keep results reproducible.

## Scope

* Only the C++ implementation and its bindings. The old pure-Python implementation lives on the
  branch `legacy-python`; never change it.
* `~/code/ModCMA-Journal-paper` (the framework paper, under review) is read-only. Use it as a
  source and cite it, but do not edit it.
  * Where the code and the paper disagree, say so in the PR description under a heading
    `## Paper`: what the paper says (file and line) and what the code does.
  * Open an issue only when the code or the docs have to change because of it. The board holds
    work on the code.
  * The paper is revised after the 2.0 release, from these PR sections.

## Where the work is

* The work is tracked as GitHub issues on the project board
  https://github.com/orgs/IOHprofiler/projects/10. The board is private; the issues are public.
* **Labels:**
  * `bug`: one fix per issue;
  * `module`: a new option of `Modules`, with a full implementation spec;
  * `framework`: a feature of `Settings` or the API;
  * `decision`: needs the maintainer;
  * `research`.
* **Board fields:**
  * Status: Backlog → Ready → In progress → In review → Done;
  * Kind;
  * Release: 2.0.0 or 2.x;
  * Size: S, M, L, XL.
* **Keep the board current.** Move the card to In progress when you start, to In review when the
  PR is open, and to Done when it is merged. A new finding becomes an issue on the board with all
  four fields set; do not just mention it in a PR. The project id is `PVT_kwDOAp7Dkc4BtJmG`;
  `gh project field-list 10 --owner IOHprofiler` lists the field and option ids.
* `dev/overhaul-2.0.md` is the frozen record of the 2.0 overhaul. Code comments cite its sections
  (T0.1, WP3, F2, ...). Do not add new tasks there.

## How a change is made

1. **Branch and PR:** one branch and one PR per issue, from `master`. The PR body says
   `Closes #N`. Do not stack PRs on other PRs: merged into their base branch instead of master,
   they never reached master.
   * **The PR body starts with a TL;DR**, before any other section, in at most three short lines:
     * what the PR changes;
     * whether results change (golden fixtures bit-identical, or regenerated and why);
     * the decisions it needs from the maintainer, or "no decisions".

     The details follow it.
2. **Commits and merges:**
   * Commit and push your branch, and open the PR.
   * Never push to `master`.
   * Merge only when the maintainer says so.
3. **Open decisions:** a methodological choice is not yours to make: a default constant, which
   paper to follow, a change of what a module does. Write it up with a recommendation (an issue
   with the `decision` label, or a question to the maintainer) and wait.
4. **Check the docs before the PR is opened.** Go through the documentation and look for
   anything the change makes wrong or incomplete, and update it in the same PR:
   * `docs/` (getting started, the guide, the compatibility page, the deep dives, the examples);
   * `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`;
   * the docstrings in `bindings/`, and the generated stubs and option tables.

   Search for the names, constants and behaviours that changed (`grep -rn`), not only the pages
   you touched. Look for "known issue" notes and F-numbers that the change resolves. The PR
   description lists the pages that were updated.
5. **Check every algorithmic change against its source:** the paper of the module, Hansen's
   tutorial (2016), pycma, `cmaes`.
   * Cite the source in a code comment and in the docstring.
   * When the paper and the reference implementation differ, say which one you follow and why.
   * Do not trust a formula from memory: read the source.

## Results: golden fixtures and the statistical tier

* `tests/golden.py` stores bit-exact trajectories of 271 configurations per toolchain
  (`tests/data/golden/`): a pairwise covering set of all module options plus special cases.
* **A change that is not meant to change results must leave every fixture bit-identical:**
  `python -m tests.golden check --drivers all`.
* **A change that does change results** needs:
  * a reason in the PR;
  * a statistical comparison on BBOB between master and the branch: `tests/statistical.py`, run in
    two environments (master installed non-editable in a separate venv; run the script with
    `python -I` so the checkout does not shadow the installed package);
  * then the regeneration of all three fixtures, below. Never regenerate a fixture just to make a
    test pass.
* **Adding an option to a `Modules` enum** changes the pairwise enumeration, and so the fixtures.
  Append new enum values last, so the existing values keep their numbers.
* **The three fixtures:**
  * **darwin**: `python -m tests.golden write` on macOS.
  * **linux gcc 13**: in `docker run --platform linux/amd64 ubuntu:24.04`. Install the package
    from a tarball of `git ls-files`. Copy `tests/` to a separate directory and run
    `python -m tests.golden write --fixture ...` there. Set `generated_from_commit` by hand: the
    container has no git.
  * **linux gcc 15**: the CI artifact `golden-fixture-ubuntu-latest` of the PR's Unittest
    workflow.
* **Ask-tell gives the same results as `run()`**, bit for bit, for every module (the golden check
  runs every driver).

## Building and testing

* **Build:** `uv pip install --python .venv/bin/python -e .`. The venv has no pip. After C++
  changes, rebuild before running the Python tests.
* **Tests:**
  * `python -m unittest discover -s tests -t .`;
  * the C++ tests: `cmake -S . -B build && cmake --build build && (cd build && ctest)`.
* **Bindings:** every bound name has a numpy-style docstring.
  * After changing the bindings, run `python tools/gen_stubs.py`, then `python tools/gen_docs.py`
    (the option tables of the guide), and commit the stubs.
  * `tests/test_c_stubs.py` fails on an undocumented name.
* **Docs:** `zensical build --strict` with `docs/requirements.txt`. Every Python block in the docs
  is executed by `tests/test_c_docs.py`.
* **Performance:** a change in the hot path reports `modcma_bench` before and after (min of at
  least 5 repetitions).
* **CI:**
  * Linux and Windows run on every PR. macOS runs only on `master` and releases, so check macOS
    locally.
  * GitHub's GraphQL API sometimes returns 503; use `gh api` (REST) and retry.

## Code conventions

* Signatures take mu before lambda, in C++ and Python.
* **The user-facing Python API stays source-compatible:** `Settings`, `ModularCMAES`, `fmin`, the
  options of `Modules`, the top-level names (`set_seed`, `settings_from_dict`, ...).
  * A renamed name keeps a deprecated alias (a `DeprecationWarning` that points at the caller).
  * A removed name raises an error that says what to use instead.
  * `tests/api_dump.py` compares against the 1.x API; an accepted break goes into its `ACCEPTED`
    list with a reason.
* **The bindings of internals may change without deprecation.** This covers the methods and helpers
  of the samplers, the step-size and selection classes, and the like: they are used for debugging,
  if at all. Note the change in the changelog, and add the `ACCEPTED` reason "internal".
* **A new module:**
  * is off by default, and leaves the results of existing configurations unchanged;
  * works with ask-tell, and does its extra evaluations as a phase of the generation;
  * has Settings that adjust or reject incompatible combinations;
  * appears in `get_configspace`;
  * has a docstring;
  * is listed on the compatibility page `docs/guide/compatibility.md`.
* **C++ style:** `.clang-format` (LLVM base, 4 spaces, 100 columns). Python: PEP 8, 100 columns.
* **Line endings:** `include/restart_strategy.hpp` uses CRLF; keep it when editing.
* **Changelog:** user-visible changes get a line in `CHANGELOG.md`, under the unreleased 2.0.0
  until the release.

## Writing

Docs, docstrings, issues and PR descriptions: plain, precise English. Short sentences, no marketing,
numbers with their source. Say what was verified and what was not.
