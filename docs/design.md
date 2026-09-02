# xxm design

## Purpose

`xxm` is a lightweight JAX package for statistical modelling, with an initial focus on hidden Markov models, linear dynamical systems, switching linear dynamical systems, and related models.

The package is inspired by functionality in [`ssm`](https://github.com/lindermanlab/ssm) and [`dynamax`](https://github.com/probml/dynamax), but aims to remain smaller, more explicit, and easier to inspect. The goal is not to provide a general probabilistic-programming framework, but clear implementations of a focused family of statistical models and the reusable mathematical machinery that supports them.

`xxm` is still pre-release. APIs and architectural assumptions may change directly when the design improves. Backwards compatibility and deprecation machinery are not priorities until a stable API is intentionally declared.

This document is the source of truth for architecture, engineering principles, coding style, conventions, and testing practices.

Mathematical definitions, derivations, and algorithmic explanations belong in `docs/math.md`.

## Design philosophy

The codebase follows a few general rules:

* Keep the package small, explicit, and easy to understand.
* Prioritize mathematical and numerical correctness and interpretability.
* Prefer simple implementations over clever abstractions.
* Avoid unnecessary indirection, framework-like machinery, and runtime dispatch.
* Extend existing concepts instead of creating parallel ways of doing the same thing.
* Add dependencies only when they provide substantial value.
* Keep important inference and optimization algorithms inspectable.
* Fail loudly on invalid states rather than silently repairing scientifically ambiguous inputs.
* Prefer immutable data and pure transformations.
* Prefer composition and generic functions over inheritance.
* Prefer symmetry where it reflects the mathematics, but do not force it where the mathematics differs.
* Keep state inspectable and unsurprising.
* Prefer clarity until measured performance justifies additional complexity.

Abstract repeated structure only when it represents a genuine shared mathematical operation and improves clarity or extensibility. Similar-looking control flow alone is not sufficient reason for abstraction.

Protocols are useful for small capability-oriented interfaces, such as posterior moments or emission capabilities. Prefer focused protocols over broad model hierarchies.

## Architecture

### Mathematical and model organization

At the lowest level are reusable mathematical components:

* probability distributions and conditional distributions;
* discrete and Gaussian chain potentials and marginals;
* reusable emission, initial-state, transition, and dynamics components;
* generic optimization and fitting routines.

Model families compose these pieces:

* an HMM uses a discrete latent chain and discrete-state emissions;
* an LDS uses a Gaussian latent chain and continuous-state emissions;
* an SLDS combines discrete and Gaussian latent structure through switching dynamics.

Within each model family, responsibilities remain distinct:

* `model` defines the generative model and its parameters;
* `inference` computes posterior quantities for fixed model parameters;
* `learning` updates model parameters and orchestrates repeated inference and fitting;
* `init` constructs useful starting models from data or simple assumptions.

Generative-model logic should live on the mathematical components where it is intrinsic. Examples include sampling, conditional distributions, conversion to chain potentials, and sufficient-statistic-based parameter updates.

Inference should orchestrate those components rather than duplicate their generative mathematics. Reusable optimization belongs below model-family learning code so that learning modules remain close to the statistical algorithms they implement.

The same model components should be used for simulation, inference, and learning whenever the mathematics permits it.

### Repository structure

The repository uses a standard `src` layout:

```text
xxm/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
│
├── src/
│   └── xxm/
│
├── tests/
├── docs/
│   ├── design.md
│   └── math.md
└── notebooks/
```

The package layout is:

```text
src/xxm/
├── __init__.py
├── py.typed
│
├── core/
│   ├── __init__.py
│   ├── affine.py
│   ├── align.py
│   ├── posteriors.py
│   │
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── discrete.py
│   │   └── gaussian.py
│   │
│   ├── dists/
│   │   ├── __init__.py
│   │   ├── categorical.py
│   │   ├── gaussian.py
│   │   └── poisson.py
│   │
│   ├── emissions/
│   │   ├── __init__.py
│   │   ├── continuous.py
│   │   ├── discrete.py
│   │   └── discrete_ar.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── discrete.py
│   │   └── gaussian.py
│   │
│   └── optim/
│       ├── __init__.py
│       ├── gaussian.py
│       ├── poisson.py
│       ├── newton.py
│       └── loop.py
│
├── hmm/
│   ├── __init__.py
│   ├── model.py
│   ├── inference.py
│   ├── learning.py
│   └── init.py
│
├── lds/
│   ├── __init__.py
│   ├── model.py
│   ├── inference.py
│   ├── learning.py
│   └── init.py
│
└── slds/
    ├── __init__.py
    ├── model.py
    ├── inference.py
    ├── learning.py
    └── init.py
```

`core` contains reusable mathematical building blocks. `hmm`, `lds`, `slds`, and future sibling packages contain complete model families built from those components.

### Dependency direction

Dependencies should remain simple and mostly one-way:

```text
core
 ↓
hmm / lds / slds / ...
```

More precisely:

* `core` does not depend on complete model families;
* HMM and LDS depend on reusable mathematical components in `core`;
* HMM and LDS should not depend on one another;
* SLDS initialization may compose HMM or LDS initialization and fitting routines where this is mathematically useful;
* model components do not depend on inference algorithms;
* inference does not depend on learning;
* learning may depend on model and inference;
* plotting, data loading, and project-specific analysis do not belong in the core modelling dependency graph.

Circular dependencies should be treated as an architectural problem rather than worked around with dynamic imports.

### Public API

`xxm` is research-oriented and does not strongly hide its implementation. Lower-level mathematical components remain accessible for inspection, experimentation, and extension.

Package `__init__.py` files and explicit `__all__` definitions provide the recommended and discoverable API.

Principles:

* `__init__.py` files act as API manifests rather than implementation modules.
* Recommended symbols are re-exported from semantic namespaces such as `xxm.hmm`, `xxm.lds`, `xxm.slds`, and relevant `xxm.core` subpackages.
* Deeper modules remain ordinary usable Python modules.
* Highly technical implementation details may use leading underscores where useful.
* Internal code imports definitions from their owning modules rather than back through the top-level `xxm` facade.
* Avoid lazy imports, dynamic exports, generated namespaces, automatic discovery, and import-time side effects.
* Recommended APIs should be easy to discover through normal IDE completion and static analysis.
* Until the first stable release, APIs may change directly.

Model-family namespaces should expose the functions and objects needed to construct, initialize, infer, fit, sample from, and inspect the model.

For example:

```python
import xxm.hmm as hmm

model = hmm.init_gaussian(...)
fit = hmm.fit_em(model, observations, ...)
posterior, log_normalizer = hmm.infer_exact(fit.model, observations)
```

## JAX and model representation

`xxm` is JAX-first. All main classes must be jax-friendly and all main algorithms jax-jittable.

### Model representation

Model objects should be lightweight immutable PyTrees.

`NamedTuple` is a good default when it naturally describes the object. Other immutable PyTree-friendly structures are appropriate when they provide a concrete benefit.

Model state should represent the mathematical model directly rather than an optimizer-specific encoding.

Transformations return updated objects rather than mutating existing ones. This includes:

* parameter fitting;
* permutation;
* dtype conversion;
* selection;
* covariance regularization;
* optimization steps.

Hidden mutable numerical state should not be required by core algorithms.

### Probability representation

Represent quantities in the parameterization natural to the model and computation:

* categorical probabilities are stored as probabilities;
* log probabilities are used locally where numerical stability requires them;
* Gaussian parameters use means and covariances or canonical potentials according to the mathematical object represented;
* Poisson log-linear models may store log rates or linear predictors, with rates exposed as derived quantities;
* optimizer-specific unconstrained parameterizations remain local to optimization code.

Do not impose a package-wide rule that all probability-related quantities must live in log space.

### JAX execution

* Use `jax.Array` for numerical array annotations.
* Prefer pure functions and immutable PyTree-compatible objects.
* Pass model parameters and numerical state explicitly.
* Avoid hidden global numerical state.
* Write numerical code that composes naturally with `jax.jit`, `jax.vmap`, and automatic differentiation.
* Avoid Python control flow that depends on traced array values.
* Keep array shapes and static configuration clear.
* Do not enable global JAX configuration such as x64 mode at package import.
* Core routines should be JIT-compatible, but compilation policy should generally remain under caller control rather than being hidden behind internal `jax.jit` wrappers.

Host-side orchestration is appropriate where necessary, but should remain separate from compiled numerical kernels. Progress reporting, multi-start bookkeeping, and user-facing convenience logic should not obscure the numerical algorithm.

### Randomness

Random functions take an explicit JAX key, conventionally as the first argument when randomness is intrinsic to the operation.

Functions split keys locally and deterministically. Reusing a key for independent random operations is a bug.

### Array conventions

Unless a mathematical object requires otherwise:

* sequence or time is the leading structural axis;
* state is the next explicit axis for state-indexed sequence quantities;
* variable and output dimensions are trailing axes;
* distribution batch dimensions precede event dimensions;
* matrices use trailing `(output_dim, input_dim)` or `(variable_dim, variable_dim)` axes.

Shape comments are encouraged where they make non-obvious tensor algebra easier to inspect.

## Naming and signatures

Names should follow standard statistical terminology while remaining readable to someone comparing the implementation with the mathematics.

### Vocabulary

* Prefer descriptive names such as `forward_probs` over one-letter algorithm names such as `alpha`.
* Use `prob` and `probs` consistently as abbreviations.
* Use zero-based indexing in mathematical explanations and implementation comments.
* Prefer `num_states`, `num_steps`, `num_lags`, `latent_dim`, `input_dim`, `output_dim`, and `variable_dim` consistently according to meaning.
* Distinguish parameters, moments, potentials, messages, and marginals rather than using generic names such as `values` where the distinction matters.

### Function families

Use consistent verbs:

* `init_*` — construct an initial model from data or simple assumptions;
* `infer_*` — compute posterior quantities for fixed model parameters;
* `fit_*` — run an iterative learning procedure;
* `*_step` — perform one learning or optimization iteration;
* `sample` — draw from a distribution or generative model;
* `log_prob`, `log_likelihood`, `log_joint` — evaluate the corresponding mathematical quantity;
* `compute_*` — construct a derived mathematical object when a more specific verb is not clearer.

### Parameter ordering

Prefer consistent argument order across model families. Not every function uses every category, but comparable functions should align.

Algorithm controls such as iteration counts, tolerances, line-search limits, regularization strengths, and progress settings should generally be keyword-only once the primary mathematical arguments are supplied.

## Numerical fitting and optimization

Low-level fitting routines should expose the statistical problem directly.

* Prefer closed-form solutions when available.
* Iterative optimization should expose meaningful convergence controls.
* Regularization should be explicit in the relevant fitting API.
* Numerical stabilization such as jitter or covariance floors should be explicit and scientifically interpretable.
* Avoid silent fallback to a different algorithm.
* Degenerate states or zero effective sample counts should have deliberate, tested behavior rather than accidental NaNs or hidden repairs.

Generic optimization machinery such as Newton search and line search should remain separate from model-family learning algorithms.

## Testing

Tests are part of the scientific design, not only software verification.

The suite should remain deterministic, readable, and fast enough to run routinely.

### Mathematical tests

* Test known analytical solutions.
* Compare chain inference with dense reference calculations where practical.
* Test sufficient-statistic and fitting formulas against hand-computable cases.
* Test limiting and degenerate cases explicitly.
* Prefer small arrays that make expected results easy to inspect.

### Model-family tests

* Test initialization, inference, one learning step, fitting, sampling, and permutation where applicable.
* Test reductions between model families when mathematically exact, such as a one-state switching model reducing to a non-switching model.
* Test objective or ELBO behavior where monotonicity is theoretically expected.

### JAX tests

* Test representative numerical routines under `jax.jit`.
* Test eager and JIT results for numerical agreement.
* Test `vmap` or differentiation where the API is intended to support them.
* Do not require every trivial accessor to have its own JIT test.

### API tests

* Test imports through recommended package namespaces.
* Ensure important examples exercise the recommended API.

### General testing rules

* Use fixed random keys.
* Prefer exact or analytical references over comparing one implementation with another implementation of the same algorithm.
* Keep tolerances explicit.
* Add regression tests for real bugs that could plausibly recur.
* Avoid large stochastic simulations when a small deterministic case tests the same property.
* Avoid excessive mocking and test-framework machinery.
* Do not chase a coverage percentage as a goal in itself.

## Typing and coding style

Typing should improve clarity and catch structural mistakes without turning the codebase into a typing experiment.

* Use inline type annotations where they clarify interfaces or important internal boundaries.
* Use `jax.Array` for array values.
* Use `typing.Self`, generics, and protocols when they materially improve correctness or reuse.
* Avoid complicated type-level encodings of array shapes.
* Avoid `Any` unless there is a strong reason.
* Generic APIs should type-check under Pyright.
* Ship a `py.typed` marker.

Static typing does not replace numerical validation or tests.

The implementation favors readable, idiomatic Python and static modular design.

* Prefer single quotes.
* Use blank lines to separate logical steps.
* Keep functions focused and reasonably small.
* Avoid deeply nested conditionals; move distinct logic into focused functions.
* Prefer module imports when they make implementation ownership clearer.
* Direct symbol imports are appropriate for vocabulary used heavily in annotations or signatures.
* Prefer absolute imports across package boundaries.
* Avoid import-time side effects.
* Avoid unnecessary classes; use light OO around immutable mathematical objects.
* Keep equations and tensor operations visible rather than wrapping every expression in helpers.
* Use comments for non-obvious reasoning, shape conventions, or numerical considerations rather than narrating obvious code.
* Code docstrings should connect mathematical notation to implementation names.

Docstrings should be concise for straightforward objects and more detailed where they connect implementation with mathematical notation or explain a non-obvious numerical algorithm.

## Documentation

Documentation has distinct roles:

* `README.md` — installation, a minimal introduction, and the shortest path to using the package;
* `docs/design.md` — engineering principles, architecture, API conventions, and development practices;
* `docs/math.md` — mathematical models, derivations, inference, and learning algorithms;
* notebooks — executable examples and exploratory demonstrations.

Avoid duplicating long mathematical derivations in both `math.md` and Python docstrings. Docstrings should provide enough notation to connect code with the mathematical reference.

## Performance

JAX compilation changes the cost model of the package, so performance work should distinguish compilation cost from execution cost.

* Avoid repeated recompilation caused by accidental changes in static structure.
* Keep model PyTrees structurally stable across fitting iterations.
* Use batching and `vmap` where they naturally express independent repeated computation.
* Benchmark representative workloads before introducing complex performance machinery.
* Keep compilation outside tight user loops where practical without internally forcing compilation policy.
* Do not prefer a more sophisticated implementation merely because it appears more vectorized.

Readability should only be traded for performance when the gain is meaningful and the mathematical behavior remains testable and documented.

## Scope and dependencies

`xxm` models statistical arrays. It is not intended to become a general data-management or experimental-analysis package.

The dependency surface should remain small. A dependency is justified when it provides substantial scientific or engineering value that would be costly or error-prone to reproduce locally.

Core modelling code should avoid dependencies on:

* plotting libraries;
* dataframe libraries;
* experiment-management frameworks;
* project-specific data containers.

Real-data workflows may motivate adjacent utilities, but they should enter `xxm` only when they are broadly useful for fitting, diagnosing, or interpreting the supported models.

Examples that may eventually belong include:

* model-specific diagnostic summaries;
* state-alignment helpers;
* posterior predictive utilities;
* concise model visualization helpers.

Project-specific loading, preprocessing, experiment metadata, publication plotting, and bespoke analysis pipelines should remain downstream.

Real-data applications should pressure-test the modelling API rather than expand `xxm` into a general analysis framework.

## Development tooling

The local development loop should be small and reproducible.

The standard tools are:

* Ruff for linting and formatting;
* Pyright for static type checking;
* Pytest for tests;
* pre-commit for routine local checks;
* GitHub Actions for CI.

Avoid overlapping tools that solve the same problem.
