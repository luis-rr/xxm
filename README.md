# xxm

A lightweight research library of state-space models and their inference routines. Implemented in JAX.

Inspired by [`ssm`](https://github.com/lindermanlab/ssm) and [`dynamax`](https://github.com/probml/dynamax).

> [!NOTE]
> Early development: the API will change.

## Models

See the notebooks in [`notebooks/examples`](notebooks/examples) for examples of how the models are used (sampling, fitting, inference). Every model supports two emission types: Gaussian and Poisson.

Currently implemented models:

| Model | Inference and learning |
|---|---|
| HMM | exact inference, EM |
| Autoregressive HMM (ARHMM) | exact inference, EM |
| LDS | exact (Gaussian) or Laplace (Poisson) inference, EM / Laplace-EM |
| SLDS | structured variational or Laplace inference, variational EM / Laplace-EM |

## Installation

Requires Python 3.11+.

```bash
pip install -e .            # core
pip install -e '.[plot]'    # with plotting helpers
```

## Development

Run tests with:

```bash
pytest
```

Format and lint with:

```bash
ruff format .
ruff check .
```
