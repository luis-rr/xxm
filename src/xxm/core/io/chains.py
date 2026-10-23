"""Explicit codecs for the existing mathematical chain marginals."""

from typing import cast

import jax

from xxm.core.dists.discrete import DiscreteChainMarginals
from xxm.core.dists.gaussian import GaussianChainMarginals
from xxm.core.io import hdf5
from xxm.core.lengths import Lengths


def discrete_to_storage(posterior: DiscreteChainMarginals) -> dict[str, object]:
    return {
        'state_probs': posterior.state_probs,
        'pair_probs': posterior.pair_probs,
        'lengths': posterior.lengths.values,
    }


def discrete_from_storage(value: object) -> DiscreteChainMarginals:
    params = cast(dict[str, jax.Array], value)
    return DiscreteChainMarginals(
        params['state_probs'],
        params['pair_probs'],
        Lengths(params['lengths']),
    )


def gaussian_to_storage(posterior: GaussianChainMarginals) -> dict[str, object]:
    return {
        'means': posterior.means,
        'covariances': posterior.covariances,
        'cross_covariances': posterior.cross_covariances,
        'lengths': posterior.lengths.values,
    }


def gaussian_from_storage(value: object) -> GaussianChainMarginals:
    params = cast(dict[str, jax.Array], value)
    return GaussianChainMarginals(
        params['means'],
        params['covariances'],
        params['cross_covariances'],
        Lengths(params['lengths']),
    )


def save_discrete_posterior(
    posterior: DiscreteChainMarginals, path: hdf5.PathLike, *, overwrite: bool = False
) -> None:
    hdf5.save(
        path,
        {'posterior': discrete_to_storage(posterior)},
        object_type='xxm.DiscreteChainMarginals',
        overwrite=overwrite,
    )


def load_discrete_posterior(path: hdf5.PathLike) -> DiscreteChainMarginals:
    return discrete_from_storage(hdf5.load(path)['posterior'])


def save_gaussian_posterior(
    posterior: GaussianChainMarginals, path: hdf5.PathLike, *, overwrite: bool = False
) -> None:
    hdf5.save(
        path,
        {'posterior': gaussian_to_storage(posterior)},
        object_type='xxm.GaussianChainMarginals',
        overwrite=overwrite,
    )


def load_gaussian_posterior(path: hdf5.PathLike) -> GaussianChainMarginals:
    return gaussian_from_storage(hdf5.load(path)['posterior'])
