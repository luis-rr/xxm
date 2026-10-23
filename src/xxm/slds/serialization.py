"""Explicit storage codec for the SLDS composite posterior."""

from typing import cast

from xxm.core.io import chains, hdf5
from xxm.slds.core import Posterior


def posterior_to_storage(posterior: Posterior) -> dict[str, object]:
    return {
        'discrete': chains.discrete_to_storage(posterior.discrete),
        'continuous': chains.gaussian_to_storage(posterior.continuous),
    }


def posterior_from_storage(value: object) -> Posterior:
    params = cast(dict[str, object], value)
    return Posterior(
        chains.discrete_from_storage(params['discrete']),
        chains.gaussian_from_storage(params['continuous']),
    )


def save_posterior(
    posterior: Posterior, path: hdf5.PathLike, *, overwrite: bool = False
) -> None:
    hdf5.save(
        path,
        {'posterior': posterior_to_storage(posterior)},
        object_type='xxm.slds.Posterior',
        overwrite=overwrite,
    )


def load_posterior(path: hdf5.PathLike) -> Posterior:
    return posterior_from_storage(hdf5.load(path)['posterior'])
