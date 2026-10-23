"""Host-side HDF5 storage for canonical parameters and result payloads."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np

PathLike = str | os.PathLike[str]


def _write(parent: h5py.Group, name: str, value: object) -> None:
    if isinstance(value, (jax.Array, np.ndarray)):
        node = parent.create_dataset(name, data=np.asarray(jax.device_get(value)))
        node.attrs['kind'] = 'array'
        return

    node = parent.create_group(name)
    if isinstance(value, Mapping):
        node.attrs['kind'] = 'mapping'
        for key, item in value.items():
            _write(node, key, item)
    elif isinstance(value, tuple):
        node.attrs['kind'] = 'tuple'
        node.attrs['length'] = len(value)
        for index, item in enumerate(value):
            _write(node, str(index), item)
    elif value is None:
        node.attrs['kind'] = 'none'
    elif type(value) in (str, bool, int, float):
        node.attrs['kind'] = type(value).__name__
        node.attrs['value'] = value
    else:
        raise TypeError(f'Unsupported HDF5 value type: {type(value).__name__}')


def _read(node: h5py.Group | h5py.Dataset) -> Any:
    """Decode the file boundary; callers supply the mathematical schema."""
    if isinstance(node, h5py.Dataset):
        return jnp.asarray(node[()])
    kind = node.attrs['kind']
    if kind == 'mapping':
        return {
            name: _read(cast(h5py.Group | h5py.Dataset, node[name]))
            for name in cast(Iterable[str], node)
        }
    if kind == 'tuple':
        return tuple(
            _read(cast(h5py.Group | h5py.Dataset, node[str(i)]))
            for i in range(cast(int, node.attrs['length']))
        )
    if kind == 'none':
        return None
    if kind == 'str':
        return str(node.attrs['value'])
    if kind == 'bool':
        return bool(node.attrs['value'])
    if kind == 'int':
        return int(cast(int, node.attrs['value']))
    if kind == 'float':
        return float(cast(float, node.attrs['value']))
    raise TypeError(f'Unsupported HDF5 node kind: {kind!r}')


def save(
    path: PathLike,
    payload: Mapping[str, object],
    *,
    object_type: str,
    overwrite: bool = False,
) -> None:
    """Close a complete temporary file before atomically publishing it."""
    destination = Path(path)
    fd, temporary = tempfile.mkstemp(
        prefix=f'.{destination.name}.',
        suffix='.tmp',
        dir=destination.parent,
    )
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as file:
            file.attrs['format_version'] = 1
            file.attrs['object_type'] = object_type
            try:
                file.attrs['xxm_version'] = version('xxm')
            except PackageNotFoundError:
                file.attrs['xxm_version'] = 'unknown'
            for name, value in payload.items():
                _write(file, name, value)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path: PathLike) -> dict[str, Any]:
    with h5py.File(path, 'r') as file:
        return {
            name: _read(cast(h5py.Group | h5py.Dataset, file[name]))
            for name in cast(Iterable[str], file)
        }


def save_model_params(
    path: PathLike,
    params: Mapping[str, object],
    *,
    model_type: str,
    overwrite: bool = False,
) -> None:
    save(path, {'params': params}, object_type=model_type, overwrite=overwrite)


def load_model_params(path: PathLike) -> dict[str, Any]:
    return load(path)['params']


def save_inferred(
    path: PathLike,
    *,
    family: str,
    model_type: str,
    params: Mapping[str, object],
    posterior: Mapping[str, object],
    data: object,
    overwrite: bool = False,
) -> None:
    save(
        path,
        {
            'model': {'object_type': model_type, 'params': params},
            'posterior': posterior,
            'data': data,
        },
        object_type=f'xxm.{family}.Inferred',
        overwrite=overwrite,
    )


def load_inferred(path: PathLike) -> tuple[str, Any, Any, Any]:
    payload = load(path)
    model = payload['model']
    return model['object_type'], model['params'], payload['posterior'], payload['data']
