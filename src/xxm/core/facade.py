"""Host-side results and collection orchestration for model facades."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import typing

import jax

from xxm.core import batch
from xxm.core.data import Dataset, DatasetCollection

Facade_co = typing.TypeVar('Facade_co', covariant=True)

Posterior_co = typing.TypeVar('Posterior_co', bound=batch.Batchable, covariant=True)

Inferred_co = typing.TypeVar(
    'Inferred_co',
    bound='Inferred[object, batch.Batchable]',
    covariant=True,
)

CoreT = typing.TypeVar('CoreT')
CoreT_co = typing.TypeVar('CoreT_co', covariant=True)


class _Facade(typing.Protocol[CoreT_co]):
    @property
    def model(self) -> CoreT_co: ...


FacadeT = typing.TypeVar('FacadeT')
InferredT = typing.TypeVar('InferredT', bound='Inferred[object, batch.Batchable]')
ResultT = typing.TypeVar('ResultT')
Posterior_co = typing.TypeVar(
    'Posterior_co',
    bound=batch.Batchable,
    covariant=True,
)
Progress = bool | str


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(typing.Generic[Facade_co, Posterior_co]):
    """Facade, pooled posterior, and source datasets."""

    model: Facade_co
    posterior: Posterior_co
    data: DatasetCollection

    def __post_init__(self) -> None:
        expected = (self.data.partition.total,)
        actual = self.posterior.batch_shape

        if actual != expected:
            raise ValueError(f'Expected posterior batch {expected}, got {actual}')

    def _session(
        self,
        session: int | None = None,
    ) -> tuple[Posterior_co, Dataset]:
        """Select one session from the pooled posterior."""

        dataset = self.data.get(session)
        index = 0 if session is None else session

        posterior = batch.select(
            self.posterior,
            self.data.partition.get_slice(index),
        )

        return posterior, dataset


class Fit(typing.NamedTuple, typing.Generic[Inferred_co]):
    """Fitted result with objective trace."""

    inferred: Inferred_co
    objective_trace: jax.Array


class _FacadeOperations(typing.Protocol[Inferred_co]):
    """Internal dynamic dispatch boundary for facade conventions."""

    def infer(
        self,
        data: DatasetCollection,
        **kwargs: typing.Any,
    ) -> Inferred_co: ...

    def fit(
        self,
        data: DatasetCollection,
        **kwargs: typing.Any,
    ) -> Fit[Inferred_co]: ...


def _with_progress(
    results: typing.Iterable[ResultT],
    *,
    total: int,
    progress: Progress,
) -> typing.Iterable[ResultT]:
    if progress is False:
        return results

    from tqdm.auto import tqdm

    description = progress if isinstance(progress, str) else 'sessions'

    return tqdm(
        results,
        total=total,
        desc=description,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class FacadeCollection(typing.Generic[FacadeT, InferredT]):
    """Independent facades, ordered by session."""

    models: tuple[FacadeT, ...]

    def get(self, session: int | None = None) -> FacadeT:
        """Get one model, requiring an index unless singleton."""
        if session is None:
            if len(self.models) != 1:
                raise ValueError('Session index required for multiple models')
            return self.models[0]

        return self.models[session]

    def map(
        self,
        data: DatasetCollection,
        function: typing.Callable[[FacadeT, DatasetCollection], ResultT],
        *,
        progress: Progress = False,
        max_workers: int | None = None,
    ) -> tuple[ResultT, ...]:
        """Apply an operation independently to each session.

        Progress tracks completed host-side operations.
        Results are always returned in session order.
        """
        pairs = tuple(zip(self.models, data.datasets, strict=True))

        def apply(pair: tuple[FacadeT, Dataset]) -> ResultT:
            facade_model, dataset = pair
            return function(facade_model, DatasetCollection((dataset,)))

        if max_workers is None or max_workers == 1:
            return self._map_sequential(pairs, apply, progress)

        else:
            return self._map_threaded(pairs, apply, progress, max_workers)

    def _map_sequential(
        self,
        pairs,
        apply: typing.Callable[[tuple[FacadeT, Dataset]], ResultT],
        progress: Progress,
    ) -> tuple[ResultT, ...]:

        results = (apply(pair) for pair in pairs)

        return tuple(
            _with_progress(
                results,
                total=len(pairs),
                progress=progress,
            )
        )

    def _map_threaded(
        self,
        pairs,
        apply: typing.Callable[[tuple[FacadeT, Dataset]], ResultT],
        progress: Progress,
        max_workers: int,
    ) -> tuple[ResultT, ...]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        ) as pool:
            futures = {
                pool.submit(apply, pair): index for index, pair in enumerate(pairs)
            }

            completed: dict[int, ResultT] = {}

            for future in _with_progress(
                concurrent.futures.as_completed(futures),
                total=len(pairs),
                progress=progress,
            ):
                completed[futures[future]] = future.result()

        return tuple(completed[index] for index in range(len(pairs)))

    def infer(
        self,
        data: DatasetCollection,
        *,
        progress: Progress = False,
        max_workers: int | None = None,
        **kwargs: typing.Any,
    ) -> tuple[InferredT, ...]:
        """Infer every session independently."""
        return self.map(
            data,
            lambda model, subset: typing.cast(
                _FacadeOperations[InferredT], model
            ).infer(subset, **kwargs),
            progress=progress,
            max_workers=max_workers,
        )

    def fit(
        self,
        data: DatasetCollection,
        *,
        progress: Progress = False,
        max_workers: int | None = None,
        **kwargs: typing.Any,
    ) -> tuple[Fit[InferredT], ...]:
        """Fit every session independently."""
        return self.map(
            data,
            lambda model, subset: typing.cast(_FacadeOperations[InferredT], model).fit(
                subset, **kwargs
            ),
            progress=progress,
            max_workers=max_workers,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class CombinableFacadeCollection(
    FacadeCollection[FacadeT, InferredT],
    typing.Generic[FacadeT, InferredT, CoreT],
):
    """Independent facades with a numerical composition strategy."""

    combine_fn: typing.Callable[..., CoreT] = dataclasses.field(repr=False)

    def _combine_inferred(
        self,
        key: jax.Array,
        inferred: tuple[InferredT, ...],
        **kwargs: typing.Any,
    ) -> FacadeT:
        """Combine independently inferred models."""

        core_models: tuple[CoreT, ...] = tuple(
            typing.cast(_Facade[CoreT], result.model).model for result in inferred
        )

        if len(core_models) == 1:
            combined = core_models[0]

        else:
            posteriors = tuple(result.posterior for result in inferred)

            combined = self.combine_fn(
                key,
                core_models,
                posteriors,
                **kwargs,
            )

        facade_factory = typing.cast(
            typing.Callable[[CoreT], FacadeT],
            type(self.models[0]),
        )

        return facade_factory(combined)

    def combine(
        self,
        key: jax.Array,
        data: DatasetCollection,
        *,
        infer_kwargs: typing.Mapping[str, typing.Any] | None = None,
        progress: Progress = False,
        max_workers: int | None = None,
        **combine_kwargs: typing.Any,
    ) -> FacadeT:
        """Infer independently, then combine."""

        inferred = self.infer(
            data,
            progress=progress,
            max_workers=max_workers,
            **(infer_kwargs or {}),
        )

        return self._combine_inferred(
            key,
            inferred,
            **combine_kwargs,
        )
