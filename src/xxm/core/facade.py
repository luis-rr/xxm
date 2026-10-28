"""Host-side facades, inference results, and collection orchestration."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import typing
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.batch import Batchable
from xxm.core.data import Dataset, DatasetCollection
from xxm.core.io.hdf5 import PathLike, save_model_params

Progress = bool | str

CoreT = typing.TypeVar('CoreT', bound=Batchable)
FacadeT = typing.TypeVar('FacadeT')
Facade_co = typing.TypeVar('Facade_co', covariant=True)
Posterior_co = typing.TypeVar('Posterior_co', bound=Batchable, covariant=True)
Inferred_co = typing.TypeVar(
    'Inferred_co', bound='Inferred[object, Batchable]', covariant=True
)
InferredT = typing.TypeVar('InferredT', bound='Inferred[object, Batchable]')
ResultT = typing.TypeVar('ResultT')


# ---------------------------------------------------------------------
# Facades and capabilities
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Facade(typing.Generic[CoreT]):
    """An unbatched numerical core and typed facade reconstruction."""

    core: CoreT

    def __post_init__(self) -> None:
        if self.core.batch_shape != ():
            raise ValueError('Facade requires an unbatched core model')

    def _with_core(self, core: CoreT) -> typing.Self:
        return dataclasses.replace(self, core=core)

    def to_params(self) -> Mapping[str, object]:
        """Return canonical parameters; concrete facades define the schema."""
        raise NotImplementedError

    def to_hdf5(self, path: PathLike, *, overwrite: bool = False) -> None:
        """Store this concrete facade's canonical parameters on the host."""
        family = type(self).__module__.removesuffix('.model')
        save_model_params(
            path,
            self.to_params(),
            model_type=f'{family}.{type(self).__name__}',
            overwrite=overwrite,
        )


class Permutable(Batchable, typing.Protocol):
    def permute_states(self, permutation: jax.Array) -> typing.Self: ...


PermutableT = typing.TypeVar('PermutableT', bound=Permutable)


class PermutableFacade(Facade[PermutableT], typing.Generic[PermutableT]):
    def permute_states(self, permutation: jax.Array) -> typing.Self:
        return self._with_core(self.core.permute_states(permutation))


class Alignable(Batchable, typing.Protocol):
    def align(self, alignment: Affine) -> typing.Self: ...


AlignableT = typing.TypeVar('AlignableT', bound=Alignable)


class AlignableFacade(Facade[AlignableT], typing.Generic[AlignableT]):
    def align(self, alignment: Affine) -> typing.Self:
        return self._with_core(self.core.align(alignment))


# ---------------------------------------------------------------------
# Individual inference and fitting results
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(typing.Generic[Facade_co, Posterior_co]):
    model: Facade_co
    posterior: Posterior_co
    data: DatasetCollection

    def __post_init__(self) -> None:
        expected = (self.data.partition.total,)
        if self.posterior.batch_shape != expected:
            raise ValueError(
                f'Expected posterior batch {expected}, got {self.posterior.batch_shape}'
            )

    def _session(self, session: int | None = None) -> tuple[Posterior_co, Dataset]:
        dataset = self.data.get(session)
        index = 0 if session is None else session
        posterior = batch.select(
            self.posterior,
            self.data.partition.get_slice(index),
        )
        return posterior, dataset


PermutableFacadeT = typing.TypeVar(
    'PermutableFacadeT', bound='PermutableFacade[typing.Any]'
)
PermutablePosterior_co = typing.TypeVar(
    'PermutablePosterior_co', bound=Permutable, covariant=True
)


class PermutableInferred(
    Inferred[PermutableFacadeT, PermutablePosterior_co],
    typing.Generic[PermutableFacadeT, PermutablePosterior_co],
):
    def permute_states(self, permutation: jax.Array) -> typing.Self:
        return dataclasses.replace(
            self,
            model=self.model.permute_states(permutation),
            posterior=self.posterior.permute_states(permutation),
        )


AlignableFacadeT = typing.TypeVar(
    'AlignableFacadeT', bound='AlignableFacade[typing.Any]'
)
AlignablePosterior_co = typing.TypeVar(
    'AlignablePosterior_co', bound=Alignable, covariant=True
)


class AlignableInferred(
    Inferred[AlignableFacadeT, AlignablePosterior_co],
    typing.Generic[AlignableFacadeT, AlignablePosterior_co],
):
    def align(self, alignment: Affine) -> typing.Self:
        posterior_alignment = alignment
        if self.posterior.batch_shape:
            posterior_alignment = alignment.broadcast(
                self.posterior.batch_shape,
                axis=0,
            )
        return dataclasses.replace(
            self,
            model=self.model.align(alignment),
            posterior=self.posterior.align(posterior_alignment),
        )


class Fit(typing.NamedTuple, typing.Generic[Inferred_co]):
    inferred: Inferred_co
    objective_trace: jax.Array


class _FacadeOperations(typing.Protocol[Inferred_co]):
    """Internal typing boundary for heterogeneous facade keyword arguments."""

    def infer(self, data: DatasetCollection, **kwargs: typing.Any) -> Inferred_co: ...

    def fit(
        self, data: DatasetCollection, **kwargs: typing.Any
    ) -> Fit[Inferred_co]: ...


# ---------------------------------------------------------------------
# Typed collections of inferred and fitted results
# ---------------------------------------------------------------------


FacadeCollectionT = typing.TypeVar(
    'FacadeCollectionT',
    bound='FacadeCollection[typing.Any, typing.Any, typing.Any]',
)

PosteriorT = typing.TypeVar(
    'PosteriorT',
    bound=Batchable,
)


@dataclasses.dataclass(frozen=True, eq=False)
class InferredCollection(typing.Generic[FacadeCollectionT, PosteriorT]):
    """Corresponding collections of models and inferred posteriors."""

    models: FacadeCollectionT
    posteriors: tuple[PosteriorT, ...]
    data: DatasetCollection

    def __post_init__(self) -> None:
        if not (
            len(self.models.models) == len(self.posteriors) == len(self.data.datasets)
        ):
            raise ValueError(
                'Models, posteriors, and datasets must have the same number of sessions'
            )

    def get(self, session: int | None = None) -> Inferred[typing.Any, PosteriorT]:
        """Return one session as the family's single-session result.

        If `session` is omitted, require a singleton collection.
        """
        model = self.models.get(session)
        index = 0 if session is None else session
        result_cls = self.models._inferred_cls(model)

        return result_cls(
            model=model,
            posterior=self.posteriors[index],
            data=DatasetCollection((self.data.datasets[index],)),
        )


# ---------------------------------------------------------------------
# Capability-based inferred collection mixins
# ---------------------------------------------------------------------


PermutableCollectionPosteriorT = typing.TypeVar(
    'PermutableCollectionPosteriorT',
    bound=Permutable,
)


class PermutableInferredCollection(
    InferredCollection[FacadeCollectionT, PermutableCollectionPosteriorT],
    typing.Generic[FacadeCollectionT, PermutableCollectionPosteriorT],
):
    """Collection supporting coordinated state relabeling."""

    def permute_states(
        self,
        permutations: jax.Array,
    ) -> typing.Self:
        """Apply one permutation to each model and its posterior."""

        if permutations.ndim != 2:
            raise ValueError('Expected an array of session permutations')

        if permutations.shape[0] != len(self.posteriors):
            raise ValueError('Expected one permutation per session')

        models = dataclasses.replace(
            self.models,
            models=tuple(
                typing.cast(
                    PermutableFacade[typing.Any],
                    model,
                ).permute_states(permutation)
                for model, permutation in zip(
                    self.models.models,
                    permutations,
                    strict=True,
                )
            ),
        )

        posteriors = tuple(
            posterior.permute_states(permutation)
            for posterior, permutation in zip(
                self.posteriors,
                permutations,
                strict=True,
            )
        )

        return dataclasses.replace(
            self,
            models=models,
            posteriors=posteriors,
        )


AlignableCollectionPosteriorT = typing.TypeVar(
    'AlignableCollectionPosteriorT',
    bound=Alignable,
)


class AlignableInferredCollection(
    InferredCollection[FacadeCollectionT, AlignableCollectionPosteriorT],
    typing.Generic[FacadeCollectionT, AlignableCollectionPosteriorT],
):
    """Collection supporting coordinated affine transformations."""

    def align(
        self,
        alignments: tuple[Affine, ...],
    ) -> typing.Self:
        """Apply one affine transformation to each model and posterior."""

        if len(alignments) != len(self.posteriors):
            raise ValueError('Expected one alignment per session')

        models = dataclasses.replace(
            self.models,
            models=tuple(
                typing.cast(
                    AlignableFacade[typing.Any],
                    model,
                ).align(alignment)
                for model, alignment in zip(
                    self.models.models,
                    alignments,
                    strict=True,
                )
            ),
        )

        posteriors = tuple(
            posterior.align(
                alignment.broadcast(
                    posterior.batch_shape,
                    axis=0,
                )
                if posterior.batch_shape
                else alignment
            )
            for posterior, alignment in zip(
                self.posteriors,
                alignments,
                strict=True,
            )
        )

        return dataclasses.replace(
            self,
            models=models,
            posteriors=posteriors,
        )


InferredCollectionT = typing.TypeVar(
    'InferredCollectionT',
    bound='InferredCollection[typing.Any, typing.Any]',
)


@dataclasses.dataclass(frozen=True, eq=False)
class FitCollection(typing.Generic[InferredCollectionT]):
    """Fitting results with the family's inferred-collection type."""

    inferred: InferredCollectionT
    objective_traces: jax.Array

    def get(self, session: int | None = None) -> Fit[typing.Any]:
        """Return one session's fit; omit `session` for a singleton collection."""
        inferred = self.inferred.get(session)
        index = 0 if session is None else session

        return Fit(
            inferred=inferred,
            objective_trace=self.objective_traces[index],
        )


# ---------------------------------------------------------------------
# Session orchestration
# ---------------------------------------------------------------------


def _with_progress(
    results: typing.Iterable[ResultT],
    *,
    total: int,
    progress: Progress,
) -> typing.Iterable[ResultT]:
    if progress is False:
        return results

    from tqdm.auto import tqdm

    return tqdm(
        results,
        total=total,
        desc=progress if isinstance(progress, str) else 'sessions',
    )


@dataclasses.dataclass(frozen=True, eq=False)
class FacadeCollection(typing.Generic[FacadeT, InferredT, InferredCollectionT]):
    """An ordered collection of independently initialized facades."""

    models: tuple[FacadeT, ...]

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError('FacadeCollection requires at least one model')
        if any(type(model) is not type(self.models[0]) for model in self.models):
            raise TypeError('FacadeCollection requires one concrete facade type')

    @classmethod
    def from_datasets(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        init_fn: typing.Callable[[jax.Array, Dataset], FacadeT],
    ) -> typing.Self:
        """Initialize one facade per session with independent random keys."""
        keys = jax.random.split(key, len(data.datasets))
        models = tuple(
            init_fn(session_key, dataset)
            for session_key, dataset in zip(keys, data.datasets, strict=True)
        )
        return cls(models=models)

    @classmethod
    def _inferred_collection_cls(cls) -> type[InferredCollectionT]:
        """Select the family's inferred collection; override once per family."""
        raise NotImplementedError(
            'Family collections must define _inferred_collection_cls()'
        )

    @classmethod
    def _inferred_cls(cls, model: FacadeT) -> type[InferredT]:
        """Select the single-session result type; override once per family."""
        raise NotImplementedError('Family collections must define _inferred_cls()')

    def _wrap_inferred(
        self,
        data: DatasetCollection,
        results: tuple[InferredT, ...],
    ) -> InferredCollectionT:
        models = dataclasses.replace(
            self,
            models=tuple(typing.cast(FacadeT, result.model) for result in results),
        )

        posteriors = tuple(result.posterior for result in results)

        return self._inferred_collection_cls()(
            models=models,
            posteriors=posteriors,
            data=data,
        )

    def get(self, session: int | None = None) -> FacadeT:
        if session is None:
            if len(self.models) != 1:
                raise ValueError('Session index required for multiple models')
            return self.models[0]
        return self.models[session]

    def map_models(self, function: typing.Callable[[FacadeT], FacadeT]) -> typing.Self:
        """Transform independent facades and retain the collection subtype."""
        return dataclasses.replace(
            self,
            models=tuple(function(model) for model in self.models),
        )

    def map(
        self,
        data: DatasetCollection,
        function: typing.Callable[[FacadeT, DatasetCollection], ResultT],
        *,
        map_progress: Progress = False,
        max_workers: int = 1,
    ) -> tuple[ResultT, ...]:
        """Map a session operation, returning results in session order.

        Progress measures host completion, not JAX device completion.
        """
        if max_workers < 1:
            raise ValueError('max_workers must be positive')

        pairs = tuple(zip(self.models, data.datasets, strict=True))

        def apply(pair: tuple[FacadeT, Dataset]) -> ResultT:
            model, dataset = pair
            return function(model, DatasetCollection((dataset,)))

        if max_workers == 1:
            return tuple(
                _with_progress(
                    (apply(pair) for pair in pairs),
                    total=len(pairs),
                    progress=map_progress,
                )
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(apply, pair): i for i, pair in enumerate(pairs)}
            completed: dict[int, ResultT] = {}
            for future in _with_progress(
                concurrent.futures.as_completed(futures),
                total=len(pairs),
                progress=map_progress,
            ):
                completed[futures[future]] = future.result()

        return tuple(completed[i] for i in range(len(pairs)))

    def infer(
        self,
        data: DatasetCollection,
        *,
        max_workers: int = 1,
        **kwargs: typing.Any,
    ) -> InferredCollectionT:
        """Infer independently, preserving the family collection type."""

        def single(model: FacadeT, subset: DatasetCollection) -> InferredT:
            typed = typing.cast(_FacadeOperations[InferredT], model)
            return typed.infer(subset, **kwargs)

        results = self.map(
            data,
            single,
            map_progress=len(self.models) > 1,
            max_workers=max_workers,
        )

        return self._wrap_inferred(data, results)

    def fit(
        self,
        data: DatasetCollection,
        *,
        max_workers: int = 1,
        **kwargs: typing.Any,
    ) -> FitCollection[InferredCollectionT]:
        """Fit independently, retaining fitted facades and their posteriors."""
        fit_kwargs = {'progress': False, **kwargs}

        def single(model: FacadeT, subset: DatasetCollection) -> Fit[InferredT]:
            typed = typing.cast(_FacadeOperations[InferredT], model)
            return typed.fit(subset, **fit_kwargs)

        fits = self.map(
            data,
            single,
            map_progress=len(self.models) > 1,
            max_workers=max_workers,
        )

        return FitCollection(
            inferred=self._wrap_inferred(data, tuple(fit.inferred for fit in fits)),
            objective_traces=jnp.stack(tuple(fit.objective_trace for fit in fits)),
        )
