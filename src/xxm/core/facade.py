"""Host-side facades, inference results, and collection orchestration."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import typing
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from xxm.core import batch, display
from xxm.core.affine import Affine
from xxm.core.batch import Batchable
from xxm.core.data import Dataset, DatasetCollection
from xxm.core.io.hdf5 import PathLike, save_model_params
from xxm.core.optim.loop import Progress

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


_SUMMARY_DIMS = (
    ('num_states', 'K'),
    ('num_motifs', 'P'),
    ('latent_dim', 'D_x'),
    ('input_dim', 'D_u'),
)


@dataclasses.dataclass(frozen=True, eq=False)
class Facade(typing.Generic[CoreT]):
    """An unbatched numerical core and typed facade reconstruction."""

    core: CoreT

    def __post_init__(self) -> None:
        if self.core.batch_shape != ():
            raise ValueError('Facade requires an unbatched core model')

    def __repr__(self) -> str:
        return display.text(self._summary())

    def _repr_markdown_(self) -> str:
        return display.markdown(self._summary())

    def _summary(self) -> display.Record:
        """Dimensions the family defines, with one `D_y` per emission session."""
        fields = [
            display.Field(name, getattr(self, name), symbol)
            for name, symbol in _SUMMARY_DIMS
            if hasattr(self, name)
        ]
        emissions = typing.cast(typing.Any, self.core).emissions.emissions
        observation_dims = tuple(emission.observation_dim for emission in emissions)
        fields.append(
            display.Field(
                'observation_dim',
                observation_dims[0] if len(observation_dims) == 1 else observation_dims,
                'D_y',
            )
        )
        return display.Record(type(self).__name__, tuple(fields))

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


def _model_summary(model: object) -> display.Record:
    """Summary of a facade held under an unbound type variable."""
    return typing.cast(Facade[typing.Any], model)._summary()


def _collection_title(collection: object, models: typing.Sequence[object]) -> str:
    return f'{type(collection).__name__} of {len(models)} {type(models[0]).__name__}'


def _inferred_fields(
    model: object,
    data: Dataset | DatasetCollection,
) -> tuple[display.Field, ...]:
    """Model fields followed by the data layout the model does not report."""
    fields = _model_summary(model).fields
    names = {field.name for field in fields}
    layout = (
        display.Field('max_steps', data.max_steps, 'T'),
        display.Field('input_dim', data.input_dim, 'D_u'),
    )
    return (*fields, *(field for field in layout if field.name not in names))


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

    def __repr__(self) -> str:
        return display.text(self._summary())

    def _repr_markdown_(self) -> str:
        return display.markdown(self._summary())

    def _summary(self) -> display.Record:
        return display.Record(
            f'{type(self).__name__} {type(self.model).__name__}',
            _inferred_fields(self.model, self.data),
        )

    def _session(self, session: int | None = None) -> tuple[Posterior_co, Dataset]:
        dataset = self.data.get(session)
        index = 0 if session is None else session
        posterior = batch.select(
            self.posterior,
            self.data.partition.get_slice(index),
        )
        return posterior, dataset

    def elbo(self) -> jax.Array:
        """Evidence lower bound of `data` under `model` at `posterior`, summed over sessions and sequences."""
        raise NotImplementedError


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

    def __repr__(self) -> str:
        return display.text(self._summary())

    def _repr_markdown_(self) -> str:
        return display.markdown(self._summary())

    def _summary(self) -> display.Record:
        return display.Record(
            f'Fit of {type(self.inferred.model).__name__}',
            display.trace_fields(self.objective_trace),
        )


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

    def __repr__(self) -> str:
        return display.text(self._summary())

    def _repr_markdown_(self) -> str:
        return display.markdown(self._summary())

    def _summary(self) -> display.Table:
        """One row per model with the layout of its session data."""
        models = self.models.models
        return display.Table(
            _collection_title(self, models),
            tuple(
                _inferred_fields(model, dataset)
                for model, dataset in zip(models, self.data.datasets, strict=True)
            ),
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

    def elbo(self) -> jax.Array:
        """Sum of the per-session ELBOs; see `Inferred.elbo`."""
        return sum(
            (self.get(i).elbo() for i in range(len(self.posteriors))),
            start=jnp.zeros(()),
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

    def __repr__(self) -> str:
        return display.text(self._summary())

    def _repr_markdown_(self) -> str:
        return display.markdown(self._summary())

    def _summary(self) -> display.Table:
        """One row per objective trace."""
        return display.Table(
            _collection_title(self, self.inferred.models.models),
            tuple(display.trace_fields(trace) for trace in self.objective_traces),
        )

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

    def __repr__(self) -> str:
        return display.text(self._summary())

    def _repr_markdown_(self) -> str:
        return display.markdown(self._summary())

    def _summary(self) -> display.Table:
        """One row per model."""
        return display.Table(
            _collection_title(self, self.models),
            tuple(_model_summary(model).fields for model in self.models),
        )

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
    def from_inferred(
        cls,
        source: InferredCollection[typing.Any, typing.Any],
        transfer_fn: typing.Callable[[int, typing.Any], FacadeT],
    ) -> typing.Self:
        """
        Transfer independent inferred source sessions one-to-one.

        `transfer_fn` receives each session index and its family-specific
        single-session `Inferred`.
        """
        return cls(
            models=tuple(
                transfer_fn(session, source.get(session))
                for session in range(len(source.posteriors))
            )
        )

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
        function: typing.Callable[[int, FacadeT, DatasetCollection], ResultT],
        *,
        map_progress: Progress = False,
        max_workers: int = 1,
    ) -> tuple[ResultT, ...]:
        """Map a session operation, returning results in session order.

        `function` receives the session index, its facade, and its
        single-session data.

        Progress measures host completion, not JAX device completion.
        """
        if max_workers < 1:
            raise ValueError('max_workers must be positive')

        pairs = tuple(zip(self.models, data.datasets, strict=True))

        def apply(session_idx: int) -> ResultT:
            model, dataset = pairs[session_idx]
            return function(session_idx, model, DatasetCollection((dataset,)))

        if max_workers == 1:
            return tuple(
                _with_progress(
                    (apply(session_idx) for session_idx in range(len(pairs))),
                    total=len(pairs),
                    progress=map_progress,
                )
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(apply, i): i for i in range(len(pairs))}
            completed: dict[int, ResultT] = {}
            for future in _with_progress(
                concurrent.futures.as_completed(futures),
                total=len(pairs),
                progress=map_progress,
            ):
                completed[futures[future]] = future.result()

        return tuple(completed[i] for i in range(len(pairs)))

    def _infer_each(
        self,
        data: DatasetCollection,
        infer: typing.Callable[[int, FacadeT, DatasetCollection], InferredT],
        *,
        max_workers: int,
    ) -> InferredCollectionT:
        """Infer each session independently; family collections type `infer`."""
        results = self.map(
            data,
            infer,
            map_progress=len(self.models) > 1,
            max_workers=max_workers,
        )

        return self._wrap_inferred(data, results)

    def _fit_each(
        self,
        data: DatasetCollection,
        fit: typing.Callable[[int, FacadeT, DatasetCollection], Fit[InferredT]],
        *,
        max_workers: int,
    ) -> FitCollection[InferredCollectionT]:
        """Fit each session independently; family collections type `fit`."""
        fits = self.map(
            data,
            fit,
            map_progress=len(self.models) > 1,
            max_workers=max_workers,
        )

        return FitCollection(
            inferred=self._wrap_inferred(data, tuple(fit.inferred for fit in fits)),
            objective_traces=jnp.stack(tuple(fit.objective_trace for fit in fits)),
        )
