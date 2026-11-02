"""Compact host-side summaries for notebook inspection.

A summary is a titled `Record` (one object, one value per field) or `Table`
(one row of fields per item). `text` renders summaries for `__repr__` with
property names; `markdown` renders them for notebooks with math symbols.

Summaries report structure, never array contents. Value reductions run only on
concrete arrays, so summaries stay valid for PyTrees traced under `jit`.
"""

import typing

import jax
import jax.core
import jax.numpy as jnp

MAX_ROWS = 20


class Field(typing.NamedTuple):
    """A summary value named as its property; `symbol` is its math name."""

    name: str
    value: object
    symbol: str | None = None


class Record(typing.NamedTuple):
    """Fields of one object."""

    title: str
    fields: tuple[Field, ...]


class Table(typing.NamedTuple):
    """Fields of each item in a collection, one row per item."""

    title: str
    rows: tuple[tuple[Field, ...], ...]


Summary = Record | Table


def text(summary: Summary) -> str:
    """Plain-text block labelled with property names."""
    header, rows = _grid(summary, lambda field: field.name)
    if isinstance(summary, Record):
        width = max(map(len, header))
        lines = [f'  {name:<{width}}  {value}' for name, value in zip(header, rows[0])]
    else:
        cells = [header, *rows]
        widths = [
            max(len(row[column]) for row in cells) for column in range(len(header))
        ]
        lines = [
            '  '
            + '  '.join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
            for row in cells
        ]
    if more := _more(summary):
        lines.append(f'  {more}')
    return '\n'.join((summary.title, *lines))


def markdown(summary: Summary) -> str:
    """Markdown table labelled with math symbols where fields define one."""

    def line(cells: typing.Iterable[str]) -> str:
        return '| ' + ' | '.join(cell.replace('|', r'\|') for cell in cells) + ' |'

    header, rows = _grid(
        summary,
        lambda field: field.name if field.symbol is None else f'${field.symbol}$',
    )
    lines = [f'**{summary.title}**', '', line(header), line('---' for _ in header)]
    lines += map(line, rows)
    if more := _more(summary):
        lines += ['', more]
    return '\n'.join(lines)


def _grid(
    summary: Summary,
    label: typing.Callable[[Field], str],
) -> tuple[list[str], list[list[str]]]:
    """Header and cell rows; tables lead with a `#` index column."""
    if isinstance(summary, Record):
        return (
            [label(field) for field in summary.fields],
            [[str(field.value) for field in summary.fields]],
        )
    return (
        ['#', *(label(field) for field in summary.rows[0])],
        [
            [str(index), *(str(field.value) for field in row)]
            for index, row in enumerate(summary.rows[:MAX_ROWS])
        ],
    )


def _more(summary: Summary) -> str:
    if isinstance(summary, Table) and len(summary.rows) > MAX_ROWS:
        return f'... ({len(summary.rows) - MAX_ROWS} more)'
    return ''


def concrete(*values: jax.Array) -> bool:
    """Whether no value is a tracer."""
    return not any(isinstance(value, jax.core.Tracer) for value in values)


def lengths_str(lengths: jax.Array) -> str:
    """Shared valid length, or `min..max` when lengths differ."""
    if not concrete(lengths):
        return '?'
    low, high = int(jnp.min(lengths)), int(jnp.max(lengths))
    return str(low) if low == high else f'{low}..{high}'


def percent_str(count: int, total: int) -> str:
    """Percentage of `total`, exact when complete."""
    if count == total:
        return '100%'
    return f'{100 * count / total:.1f}%'


def trace_fields(trace: jax.Array) -> tuple[Field, ...]:
    """Iterations and initial/final values of an objective trace `(1 + iters,)`."""
    initial = final = '?'
    if concrete(trace):
        initial, final = f'{float(trace[0]):.4g}', f'{float(trace[-1]):.4g}'
    return (
        Field('num_iters', trace.shape[-1] - 1),
        Field('initial_objective', initial),
        Field('final_objective', final),
    )
