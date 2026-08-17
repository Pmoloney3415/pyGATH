"""Static source-field indices and compact spatial-field selections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

from pyGATH.raytracing.raystatelayout import RaySheetLayout


@dataclass(frozen=True)
class FieldLayout(RaySheetLayout):
    """Full layout available when constructing spatial laser fields.

    This initially matches :class:`~pyGATH.raytracing.RaySheetLayout`.
    Additional derived field quantities should be appended here when their
    calculations are introduced; unnamed placeholder indices are deliberately
    avoided.
    """


FIELD_LAYOUT = FieldLayout()


def _layout_indices(value, nsource: int) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError("boolean values are not field selectors")
    if isinstance(value, Integral):
        index = int(value)
        if index < 0:
            index += nsource
        if not 0 <= index < nsource:
            raise IndexError(f"field index {value} is outside a width of {nsource}")
        return (index,)
    if isinstance(value, slice):
        return tuple(range(*value.indices(nsource)))
    raise TypeError("field selectors must be names, integer indices, or slices")


def _selector_indices(selector, nsource: int) -> tuple[int, ...]:
    if isinstance(selector, str):
        if selector == "n_attributes" or not hasattr(FIELD_LAYOUT, selector):
            raise ValueError(f"unknown field name {selector!r}")
        return _layout_indices(getattr(FIELD_LAYOUT, selector), nsource)
    return _layout_indices(selector, nsource)


@dataclass(frozen=True)
class FieldSelection:
    """Mapping from the full source layout to compact interpolated values.

    Named attributes use compact output indices. For example, selecting
    ``("ray_power", "capped_amplitude")`` makes ``selection.ray_power`` zero
    and ``selection.capped_amplitude`` one. A selected vector is represented
    by a slice when its components remain contiguous.
    """

    source_indices: tuple[int, ...]

    @property
    def n_attributes(self) -> int:
        """Number of compact field components in the selection."""
        return len(self.source_indices)

    def __getattr__(self, name: str):
        if name == "n_attributes" or not hasattr(FIELD_LAYOUT, name):
            raise AttributeError(name)
        source = _layout_indices(getattr(FIELD_LAYOUT, name), FIELD_LAYOUT.n_attributes)
        try:
            compact = tuple(self.source_indices.index(index) for index in source)
        except ValueError as error:
            raise AttributeError(f"field {name!r} is not fully selected") from error
        if len(compact) == 1:
            return compact[0]
        if compact == tuple(range(compact[0], compact[0] + len(compact))):
            return slice(compact[0], compact[-1] + 1)
        return compact


def resolve_field_selection(
    selectors: str | int | slice | Iterable[str | int | slice] | None,
    *,
    nsource: int,
) -> FieldSelection:
    """Validate field selectors and return their compact source mapping."""
    if nsource != FIELD_LAYOUT.n_attributes:
        raise ValueError(
            "sheet field width does not match FIELD_LAYOUT: "
            f"got {nsource}, expected {FIELD_LAYOUT.n_attributes}"
        )
    if selectors is None:
        return FieldSelection(tuple(range(nsource)))
    if isinstance(selectors, bool):
        raise TypeError("boolean values are not field selectors")
    if isinstance(selectors, (str, Integral, slice)):
        selector_items = (selectors,)
    else:
        try:
            selector_items = tuple(selectors)
        except TypeError as error:
            raise TypeError(
                "fields must be a name, integer, slice, or iterable of selectors"
            ) from error
    if not selector_items:
        raise ValueError("at least one field must be selected")

    indices = tuple(
        index
        for selector in selector_items
        for index in _selector_indices(selector, nsource)
    )
    if not indices:
        raise ValueError("field selection cannot be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("field selection contains duplicate source components")
    return FieldSelection(indices)


__all__ = [
    "FIELD_LAYOUT",
    "FieldLayout",
    "FieldSelection",
    "resolve_field_selection",
]
