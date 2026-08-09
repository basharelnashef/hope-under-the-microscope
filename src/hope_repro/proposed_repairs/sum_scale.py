"""Independent sum-preserving parent-scale repair for HOPE v1.

This module is not part of the printed paper method. It reuses the neutral
physical reconstruction in ``paper_literal`` while selecting the independent
scale ``s = b``.
"""

from __future__ import annotations

from typing import Any

from hope_repro.paper_literal.core import HopeEncoder


def sum_preserving_scale(layer: Any, first: int, second: int) -> float:
    """Return the independently proposed unnormalized scale ``s = b``."""
    return float(layer.b_mat[first, second])


def make_sum_scale_parent(
    layer: Any,
    first: int,
    second: int,
) -> dict[str, Any] | None:
    """Build a parent with the independent sum-preserving scale policy."""
    return layer._realize_parent_at_scale(
        first,
        second,
        sum_preserving_scale(layer, first, second),
    )


class SumScaleHopeEncoder(HopeEncoder):
    """Adapted encoder that opts into the independent ``s = b`` repair."""

    def _make_parent(self, layer: Any, first: int, second: int):
        return make_sum_scale_parent(layer, first, second)
