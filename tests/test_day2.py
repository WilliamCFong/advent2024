import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from advent.day2 import count_safe_rows, determine_safe_rows


def reactor_levels():
    return st.lists(
        st.lists(st.integers(), min_size=2, max_size=10), min_size=1
    )


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
)
@given(reactor_levels())
def test_determine_safe_rows(matrix):
    safe_row_mask = determine_safe_rows(matrix)
    for levels, check in zip(matrix, safe_row_mask):
        diff = np.diff(levels)
        consistent_delta = np.logical_or(np.all(diff > 0), np.all(diff < 0))
        within_tolerance = np.all(np.abs(diff) < 4)
        ref = consistent_delta & within_tolerance
        assert ref == check


def test_advent_reference_part_1():
    ref = [
        np.array([7, 6, 4, 2, 1]),
        np.array([1, 2, 7, 8, 9]),
        np.array([9, 7, 6, 2, 1]),
        np.array([1, 3, 2, 4, 5]),
        np.array([8, 6, 4, 4, 1]),
        np.array([1, 3, 6, 7, 9]),
    ]
    assert count_safe_rows(ref) == 2


def test_advent_reference_part_2():
    ref = [
        np.array([7, 6, 4, 2, 1]),
        np.array([1, 2, 7, 8, 9]),
        np.array([9, 7, 6, 2, 1]),
        np.array([1, 3, 2, 4, 5]),
        np.array([8, 6, 4, 4, 1]),
        np.array([1, 3, 6, 7, 9]),
    ]
    assert count_safe_rows(ref, tolerance=1) == 4
