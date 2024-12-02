import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from advent.day2 import count_safe_rows, determine_safe_rows


def reactor_levels():
    return np_st.arrays(
        np.int32,
        shape=st.tuples(
            st.integers(min_value=1, max_value=1000),
            st.just(5),
        ),
        unique=True,
    )


def all_increasing_levels():
    return reactor_levels().map(lambda matrix: np.sort(matrix, axis=-1))


def all_decreasing_levels():
    return reactor_levels().map(
        lambda matrix: np.sort(matrix, axis=-1)[:, ::-1]
    )


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large]
)
@given(st.one_of(all_increasing_levels(), all_decreasing_levels()))
def test_determine_safe_rows(matrix):
    safe_row_mask = determine_safe_rows(matrix)
    for levels, check in zip(matrix, safe_row_mask):
        diff = np.diff(levels)
        consistent_delta = np.logical_or(np.all(diff > 0), np.all(diff < 0))
        within_tolerance = np.all(np.abs(diff) < 4)
        ref = consistent_delta & within_tolerance
        assert ref == check


def test_advent_reference_part_1():
    ref = np.array(
        [
            [7, 6, 4, 2, 1],
            [1, 2, 7, 8, 9],
            [9, 7, 6, 2, 1],
            [1, 3, 2, 4, 5],
            [8, 6, 4, 4, 1],
            [1, 3, 6, 7, 9],
        ]
    )
    assert count_safe_rows(ref) == 2
