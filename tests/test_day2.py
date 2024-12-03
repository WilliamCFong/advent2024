import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from advent import day2
from advent.day2 import count_safe_rows


def reactor_levels():
    return st.lists(
        st.lists(st.integers(), min_size=2, max_size=10), min_size=1
    )


def ascending_reactor_levels():
    return st.builds(
        np.arange,
        start=st.just(0),
        stop=st.integers(min_value=2, max_value=1000),
    ).map(list)


def descending_reactor_levels():
    return ascending_reactor_levels().map(lambda arr: arr[::-1])


def safe_reactor_levels():
    return st.one_of(ascending_reactor_levels(), descending_reactor_levels())


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    deadline=None,
)
@given(safe_reactor_levels(), st.data())
def test_tolerance_w_zero_diff(levels: list[int], data):
    idx = data.draw(st.integers(min_value=0, max_value=len(levels) - 1))
    levels.insert(idx, levels[idx])
    assert not day2.reactor_report_tolerable(np.array(levels))
    assert day2.reactor_report_tolerable(np.array(levels), tolerance=1)


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    deadline=None,
)
@given(safe_reactor_levels(), st.data())
def test_tolerance_w_opposite_sign(levels: list[int], data):
    idx = data.draw(st.integers(min_value=0, max_value=len(levels) - 1))
    levels.insert(idx, levels[idx] * -1)
    assert not day2.reactor_report_tolerable(np.array(levels))
    assert day2.reactor_report_tolerable(np.array(levels), tolerance=1)


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
