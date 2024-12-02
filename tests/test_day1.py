import pathlib

import numpy as np
from hypothesis import strategies as st

from advent import day1


def location_ids():
    return st.integers(min_value=10**5, max_value=10 * 6 - 1)


def day1_payloads(**list_kwargs):
    return st.lists(st.tuples(location_ids(), location_ids()), **list_kwargs)


def write_payload(output: pathlib.Path, payload: list[tuple[int, int]]):
    with open(output, "wt") as fout:
        for lhs, rhs in payload:
            fout.write(f"{lhs}    {rhs}\n")


def test_advent_reference_part_1():
    # Test the example given by https://adventofcode.com/2024/day/1
    reference = np.array(
        [
            [3, 4],
            [4, 3],
            [2, 5],
            [1, 3],
            [3, 9],
            [3, 3],
        ]
    ).T
    distance = day1.measure_distance(reference[0], reference[1])
    assert distance == 11


def test_advent_reference_part_2():
    reference = np.array(
        [
            [3, 4],
            [4, 3],
            [2, 5],
            [1, 3],
            [3, 9],
            [3, 3],
        ]
    ).T
    similarity = day1.calculate_simularity(reference[0], reference[1])
    assert similarity == 31
