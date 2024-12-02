import fileinput
from os import PathLike

import deal
import numpy as np
from numpy._typing import NDArray


@deal.post(
    lambda arr: arr.shape[0] == 2,
    message="Returned array must be of two lists of equal size",
)
def interpret_payload(*filepaths: PathLike) -> NDArray:
    """
    Interpret Day 1 Location IDs in the pattern of
    (int)   (int)
    """
    array = np.array(
        [
            tuple(map(int, line.strip().split()))
            for line in fileinput.input(*filepaths)
        ]
    )
    return array.T


@deal.pre(
    lambda lhs, rhs: lhs.shape == rhs.shape,
    message="Arrays must be of the same dimensionality.",
)
def measure_distance(lhs: NDArray, rhs: NDArray) -> int:
    """
    Calculate 'distance' between the provided arrays. Distance in this case
    is measured by the absolute difference between each list sorted.
    """
    lhs_sorted_idx = np.argsort(lhs)
    rhs_sorted_idx = np.argsort(rhs)

    diff = np.sum(np.abs(lhs[lhs_sorted_idx] - rhs[rhs_sorted_idx]))
    return diff


def _load_counter(array: NDArray) -> dict[int, int]:
    """
    Processess the given array, loading in each unique element into a
    dictionary lookup with the number of times that element occurs in the
    provided array.
    """
    value, occurances = np.unique(array, return_counts=True)
    return dict(zip(value, occurances))


@deal.pre(
    lambda lhs, rhs: lhs.shape == rhs.shape,
    message="Arrays must be of the same dimensionality.",
)
def calculate_simularity(lhs: NDArray, rhs: NDArray) -> int:
    lhs_counter = _load_counter(lhs)
    rhs_counter = _load_counter(rhs)

    # 'similary' requires elements to be members of both lists
    common_elements = set(lhs_counter.keys()) & set(rhs_counter.keys())
    similarity = 0
    for element in common_elements:
        similarity += element * lhs_counter[element] * rhs_counter[element]
    return similarity
