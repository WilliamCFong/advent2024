import fileinput
from os import PathLike

import deal
import numpy as np
from numpy._typing import NDArray


@deal.post(lambda arr: arr.shape[1] == 2)
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
    return array


@deal.pre(lambda arr: arr.shape[1] == 2, message="Array shape must be (Nx2)")
def measure_distance(array: NDArray) -> int:
    by_list = array.T
    lhs_sorted_idx = np.argsort(by_list[0])
    rhs_sorted_idx = np.argsort(by_list[1])

    diff = np.sum(
        np.abs(by_list[0][lhs_sorted_idx] - by_list[1][rhs_sorted_idx])
    )
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
