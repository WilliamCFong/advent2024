import fileinput
from os import PathLike

import deal
import numpy as np
from numpy._typing import NDArray


@deal.post(lambda arr: arr.shape[1] == 2)
def interpret_payload(*filepaths: PathLike) -> NDArray:
    """
    Interpret Day 1 Location IDs in the pattern of
    ``^\d+\s+\d+$``.  # noqa
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
