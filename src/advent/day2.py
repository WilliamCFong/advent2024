import pathlib

import deal
import numpy as np
from numpy import typing as npt


def interpret_payload(path: pathlib.Path) -> npt.NDArray:
    return np.loadtxt(path, dtype=np.int32)


@deal.pre(
    lambda matrix: len(matrix.shape) == 2, message="Input matrix must be 2D."
)
@deal.ensure(
    lambda _: _.matrix.shape[0] == _.result.shape[0],
    message="Returned matrix must be same dimension as first"
    " dimension of input.",
)
def determine_safe_rows(matrix: npt.NDArray) -> npt.NDArray:
    level_delta = np.diff(matrix)
    negative_delta = np.all(level_delta < 0, axis=-1)
    positive_delta = np.all(level_delta > 0, axis=-1)
    homogenous_delta = negative_delta | positive_delta

    delta_within_tolerance = np.logical_and(
        np.all(np.abs(level_delta) > 0, axis=-1),
        np.all(np.abs(level_delta) <= 3, axis=-1),
    )

    return homogenous_delta & delta_within_tolerance


def count_safe_rows(payload: npt.NDArray[np.bool]) -> int:
    safe_row_mask = determine_safe_rows(payload)
    return np.count_nonzero(safe_row_mask)
