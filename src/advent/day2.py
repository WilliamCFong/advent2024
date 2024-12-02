import pathlib

import deal
import numpy as np
from numpy import typing as npt


def interpret_payload(path: pathlib.Path) -> list[npt.NDArray]:
    all_reports = []
    with open(path, "rt") as fin:
        for raw_report in fin:
            report = np.array(
                int(token) for token in raw_report.strip().split()
            )
            all_reports.append(report)

    return all_reports


@deal.ensure(
    lambda _: len(_.matrix) == _.result.shape[0],
    message="Returned matrix must be same dimension as first"
    " dimension of input.",
)
def determine_safe_rows(matrix: list[npt.NDArray]) -> npt.NDArray:
    bucket = []
    for row in matrix:
        level_delta = np.diff(row)
        negative_delta = np.all(level_delta < 0)
        positive_delta = np.all(level_delta > 0)
        homogenous_delta = negative_delta | positive_delta

        delta_within_tolerance = np.logical_and(
            np.all(np.abs(level_delta) > 0),
            np.all(np.abs(level_delta) <= 3),
        )
        bucket.append(homogenous_delta & delta_within_tolerance)

    return np.array(bucket)


def count_safe_rows(payload: list[npt.NDArray]) -> int:
    safe_row_mask = determine_safe_rows(payload)
    return np.count_nonzero(safe_row_mask)
