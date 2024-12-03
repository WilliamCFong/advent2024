import pathlib

import deal
import numpy as np
from numpy import typing as npt


def interpret_payload(path: pathlib.Path) -> list[npt.NDArray]:
    all_reports = []
    with open(path, "rt") as fin:
        for raw_report in fin:
            report = np.array(
                list(int(token) for token in raw_report.strip().split())
            )
            all_reports.append(report)

    return all_reports


@deal.pre(
    lambda _: len(_.reports) > 1,
    message="Array must be longer than 1 element to calculate difference.",
)
def search_for_safe_path(
    reports: npt.NDArray, tolerance: int = 0
) -> npt.NDArray:
    """
    Kind of a hacky solution for both part 1 and part 2 of day 2. Namely
    speaking I am treating this as a search for a safe path. This is done by
    treating the solution space as a NxN array of possible differences of
    "reports". We then treat the possible paths as a "down-right" diagonal,
    straight down, or straight right.

    Down-right diagonal moves are possible when the difference is within
    the [0, 3] difference range and the sign of the difference is the same
    as the previous, if applicable.

    Straight down movements are possible if the current "tolerance" is not
    negative (i.e "recovery" in part 2). When this occurs. The next step must
    be a straight right motion.
    """
    solution_path = []
    i = 1
    j = 0
    sign = 0
    skipped = False
    while i < len(reports) and tolerance >= 0:
        diff = reports[i] - reports[j]
        if abs(diff) == 0 or abs(diff) > 3:
            i += 1
            tolerance -= 1
            skipped = True
            continue
        if (sign != 0) and (
            (diff > 0 and sign < 0) or (diff < 0 and sign > 0)
        ):
            i += 1
            tolerance -= 1
            skipped = True
            continue
        sign = diff / abs(diff)
        solution_path.append(diff)
        j += 1
        if not skipped:
            i += 1
        skipped = False
    return np.array(solution_path)


@deal.ensure(
    lambda _: len(_.matrix) == _.result.shape[0],
    message="Returned matrix must be same dimension as first"
    " dimension of input.",
)
def determine_safe_rows(
    matrix: list[npt.NDArray], tolerance: int = 0
) -> npt.NDArray:
    bucket = []
    for row in matrix:
        solution_path = search_for_safe_path(row, tolerance=tolerance)
        bucket.append(len(solution_path) == len(row) - 1)

    return np.array(bucket)


def count_safe_rows(payload: list[npt.NDArray], tolerance: int = 0) -> int:
    safe_row_mask = determine_safe_rows(payload, tolerance=tolerance)
    return np.count_nonzero(safe_row_mask)
