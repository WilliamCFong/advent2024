import pathlib
from typing import Optional

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


def valid_report(diff: int, prev_diff: Optional[int]):
    if abs(diff) > 3 or diff == 0:
        return False

    if (prev_diff is not None) and (
        (diff < 0 < prev_diff) or (prev_diff < 0 < diff)
    ):
        # Difference must have same sign as any past
        return False

    return True


@deal.pre(
    lambda _: len(_.reports) > 1,
    message="Array must be longer than 1 element to calculate difference.",
)
def reactor_report_tolerable(reports: npt.NDArray, tolerance: int = 0) -> bool:
    """
    Kind of a hacky solution for both part 1 and part 2 of day 2. Namely
    speaking I am treating this as a search for a safe path. This is done by
    treating the solution space as a NxN array of possible differences of
    "reports". We then treat the possible paths as a "down-right" diagonal
    moves. We have a "tolerance" counter where as long as it is above 0
    the possible moves may include omitting the considered rows.

    The search ends when the "lhs" indexer reaches the amount of masked
    reports in which case, the search succeeded and the "reactor" is in a
    safe state. However if the search path queue is at any point empty, there
    are no possible solutions.
    """
    search_queue = [(1, 0, np.full_like(reports, True, dtype=np.bool_), None)]

    while len(search_queue) > 0:
        lhs, rhs, mask, prev_diff = search_queue.pop()

        # Exit possibility
        if lhs >= len(reports[mask]):
            # We have reached a possible exit path, return
            return True

        diff = reports[mask][lhs] - reports[mask][rhs]

        if valid_report(diff, prev_diff):
            # Continuing the path is a possible search path
            search_queue.append((lhs + 1, rhs + 1, mask, diff))
        if np.count_nonzero(~mask) < tolerance:
            # Check possible alternatives
            new_mask = np.copy(mask)
            new_mask[np.where(mask)[0][lhs]] = False
            search_queue.append((1, 0, new_mask, None))
            new_mask = np.copy(mask)
            new_mask[np.where(mask)[0][rhs]] = False
            search_queue.append((1, 0, new_mask, None))
    # We have exhausted all possible paths
    return False


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
        tolerable = reactor_report_tolerable(row, tolerance=tolerance)
        bucket.append(tolerable)

    return np.array(bucket)


def count_safe_rows(payload: list[npt.NDArray], tolerance: int = 0) -> int:
    safe_row_mask = determine_safe_rows(payload, tolerance=tolerance)
    return np.count_nonzero(safe_row_mask)
