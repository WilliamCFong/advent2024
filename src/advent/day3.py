import pathlib
import re
from typing import Callable, Generator

MUL_PATTERN = r"mul\((?P<lhs>\d{1,3}),(?P<rhs>\d{1,3})\)"
PART2_PATTERN = r"mul\((?P<lhs>\d{1,3}),(?P<rhs>\d{1,3})\)|do\(\)|don't\(\)"


def interpret_payload(path: pathlib.Path) -> str:
    with open(path, "rt") as fin:
        return fin.read()


def return_all_mul(text: str) -> list[tuple[int, int]]:
    return [(int(lhs), int(rhs)) for lhs, rhs in re.findall(MUL_PATTERN, text)]


def yield_valid_mul(text: str) -> Generator[tuple[int, int], None, None]:
    enabled = True
    for match in re.finditer(PART2_PATTERN, text):
        token = match.string[match.start() : match.end()]  # noqa
        if token == "don't()":
            enabled = False
        elif token == "do()":
            enabled = True
        elif enabled and match.groupdict()["lhs"]:
            data = match.groupdict()
            yield int(data["lhs"]), int(data["rhs"])


def find_product_sum(text: str, method: Callable) -> int:
    sum_ = 0
    for lhs, rhs in method(text):
        sum_ += lhs * rhs
    return sum_
