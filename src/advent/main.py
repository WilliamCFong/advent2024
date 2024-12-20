import pathlib

import click
import deal

from advent import day1, day2, day3
from advent.profile import profile


@click.group()
@click.option("--no-contracts/--contracts", "contracts", default=False)
def advent(contracts: bool):
    if not contracts:
        deal.disable()


@advent.command()
@click.argument("input-files", type=pathlib.Path, nargs=-1)
@click.option("--part-1/--part-2", default=True)
@click.option("--no-trace/--trace", "trace", default=False)
def day_1(input_files: list[pathlib.Path], part_1: bool, trace: bool):
    for file in input_files:
        payload = day1.interpret_payload(file)
        lhs = payload[0]
        rhs = payload[1]

        if part_1:
            with profile(pass_through=not trace):
                distance = day1.measure_distance(lhs, rhs)
            print(f"{file.name} distance: {distance}")
        else:
            with profile(pass_through=trace):
                similarity = day1.calculate_simularity(lhs, rhs)
            print(f"{file.name} similarity: {similarity}")


@advent.command()
@click.argument("input-file", type=pathlib.Path)
@click.option("--part-1/--part-2", default=True)
@click.option("--no-trace/--trace", "trace", default=False)
def day_2(input_file: pathlib.Path, part_1: bool, trace: bool):
    payload = day2.interpret_payload(input_file)

    if part_1:
        with profile(pass_through=not trace):
            n_safe_rows = day2.count_safe_rows(payload)
    else:
        with profile(pass_through=not trace):
            n_safe_rows = day2.count_safe_rows(payload, tolerance=1)
    print(f"{input_file.name} has {n_safe_rows} safe reports")


@advent.command()
@click.argument("input-file", type=pathlib.Path)
@click.option("--part-1/--part-2", default=True)
@click.option("--no-trace/--trace", "trace", default=False)
def day_3(input_file: pathlib.Path, part_1: bool, trace: bool):
    payload = day3.interpret_payload(input_file)
    if part_1:
        with profile(pass_through=not trace):
            sum_ = day3.find_product_sum(payload, day3.return_all_mul)
    else:
        with profile(pass_through=not trace):
            sum_ = day3.find_product_sum(payload, day3.yield_valid_mul)
    print(f"{input_file.name} product sum is {sum_}")


if __name__ == "__main__":
    advent()
