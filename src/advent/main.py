import pathlib

import click
import deal

from advent import day1
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
            with profile(pass_through=trace):
                distance = day1.measure_distance(lhs, rhs)
            print(f"{file.name} distance: {distance}")
        else:
            with profile(pass_through=trace):
                similarity = day1.calculate_simularity(lhs, rhs)
            print(f"{file.name} similarity: {similarity}")


if __name__ == "__main__":
    advent()
