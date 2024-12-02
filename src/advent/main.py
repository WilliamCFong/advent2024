import pathlib

import click

from advent import day1


@click.group()
def advent():
    pass


@advent.command()
@click.argument("input-files", type=pathlib.Path, nargs=-1)
@click.option("--part-1/--part-2", default=True)
def day_1(input_files: list[pathlib.Path], part_1: bool):
    for file in input_files:
        array = day1.interpret_payload(file)

        if part_1:
            distance = day1.measure_distance(array)
            print(f"{file.name} distance: {distance}")
        else:
            similarity = day1.calculate_simularity(array.T[0], array.T[1])
            print(f"{file.name} similarity: {similarity}")


if __name__ == "__main__":
    advent()
