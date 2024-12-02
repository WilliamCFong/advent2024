import pathlib

import click

from advent import day1


@click.group()
def advent():
    pass


@advent.command()
@click.argument("input-files", type=pathlib.Path, nargs=-1)
def day_1(input_files: list[pathlib.Path]):
    for file in input_files:
        array = day1.interpret_payload(file)
        distance = day1.measure_distance(array)

        print(f"{file.name} distance: {distance}")


if __name__ == "__main__":
    advent()
