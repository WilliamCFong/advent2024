"""
Some quick profiling scripts for curiosities and optimization.
"""

from contextlib import contextmanager


@contextmanager
def profile(pass_through=False):
    if pass_through:
        yield
        return
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield
    finally:
        profiler.disable()

    pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(
        100
    )
