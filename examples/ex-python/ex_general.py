#!@PYTHON_EXECUTABLE@

import sys
import json
import argparse
import timemory
from timemory.profiler import profile
from timemory.bundle import auto_timer, auto_tuple, marker


def get_config(items=["wall_clock", "cpu_clock"]):
    return [getattr(timemory.component, x) for x in items]


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


@profile(["wall_clock", "peak_rss"])
def run_profile(n):
    """ Run full profiler """
    return fib(int(n))


@auto_timer()
def run_auto_timer(n):
    """ Decorator and context manager for high-level pre-defined collection """
    fib(n)
    with auto_timer(key="auto_timer_ctx_manager"):
        fib(n)


@marker(["wall_clock", "peak_rss"])
def run_marker(n):
    """ Decorator and context manager for high-level custom collection """
    fib(n)
    with auto_tuple(get_config(), key="auto_tuple_ctx_manager"):
        fib(n)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--nfib", type=int, default=10, help="Fibonacci depth"
    )

    # demonstrate how to add command-line support
    # The subparser stuff is not strictly necessary: if a
    # subparser is not provided, the default behavior is
    # to do this internally. If you want to disable the
    # subparser behavior, pass 'subparser=None' or
    # subparser=False
    subparser = parser.add_subparsers()
    timemory.settings.add_arguments(
        parser,
        subparser=subparser.add_parser(
            "timemory-config",
            parents=[parser],
            conflict_handler="resolve",
            description="Configure settings for timemory",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        ),
    )

    args = parser.parse_args()
    n = args.nfib

    run_profile(n)
    run_auto_timer(n)
    run_marker(n)

    # Get the results as dictionary
    data = timemory.get()

    print("\n{}".format(json.dumps(data, indent=4, sort_keys=True)))
    # Generate output
    timemory.finalize()
