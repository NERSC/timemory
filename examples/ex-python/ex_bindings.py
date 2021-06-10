#!@PYTHON_EXECUTABLE@


import numpy
import argparse

use_mpi = True
try:
    import mpi4py  # noqa: F401
    from mpi4py import MPI  # noqa: F401
except ImportError:
    use_mpi = False
    pass

import timemory  # noqa: E402
from timemory.profiler import profile  # noqa: E402
import libex_python_bindings as ex_bindings  # noqa: E402

if use_mpi:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    comm = None
    rank = 0
    size = 1

if size > 2:
    raise RuntimeError("This example only supports two MPI procs")


def run_profile(nitr=100, nsize=1000000):
    return ex_bindings.run(nitr, nsize)


def run_mpi(nitr=100, nsize=1000000):

    if size != 2:
        return

    for i in range(nitr):
        # passing MPI datatypes explicitly
        if rank == 0:
            data = numpy.arange(nsize, dtype="i")
            comm.Send([data, MPI.INT], dest=1, tag=77)
        elif rank == 1:
            data = numpy.empty(nsize, dtype="i")
            comm.Recv([data, MPI.INT], source=0, tag=77)

        # automatic MPI datatype discovery
        if rank == 0:
            data = numpy.empty(nsize, dtype=numpy.float64)
            comm.Recv(data, source=1, tag=13)
        elif rank == 1:
            data = numpy.arange(nsize, dtype=numpy.float64)
            comm.Send(data, dest=0, tag=13)


def main(args):
    # start MPI wrappers
    id = timemory.start_mpip()

    run_mpi(args.iterations)
    ans = run_profile(args.iterations, args.size)

    # stop MPI wrappers
    timemory.stop_mpip(id)
    return ans


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iterations",
        required=False,
        default=100,
        type=int,
        help="Iterations",
    )
    parser.add_argument(
        "-n",
        "--size",
        required=False,
        default=1000000,
        type=int,
        help="Array size",
    )

    args = parser.parse_args()
    timemory.enable_signal_detection()
    timemory.settings.width = 12
    timemory.settings.precision = 6
    timemory.settings.plot_output = True
    timemory.settings.dart_output = True

    with profile(
        [
            "wall_clock",
            "user_clock",
            "system_clock",
            "cpu_util",
            "peak_rss",
            "thread_cpu_clock",
            "thread_cpu_util",
        ]
    ):
        ans = main(args)
        print("Answer = {}".format(ans))

    timemory.finalize()
    print("Python Finished")
