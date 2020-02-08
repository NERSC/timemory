#!@PYTHON_EXECUTABLE@


import sys
import numpy
import argparse
import mpi4py
from mpi4py import MPI

import timemory
from timemory.profiler import profile
from timemory.util import auto_timer

import libex_python_bindings as ex_bindings

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

@auto_timer
def run_profile(nitr=10, nsize=5000):
    return ex_bindings.run(nitr, nsize)

def run_mpi():

    # passing MPI datatypes explicitly
    if rank == 0:
        data = numpy.arange(1000, dtype='i')
        comm.Send([data, MPI.INT], dest=1, tag=77)
    elif rank == 1:
        data = numpy.empty(1000, dtype='i')
        comm.Recv([data, MPI.INT], source=0, tag=77)

    # automatic MPI datatype discovery
    if rank == 0:
        data = numpy.arange(100, dtype=numpy.float64)
        comm.Send(data, dest=1, tag=13)
    elif rank == 1:
        data = numpy.empty(100, dtype=numpy.float64)
        comm.Recv(data, source=0, tag=13)

def main(args):
    # run_mpi()
    ans = run_profile(args.iterations, args.size)
    print("Answer = {}".format(ans))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', required=False,
                        default=10, type=int, help="Iterations")
    parser.add_argument('-n', '--size', required=False,
                        default=5000, type=int, help="Array size")

    args = parser.parse_args()
    timemory.enable_signal_detection()

    id = timemory.init_mpip()
    main(args)
    timemory.stop_mpip(id)
    timemory.finalize()
    print("Finished")


