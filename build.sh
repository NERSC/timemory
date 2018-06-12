#!/bin/bash

# by default don't enable MPI
: ${TIMEMORY_USE_MPI:=OFF}
export TIMEMORY_USE_MPI

$PYTHON setup.py install
