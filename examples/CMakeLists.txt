cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

cmake_policy(PUSH)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#----------------------------------------------------------------------------------------#
#   timemory examples project
#
project(timemory-Examples C CXX)

#----------------------------------------------------------------------------------------#
#   Add the examples
#

add_subdirectory(ex-c)
add_subdirectory(ex-minimal)
add_subdirectory(ex-derived)
add_subdirectory(ex-optional)
add_subdirectory(ex-cxx-basic)
add_subdirectory(ex-cxx-tuple)
add_subdirectory(ex-cxx-overhead)
add_subdirectory(ex-statistics)

# external package related
add_subdirectory(ex-caliper)
add_subdirectory(ex-gotcha)
add_subdirectory(ex-likwid)

# GPU related
add_subdirectory(ex-cuda-event)

# Roofline related
add_subdirectory(ex-ert)
add_subdirectory(ex-cpu-roofline)
add_subdirectory(ex-gpu-roofline)

# Python related
add_subdirectory(ex-python)

# Dynamic instrumentation
add_subdirectory(ex-custom-dynamic-instr)
cmake_policy(POP)
