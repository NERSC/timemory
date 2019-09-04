cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#----------------------------------------------------------------------------------------#
#   TiMemory examples project
#
project(TiMemory-Examples C CXX)

#----------------------------------------------------------------------------------------#
#   Add the examples
#

add_subdirectory(ex-c)
add_subdirectory(ex-optional)
add_subdirectory(ex-cxx-basic)
add_subdirectory(ex-cxx-tuple)
add_subdirectory(ex-cxx-overhead)
add_subdirectory(ex-cpu-roofline)
add_subdirectory(ex-gpu-roofline)
add_subdirectory(ex-cuda-event)
add_subdirectory(ex-caliper)
add_subdirectory(ex-ert)

