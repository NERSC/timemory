# Try to find the libraries and headers for Caliper optional dependencies
# Usage of this module is as follows
#
#     include(CaliperDepends)
#

set(CALIPER_OPTION_PREFIX ON CACHE INTERNAL "Prefix caliper options with CALIPER_")

# caliper uses PAPI high-level API so do not enable by default
set(CALIPER_WITH_PAPI       OFF     CACHE BOOL "Enable PAPI in Caliper")
set(CALIPER_WITH_MPI        OFF     CACHE BOOL "Enable MPI in Caliper")
set(CALIPER_WITH_CUPTI      OFF     CACHE BOOL "Enable CUPTI in Caliper")

if(TIMEMORY_USE_CUPTI)
    set(CALIPER_WITH_CUPTI OFF CACHE BOOL "Enable cupti in Caliper")
endif()

find_path(LIBUNWIND_INCLUDE_DIR
    NAMES           unwind.h libunwind.h
    PATH_SUFFIXES   include)

find_library(LIBUNWIND_LIBRARY
    NAMES           unwind
    PATH_SUFFIXES   lib lib64)

find_library(LIBUNWIND_STATIC_LIBRARY
    NAMES           libunwind.a
    PATH_SUFFIXES   lib lib64)

if(LIBUNWIND_INCLUDE_DIR AND LIBUNWIND_LIBRARY)
    set(CALIPER_WITH_CALLPATH ON  CACHE BOOL "Enable libunwind in Caliper")
else()
    set(CALIPER_WITH_CALLPATH OFF CACHE BOOL "Enable libunwind in Caliper")
endif()

if(APPLE)
    set(CALIPER_WITH_GOTCHA OFF CACHE BOOL "Enable gotcha in Caliper")
endif()
