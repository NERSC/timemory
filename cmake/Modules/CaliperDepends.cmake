# Try to find the libraries and headers for Caliper optional dependencies
# Usage of this module is as follows
#
#     include(CaliperDepends)
#


set(WITH_PAPI       ${TIMEMORY_USE_PAPI})
set(WITH_MPI        ${TIMEMORY_USE_MPI})
set(WITH_CUPTI      ${TIMEMORY_USE_CUPTI})
set(WITH_CALLPATH   OFF)
set(WITH_LIBPFM     OFF)
# set(WITH_TAU        OFF)
# set(WITH_NVPROF     ${TIMEMORY_USE_CUDA})

if(PAPI_pfm_LIBRARY)
    set(WITH_LIBPFM     ON)
    set(LIBPFM_LIBRARY  ${PAPI_pfm_LIBRARY})
endif()

if(TIMEMORY_USE_CUPTI)
    set(CUPTI_PREFIX ${CUPTI_ROOT_DIR})
    set(CUPTI_LIBRARY ${CUDA_cupti_LIBRARY})
endif()

#find_package(TAU QUIET)
#if(TAU_FOUND)
    # set(WITH_TAU ${TAU_FOUND})
    # set(TAU_DIR ${TAU_ROOT_DIR})
#endif()

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
    set(WITH_CALLPATH ON)
endif()
