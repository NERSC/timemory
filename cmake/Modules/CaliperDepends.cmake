# Try to find the libraries and headers for Caliper optional dependencies
# Usage of this module is as follows
#
#     include(CaliperDepends)
#

# caliper uses PAPI high-level API so do not enable by default
set(WITH_PAPI       OFF                   CACHE BOOL "Enable PAPI in Caliper")
set(WITH_MPI        ${TIMEMORY_USE_MPI}   CACHE BOOL "Enable MPI in Caliper")
set(WITH_CUPTI      ${TIMEMORY_USE_CUPTI} CACHE BOOL "Enable CUPTI in Caliper")
set(WITH_CALLPATH   OFF                   CACHE BOOL "Enable libunwind in Caliper")
# set(WITH_TAU        OFF)
# set(WITH_NVPROF     ${TIMEMORY_USE_CUDA})

#if(TIMEMORY_BUILD_GOTCHA)
#    set(USE_EXTERNAL_GOTCHA ON)
#endif()

if(PAPI_pfm_LIBRARY)
    set(WITH_LIBPFM     ON CACHE BOOL "Enable libpfm in Caliper")
    set(LIBPFM_LIBRARY  ${PAPI_pfm_LIBRARY})
else()
    set(WITH_LIBPFM     OFF)
endif()

if(TIMEMORY_USE_CUPTI)
    set(CUPTI_PREFIX      ${CUPTI_ROOT_DIR}         CACHE PATH      "CUpti root directory")
    set(CUPTI_INCLUDE_DIR ${CUDA_cupti_INCLUDE_DIR} CACHE PATH      "CUpti include directory")
    set(CUPTI_LIBRARY     ${CUDA_cupti_LIBRARY}     CACHE FILEPATH  "CUpti library")
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
    set(WITH_CALLPATH ON  CACHE BOOL "Enable libunwind in Caliper")
else()
    set(WITH_CALLPATH OFF CACHE BOOL "Enable libunwind in Caliper")
endif()

if(APPLE)
    set(WITH_GOTCHA OFF)
endif()
