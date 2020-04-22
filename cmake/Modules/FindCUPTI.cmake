# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

#----------------------------------------------------------------------------------------#

set(_CUDA_PATHS $ENV{CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR})
set(_CUDA_INC ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#----------------------------------------------------------------------------------------#

find_path(CUPTI_ROOT_DIR
    NAMES           include/cupti.h
    HINTS           ${_CUDA_PATHS}
    PATHS           ${_CUDA_PATHS}
    PATH_SUFFIXES   extras/CUPTI)

#----------------------------------------------------------------------------------------#

# try to find cupti header
find_path(CUPTI_INCLUDE_DIR
    NAMES           cupti.h
    HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_INC} ${_CUDA_PATHS}
    PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_INC} ${_CUDA_PATHS}
    PATH_SUFFIXES   extras/CUPTI/include extras/CUPTI extras/include CUTPI/include include)

#----------------------------------------------------------------------------------------#

# try to find cuda driver library
find_library(CUPTI_cupti_LIBRARY
    NAMES           cupti
    HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia)

if(CUPTI_cupti_LIBRARY)
    get_filename_component(CUPTI_cupti_LIBRARY_DIR "${CUPTI_cupti_LIBRARY}" PATH CACHE)
endif()

#----------------------------------------------------------------------------------------#

# try to find nvperf_host library
find_library(CUPTI_nvperf_host_LIBRARY
    NAMES           nvperf_host
    HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia)

#----------------------------------------------------------------------------------------#

# try to find nvperf_host library
find_library(CUPTI_nvperf_host_STATIC_LIBRARY
    NAMES           nvperf_host_static
    HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia)

#----------------------------------------------------------------------------------------#

# try to find nvperf_target library
find_library(CUPTI_nvperf_target_LIBRARY
    NAMES           nvperf_target
    HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia)

#----------------------------------------------------------------------------------------#

# try to find cuda driver library
find_library(CUPTI_cuda_LIBRARY
    NAMES           cuda
    HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
    PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia)

#----------------------------------------------------------------------------------------#

# try to find cuda driver stubs library if no real driver
if(NOT CUPTI_cuda_LIBRARY)
    find_library(CUPTI_cuda_LIBRARY
        NAMES           cuda
        HINTS           ${_CUDA_PATHS}
        PATHS           ${_CUDA_PATHS}
        PATH_SUFFIXES   lib/stubs lib64/stubs stubs)
    set(HAS_CUDA_DRIVER_LIBRARY OFF CACHE BOOL "Using stubs library")
else()
    set(HAS_CUDA_DRIVER_LIBRARY ON CACHE BOOL "Using stubs library")
endif()

#------------------------------------------------------------------------------#

mark_as_advanced(
    CUPTI_INCLUDE_DIR
    CUPTI_cupti_LIBRARY_DIR
    CUPTI_cupti_LIBRARY
    CUPTI_cuda_LIBRARY)

#------------------------------------------------------------------------------#

find_package_handle_standard_args(CUPTI DEFAULT_MSG
    CUPTI_INCLUDE_DIR
    CUPTI_cupti_LIBRARY
    CUPTI_cuda_LIBRARY
    CUPTI_cupti_LIBRARY_DIR)

#------------------------------------------------------------------------------#

if(CUPTI_FOUND)
    set(CUPTI_INCLUDE_DIRS ${CUPTI_INCLUDE_DIR})
    set(CUPTI_LIBRARIES ${CUPTI_cupti_LIBRARY} ${CUPTI_cuda_LIBRARY})
    set(CUPTI_LIBRARY_DIRS ${CUPTI_cupti_LIBRARY_DIR})
endif()

#------------------------------------------------------------------------------#

if(CUPTI_nvperf_host_LIBRARY)
    list(APPEND CUPTI_LIBRARIES ${CUPTI_nvperf_host_LIBRARY})
endif()

#------------------------------------------------------------------------------#

if(CUPTI_nvperf_target_LIBRARY)
    list(APPEND CUPTI_LIBRARIES ${CUPTI_nvperf_target_LIBRARY})
endif()

#------------------------------------------------------------------------------#

unset(_CUDA_PATHS)
unset(_CUDA_INC)

#------------------------------------------------------------------------------#
