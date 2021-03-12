
if(NOT TIMEMORY_BUILD_ERT)
    return()
endif()

set(CMAKE_UNITY_BUILD OFF)

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE cu_source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
    list(APPEND source_files ${cu_source_files})
endif()

build_intermediate_library(
    USE_INTERFACE
    NAME                ert
    TARGET              ert
    CATEGORY            GLOBAL
    FOLDER              components
    HEADERS             ${header_files}
    SOURCES             ${source_files}
    DEPENDS             timemory::timemory-vector
                        timemory::timemory-cuda
                        timemory::timemory-core
                        timemory::timemory-manager
    PRIVATE_LINK        timemory::timemory-roofline-options)
