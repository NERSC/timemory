
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
else()
    file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
endif()

build_intermediate_library(
    USE_INTERFACE
    NAME                ert
    TARGET              ert
    CATEGORY            EXTERN
    FOLDER              components
    HEADERS             ${header_files}
    SOURCES             ${source_files}
    DEPENDS             timemory-vector
                        timemory-cuda
                        # timemory-cudart
                        timemory-timing-component
    PRIVATE_LINK        timemory-roofline-options
    PROPERTY_DEPENDS    GLOBAL)
