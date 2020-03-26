
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
    set_source_files_properties(${source_files} PROPERTIES
        LANGUAGE CUDA LINKER_LANGUAGE CUDA)
else()
    file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
endif()

build_intermediate_library(
    NAME                ert
    TARGET              ert
    CATEGORY            EXTERN
    FOLDER              extern
    HEADERS             ${header_files}
    SOURCES             ${source_files}
    DEPENDS             timemory-vector
                        timemory-timing-component
                        timemory-cuda
                        timemory-cuda-compiler
                        # timemory-cudart
    PROPERTY_DEPENDS    GLOBAL)
