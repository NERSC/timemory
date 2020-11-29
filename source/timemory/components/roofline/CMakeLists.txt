
if(TIMEMORY_USE_PAPI OR TIMEMORY_USE_CUPTI)
    set(NAME roofline)
    set(DEPS timemory-papi-component timemory-cupti-component)

    file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
    if(CMAKE_CUDA_COMPILER AND TIMEMORY_USE_CUPTI)
        file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                       ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
    else()
        file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    endif()

    if(TIMEMORY_USE_CUPTI AND CMAKE_CUDA_COMPILER)
        set_source_files_properties(${source_files} PROPERTIES
            LINKER_LANGUAGE CUDA)
    endif()

    build_intermediate_library(USE_INTERFACE
        NAME                ${NAME}
        TARGET              ${NAME}-component
        CATEGORY            COMPONENT
        FOLDER              components
        HEADERS             ${header_files}
        SOURCES             ${source_files}
        DEPENDS             ${DEPS}
        PRIVATE_LINK        timemory-cpu-roofline timemory-gpu-roofline timemory-roofline
        PROPERTY_DEPENDS    GLOBAL)
endif()
