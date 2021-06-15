
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

if(TIMEMORY_USE_CUPTI)
    set(NAME cupti)
    set(DEPS timemory-cupti timemory-cuda timemory-cuda-device)

    if(CMAKE_CUDA_COMPILER)
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
        PROPERTY_DEPENDS    GLOBAL)
endif()

list(REMOVE_ITEM source_files ${CMAKE_CURRENT_SOURCE_DIR}/extern.cpp)
timemory_install_header_files(
    ${header_files}
    ${source_files})
