# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Creates a 'format' target that runs clang-format
#
##########################################################################################


find_program(CLANG_FORMATTER
    NAMES
        clang-format-8.0
        clang-format-7.0
        clang-format-6.0
        clang-format)

if(CLANG_FORMATTER)
    file(GLOB headers
        ${PROJECT_SOURCE_DIR}/source/timemory/*.h
        ${PROJECT_SOURCE_DIR}/source/timemory/*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/impl/*.icpp
        ${PROJECT_SOURCE_DIR}/source/timemory/components/*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/*.thpp
        ${PROJECT_SOURCE_DIR}/source/python/*.hpp)
    file(GLOB sources
        ${PROJECT_SOURCE_DIR}/source/*.c
        ${PROJECT_SOURCE_DIR}/source/*.cpp
        ${PROJECT_SOURCE_DIR}/source/tools/*.cpp
        ${PROJECT_SOURCE_DIR}/source/python/*.cpp)
    file(GLOB_RECURSE examples
        ${PROJECT_SOURCE_DIR}/examples/*.h
        ${PROJECT_SOURCE_DIR}/examples/*.c
        ${PROJECT_SOURCE_DIR}/examples/*.hpp
        ${PROJECT_SOURCE_DIR}/examples/*.cpp
        ${PROJECT_SOURCE_DIR}/examples/*.cuh
        ${PROJECT_SOURCE_DIR}/examples/*.cu)

    set(FORMAT_NAME format)
    if(TARGET format)
        set(FORMAT_NAME format-timemory)
    endif()

    add_custom_target(${FORMAT_NAME}
        COMMAND ${CLANG_FORMATTER} -i ${headers} ${sources} ${examples}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running '${CLANG_FORMATTER}'..."
        SOURCES ${headers} ${sources} ${examples})
endif()
