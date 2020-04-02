# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Creates a 'format' target that runs clang-format
#
##########################################################################################

set(_FMT ON)
# Visual Studio GUI reports "errors" occasionally
if(WIN32)
    set(_FMT OFF)
endif()

option(TIMEMORY_FORMAT_TARGET "Enable a clang-format target" ${_FMT})
mark_as_advanced(TIMEMORY_FORMAT_TARGET)

if(NOT TIMEMORY_FORMAT_TARGET)
    return()
endif()

# prefer clang-format 6.0
find_program(CLANG_FORMATTER
    NAMES
        clang-format-6
        clang-format-6.0
        clang-format-mp-6.0
        clang-format)

if(CLANG_FORMATTER)
    file(GLOB_RECURSE headers
        ${PROJECT_SOURCE_DIR}/source/tools/*.hpp
        ${PROJECT_SOURCE_DIR}/source/python/*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/*.h
        ${PROJECT_SOURCE_DIR}/source/timemory/*.hpp)
    if(TIMEMORY_SOURCE_GROUP)
        source_group(TREE ${PROJECT_SOURCE_DIR}/source FILES ${headers})
    endif()
    file(GLOB_RECURSE sources
        ${PROJECT_SOURCE_DIR}/source/*.c
        ${PROJECT_SOURCE_DIR}/source/*.cu
        ${PROJECT_SOURCE_DIR}/source/*.cpp)
    if(TIMEMORY_SOURCE_GROUP)
        source_group(TREE ${PROJECT_SOURCE_DIR}/source FILES ${sources})
    endif()
    if(TIMEMORY_BUILD_EXAMPLES)
        file(GLOB_RECURSE examples
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.h
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.c
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.hpp
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.cpp
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.cuh
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.cu)
        if(TIMEMORY_SOURCE_GROUP)
            source_group(TREE ${PROJECT_SOURCE_DIR}/examples FILES ${examples})
        endif()
    else()
        set(examples)
    endif()

    set(FORMAT_NAME format)
    if(TARGET format)
        set(FORMAT_NAME format-timemory)
    endif()

    # always have files
    set(_COMMAND
        COMMAND ${CLANG_FORMATTER} -i ${headers}
        COMMAND ${CLANG_FORMATTER} -i ${sources})

    # might be empty
    if(TIMEMORY_BUILD_EXAMPLES)
        set(_COMMAND ${_COMMAND}
            COMMAND ${CLANG_FORMATTER} -i ${examples})
    endif()

    add_custom_target(${FORMAT_NAME}
        ${_COMMAND}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running '${CLANG_FORMATTER}'..."
        SOURCES ${headers} ${sources} ${examples})
endif()
