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

# python formatting
find_program(BLACK_FORMATTER
    NAMES black)

if(CLANG_FORMATTER)
    file(GLOB_RECURSE _headers
        ${PROJECT_SOURCE_DIR}/source/tools/*.hpp
        ${PROJECT_SOURCE_DIR}/source/tests/*.hpp
        ${PROJECT_SOURCE_DIR}/source/python/*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/*.h
        ${PROJECT_SOURCE_DIR}/source/timemory/*.hpp)
    if(TIMEMORY_SOURCE_GROUP)
        source_group(TREE ${PROJECT_SOURCE_DIR}/source FILES ${headers})
    endif()
    file(GLOB_RECURSE sources
        ${PROJECT_SOURCE_DIR}/source/*.c
        ${PROJECT_SOURCE_DIR}/source/*.cu
        ${PROJECT_SOURCE_DIR}/source/*.cpp
        ${PROJECT_SOURCE_DIR}/source/*.cpp.in
        ${PROJECT_SOURCE_DIR}/source/tools/*.c
        ${PROJECT_SOURCE_DIR}/source/tools/*.cu
        ${PROJECT_SOURCE_DIR}/source/tools/*.cpp
        ${PROJECT_SOURCE_DIR}/source/tools/*.cpp.in
        ${PROJECT_SOURCE_DIR}/source/tests/*.c
        ${PROJECT_SOURCE_DIR}/source/tests/*.cu
        ${PROJECT_SOURCE_DIR}/source/tests/*.cpp
        ${PROJECT_SOURCE_DIR}/source/tests/*.cpp.in)
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

    # remove rapidjson and rapidxml
    set(_tpl_base "${PROJECT_SOURCE_DIR}/source/timemory/tpls")
    file(GLOB_RECURSE _tpl_headers "${_tpl_base}/*")
    file(GLOB_RECURSE _ext_headers "${_tpl_base}/cereal/cereal/external/*")
    set(headers)
    set(tpl_headers)
    foreach(_h ${_headers})
        if(NOT "${_h}" IN_LIST _tpl_headers)
	        list(APPEND headers ${_h})
        elseif(NOT "${_h}" IN_LIST _ext_headers)
	        list(APPEND tpl_headers ${_h})
    	endif()
    endforeach()
    unset(_tpl_headers)
    unset(_ext_headers)
    unset(_headers)

    # name of the format target
    set(FORMAT_NAME format)
    if(TARGET format)
        set(FORMAT_NAME format-timemory)
    endif()

    # always have files
    set(_COMMAND
        COMMAND ${CLANG_FORMATTER} -i ${headers}
        COMMAND ${CLANG_FORMATTER} -i ${sources})

    # might have many files
    if(tpl_headers)
        set(_COMMAND ${_COMMAND}
            COMMAND ${CLANG_FORMATTER} -i ${tpl_headers})
    endif()

    # might be empty
    if(TIMEMORY_BUILD_EXAMPLES)
        set(_COMMAND ${_COMMAND}
            COMMAND ${CLANG_FORMATTER} -i ${examples})
    endif()

    set(_MSG "'${CLANG_FORMATTER}'")
    if(BLACK_FORMATTER)
        set(_COMMAND ${_COMMAND}
            COMMAND ${BLACK_FORMATTER} -q ${PROJECT_SOURCE_DIR})
        set(_MSG "${_MSG} and '${BLACK_FORMATTER}'")
    endif()

    add_custom_target(${FORMAT_NAME}
        ${_COMMAND}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "[${PROJECT_NAME}] Running ${_MSG}..."
        SOURCES ${headers} ${sources} ${examples})

    unset(_MSG)
endif()
