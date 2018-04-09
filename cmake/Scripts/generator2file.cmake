#  script for generating files based on generator expressions
#  (C) Copyright Jonathan Madsen 2018.

# Variables that must be defined
#   OUTPUT (output files)
#   NAMES (list of variables)
#   VALUES (list of variable values)
#   LENGTH (length of NAMES and VALUES lists)
#

# Invocation example:
#
#   set(TIMEMORY_INFO_FILE ${CMAKE_BINARY_DIR}/timemory/lib/timemory_pathinfo.txt)
#   add_custom_target(${PROJECT_NAME}-shared-library-path ALL
#       COMMAND ${CMAKE_COMMAND}
#           -DOUTPUT=${TIMEMORY_INFO_FILE}
#           -DLENGTH=2
#           -DNAMES="timemory_shared_library\;timemory_dynamic_library"
#           -DVALUES="$<TARGET_FILE:${LIBNAME}-shared>\;$<TARGET_FILE:${LIBNAME}>"
#           -P ${PROJECT_SOURCE_DIR}/cmake/Scripts/generator2file.cmake
#       COMMENT "Generating timemory_pathinfo.txt..."
#       VERBATIM)
#

macro(check_defined VAR)
    if(NOT DEFINED ${VAR} OR "${${VAR}}" STREQUAL "")
        message(FATAL_ERROR "Variable ${VAR} must be defined to run Version.cmake")
    endif()
endmacro()

check_defined(OUTPUT)
check_defined(NAMES)
check_defined(VALUES)
check_defined(LENGTH)

# function for generating a list of indices that can be used in foreach(...)
# usage:
#   for_loop_list(_LEN 0 4)    --> *same as* --> set(_LEN 0 1 2 3)
#   for_loop_list(_LEN 4 0 -1) --> *same as* --> set(_LEN 4 3 2 1)
#
# note: using non-integers will cause problems
#
function(FOR_LOOP_LIST _VAR _BEGIN _END)
    set(_INDEX ${_BEGIN})
    set(_LIST )

    if(NOT "${ARGN}" STREQUAL "")
        set(_INCR ${ARGN})
    else(NOT "${ARGN}" STREQUAL "")
        set(_INCR 1)
    endif(NOT "${ARGN}" STREQUAL "")

    while(NOT "${_INDEX}" STREQUAL "${_END}")
        list(APPEND _LIST ${_INDEX})
        math(EXPR _INDEX "${_INDEX} + (${_INCR})")
        #message("LIST: ${_LIST}")
        #message("INDEX: ${_INDEX}")
    endwhile(NOT "${_INDEX}" STREQUAL "${_END}")

    set(${_VAR} ${_LIST} PARENT_SCOPE)
endfunction(FOR_LOOP_LIST _VAR _BEGIN _END)

set(_STR )
for_loop_list(_LEN 0 ${LENGTH})
#message(STATUS "FOR LOOP LIST: ${_LEN}")

foreach(_I ${_LEN})
    #message(STATUS "-- i = ${_I}")
    list(GET NAMES ${_I} _NAME)
    list(GET VALUES ${_I} _VALUE)
    #message(STATUS "NAME = ${_NAME}, VALUE = ${_VALUE}")
    if(WIN32)
        string(REPLACE "/" "\\" _NAME "${_NAME}")
        string(REPLACE "/" "\\" _VALUE "${_VALUE}")
    endif(WIN32)
    string(REPLACE " " "\\ " _NAME "${_NAME}")
    string(REPLACE " " "\\ " _VALUE "${_VALUE}")
    string(REPLACE "\"" "" _NAME "${_NAME}")
    string(REPLACE "\"" "" _VALUE "${_VALUE}")
    set(_STR "${_STR}${_NAME} ${_VALUE}\n")
endforeach(_I ${_LEN})

file(WRITE ${OUTPUT} "${_STR}")
