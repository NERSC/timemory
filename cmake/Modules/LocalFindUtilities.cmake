# Utilities for find_package
#
#
include(CMakeParseArguments)

#
#   find_static_library(...)
#       finds the static library version
#
FUNCTION(FIND_STATIC_LIBRARY _VAR)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    FIND_LIBRARY(${_VAR} ${ARGN})
ENDFUNCTION()

#
#   find_root_hints(...)
#       get the realpath to certain files
#       and then walks up the directory tree
#
FUNCTION(FIND_ROOT_PATH _OUTPUT_VAR _FNAME)

    IF(DEFINED ${_OUTPUT_VAR} AND ${_OUTPUT_VAR})
        RETURN()
    ENDIF()

    # default behavior is local variable in parent scope
    SET(_SCOPED ON)
    IF("CACHE" IN_LIST ARGN)
        SET(_SCOPED OFF)
        LIST(REMOVE_ITEM ARGN CACHE)
    ENDIF()

    # find the file
    FIND_FILE(_FILE_REALPATH ${_FNAME} ${ARGN})

    SET(_HINT)
    IF(_FILE_REALPATH)
        # get the realpath
        GET_FILENAME_COMPONENT(_FILE_REALPATH_0 "${_FILE_REALPATH}" REALPATH)
        # for the loop
        SET(_LAST_VAR _FILE_REALPATH_0)
        # loop over two parent directories
        FOREACH(_IDX 1 2)
            # this is the current variable
            SET(_CURR_VAR _FILE_REALPATH_${_IDX})
            # get the directory of the current variable
            GET_FILENAME_COMPONENT(${_CURR_VAR} "${${_LAST_VAR}}" DIRECTORY)
            # append the hint
            SET(_HINT ${${_CURR_VAR}})
            # set the last variable to the parent directory
            SET(_LAST_VAR ${_CURR_VAR})
        ENDFOREACH()
        # set the variable
        IF(_SCOPED)
            SET(${_OUTPUT_VAR} "${_HINT}" PARENT_SCOPE)
        ELSE()
            SET(${_OUTPUT_VAR} "${_HINT}" CACHE STRING "Root path hints to ${_FNAME}")
        ENDIF()
    ELSE()
        SET(${_OUTPUT_VAR} ${_OUTPUT_VAR}-NOTFOUND PARENT_SCOPE)
    ENDIF()

    # remove the cache entry for FIND_FILE
    UNSET(_FILE_REALPATH CACHE)
ENDFUNCTION()
