# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

#----------------------------------------------------------------------------------------#

IF (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (gperftools_CHECK_COMPONENTS FALSE)
ELSE (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (gperftools_CHECK_COMPONENTS TRUE)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.7)

#----------------------------------------------------------------------------------------#

# Component options
set(_gperftools_COMPONENT_OPTIONS
    profiler
    tcmalloc
    tcmalloc_and_profiler
    tcmalloc_debug
    tcmalloc_minimal
    tcmalloc_minimal_debug
)

#----------------------------------------------------------------------------------------#

IF("${gperftools_FIND_COMPONENTS}" STREQUAL "")
    IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        LIST(APPEND gperftools_FIND_COMPONENTS profiler tcmalloc_debug)
    ELSE()
        LIST(APPEND gperftools_FIND_COMPONENTS tcmalloc_and_profiler)
    ENDIF()
ENDIF("${gperftools_FIND_COMPONENTS}" STREQUAL "")

#----------------------------------------------------------------------------------------#

set(_gperftools_POSSIBLE_LIB_SUFFIXES lib lib64 lib32)

#----------------------------------------------------------------------------------------#

find_path(gperftools_ROOT_DIR
    NAMES
        include/gperftools/tcmalloc.h
        include/google/tcmalloc.h
        include/gperftools/profiler.h
        include/google/profiler.h
    DOC "Google perftools root directory")

#----------------------------------------------------------------------------------------#

find_path(gperftools_INCLUDE_DIR
    NAMES
        gperftools/tcmalloc.h
        google/tcmalloc.h
        gperftools/profiler.h
        google/profiler.h
    HINTS
        ${gperftools_ROOT_DIR}
    PATH_SUFFIXES
        include
    DOC "Google perftools profiler include directory")

#----------------------------------------------------------------------------------------#

set(gperftools_INCLUDE_DIRS ${gperftools_INCLUDE_DIR})

#----------------------------------------------------------------------------------------#
# Find components

FOREACH (_gperftools_COMPONENT ${gperftools_FIND_COMPONENTS})
    IF(NOT "${_gperftools_COMPONENT_OPTIONS}" MATCHES "${_gperftools_COMPONENT}")
        MESSAGE(WARNING "${_gperftools_COMPONENT} is not listed as a real component")
    ENDIF()

    STRING (TOUPPER ${_gperftools_COMPONENT} _gperftools_COMPONENT_UPPER)
    SET (_gperftools_LIBRARY_BASE gperftools_${_gperftools_COMPONENT_UPPER}_LIBRARY)

    SET (_gperftools_LIBRARY_NAME ${_gperftools_COMPONENT})

    FIND_LIBRARY (${_gperftools_LIBRARY_BASE}
        NAMES ${_gperftools_LIBRARY_NAME}
        HINTS ${gperftools_ROOT_DIR}
        PATH_SUFFIXES ${_gperftools_POSSIBLE_LIB_SUFFIXES}
        DOC "MKL ${_gperftools_COMPONENT} library")

    MARK_AS_ADVANCED (${_gperftools_LIBRARY_BASE})

    SET (gperftools_${_gperftools_COMPONENT_UPPER}_FOUND TRUE)

    IF (NOT ${_gperftools_LIBRARY_BASE})
        # Component missing: record it for a later report
        LIST (APPEND _gperftools_MISSING_COMPONENTS ${_gperftools_COMPONENT})
        SET (gperftools_${_gperftools_COMPONENT_UPPER}_FOUND FALSE)
    ENDIF (NOT ${_gperftools_LIBRARY_BASE})

    SET (gperftools_${_gperftools_COMPONENT}_FOUND
        ${gperftools_${_gperftools_COMPONENT_UPPER}_FOUND})

    IF (${_gperftools_LIBRARY_BASE})
        # setup the gperftools_<COMPONENT>_LIBRARIES variable
        SET (gperftools_${_gperftools_COMPONENT_UPPER}_LIBRARIES
            ${${_gperftools_LIBRARY_BASE}})
        LIST (APPEND gperftools_LIBRARIES ${${_gperftools_LIBRARY_BASE}})
    ELSE (${_gperftools_LIBRARY_BASE})
        LIST (APPEND _gperftools_MISSING_LIBRARIES ${_gperftools_LIBRARY_BASE})
    ENDIF (${_gperftools_LIBRARY_BASE})

ENDFOREACH (_gperftools_COMPONENT ${gperftools_FIND_COMPONENTS})


#----- Missing components
IF (DEFINED _gperftools_MISSING_COMPONENTS AND _gperftools_CHECK_COMPONENTS)
    IF (NOT gperftools_FIND_QUIETLY)
        MESSAGE (STATUS "One or more MKL components were not found:")
        # Display missing components indented, each on a separate line
        FOREACH (_gperftools_MISSING_COMPONENT ${_gperftools_MISSING_COMPONENTS})
            MESSAGE (STATUS "  " ${_gperftools_MISSING_COMPONENT})
        ENDFOREACH (_gperftools_MISSING_COMPONENT ${_gperftools_MISSING_COMPONENTS})
    ENDIF (NOT gperftools_FIND_QUIETLY)
ENDIF (DEFINED _gperftools_MISSING_COMPONENTS AND _gperftools_CHECK_COMPONENTS)

#----------------------------------------------------------------------------------------#

mark_as_advanced(gperftools_INCLUDE_DIR gperftools_COMPONENTS gperftools_PROFILER_LIBRARY
    gperftools_ROOT_DIR gperftools_TCMALLOC_LIBRARY)

find_package_handle_standard_args(gperftools DEFAULT_MSG
    gperftools_ROOT_DIR
    gperftools_INCLUDE_DIRS ${_gperftools_MISSING_LIBRARIES})
