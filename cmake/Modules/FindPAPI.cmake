# Try to find the PAPI library and headers
# Usage of this module is as follows
#
#     find_package( PAPI )
#     if(PAPI_FOUND)
#         include_directories(${PAPI_INCLUDE_DIRS})
#         add_executable(foo foo.cc)
#         target_link_libraries(foo ${PAPI_LIBRARIES})
#     endif()
#
# You can provide a minimum version number that should be used.
# If you provide this version number and specify the REQUIRED attribute,
# this module will fail if it can't find a PAPI of the specified version
# or higher. If you further specify the EXACT attribute, then this module
# will fail if it can't find a PAPI with a version eaxctly as specified.
#
# ===========================================================================
# Variables used by this module which can be used to change the default
# behaviour, and hence need to be set before calling find_package:
#
#  PAPI_ROOT_DIR
#       The preferred installation prefix for searching for PAPI
#       Set this if the module has problems finding the proper PAPI installation.
#
# If you don't supply PAPI_ROOT_DIR, the module will search on the standard
# system paths.
#
# ============================================================================
# Variables set by this module:
#
#  PAPI_FOUND           System has PAPI.
#
#  PAPI_INCLUDE_DIRS    PAPI include directories: not cached.
#
#  PAPI_LIBRARIES       Link to these to use the PAPI library: not cached.
#
# ===========================================================================
# If PAPI is installed in a non-standard way, e.g. a non GNU-style install
# of <prefix>/{lib,include}, then this module may fail to locate the headers
# and libraries as needed. In this case, the following cached variables can
# be editted to point to the correct locations.
#
#  PAPI_INCLUDE_DIR    The path to the PAPI include directory: cached
#
#  PAPI_LIBRARY        The path to the PAPI library: cached
#
# You should not need to set these in the vast majority of cases
#

#----------------------------------------------------------------------------------------#
include(CMakeParseArguments)

function(FIND_STATIC_LIBRARY _VAR)
    set(_options    )
    set(_onevalue   DOC)
    set(_multival   NAMES HINTS PATHS PASS_SUFFIXES)

    cmake_parse_arguments(
        LIBRARY "${_options}" "${_onevalue}" "${_multival}" ${ARGN})

    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    find_library(${_VAR}
                NAMES ${LIBRARY_NAMES}
                HINTS ${LIBRARY_HINTS}
                PATHS ${LIBRARY_PATHS}
                PATH_SUFFIXES ${LIBRARY_PATH_SUFFIXES}
                DOC "${LIBRARY_DOC}")
endfunction()

#----------------------------------------------------------------------------------------#

find_path(PAPI_ROOT_DIR
    NAMES
        include/papi.h
    HINTS
        ENV PAPI_ROOT_DIR
    DOC
        "PAPI root installation directory")

#----------------------------------------------------------------------------------------#

find_path(PAPI_INCLUDE_DIR
    NAMES
        papi.h
    HINTS
        ${PAPI_ROOT_DIR}
        ENV PAPI_ROOT_DIR
        ENV CPATH
    PATH_SUFFIXES
        include
    DOC
        "Path to the PAPI headers")

#----------------------------------------------------------------------------------------#

find_library(PAPI_LIBRARY
    NAMES
        papi
    HINTS
        ${PAPI_ROOT_DIR}
        ENV PAPI_ROOT_DIR
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
        ENV DYLD_LIBRARY_PATH
    PATH_SUFFIXES
        lib
        lib64
    DOC
        "Path to the PAPI library")

#----------------------------------------------------------------------------------------#

find_library(PAPI_pfm_LIBRARY
    NAMES
        pfm libpfm.so libpfm.so.4
    HINTS
        ${PAPI_ROOT_DIR}
        ENV PAPI_ROOT_DIR
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
        ENV DYLD_LIBRARY_PATH
    PATH_SUFFIXES
        lib
        lib64
    DOC
        "Path to the PAPI library")

#----------------------------------------------------------------------------------------#

find_static_library(PAPI_STATIC_LIBRARY
    NAMES
        papi
    HINTS
        ${PAPI_ROOT_DIR}
        ENV PAPI_ROOT_DIR
        ENV LD_LIBRARY_PATH
        ENV LIBRARY_PATH
        ENV DYLD_LIBRARY_PATH
    PATH_SUFFIXES
        lib
        lib64
    DOC
        "Path to the PAPI library")

#
#----------------------------------------------------------------------------------------#

find_static_library(PAPI_pfm_STATIC_LIBRARY
    NAMES
        pfm libpfm.a
    HINTS
        ${PAPI_ROOT_DIR}
        ENV PAPI_ROOT_DIR
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
        ENV DYLD_LIBRARY_PATH
    PATH_SUFFIXES
        lib
        lib64
    DOC
        "Path to the PAPI library")

#----------------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set PAPI_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PAPI DEFAULT_MSG
    PAPI_INCLUDE_DIR PAPI_LIBRARY)

#----------------------------------------------------------------------------------------#

if(PAPI_FOUND)
    add_library(papi-shared INTERFACE)
    add_library(papi-static INTERFACE)
    target_link_libraries(papi-shared INTERFACE ${PAPI_LIBRARY})
    target_link_libraries(papi-static INTERFACE ${PAPI_STATIC_LIBRARY})
    target_include_directories(papi-shared INTERFACE ${PAPI_INCLUDE_DIR})
    target_include_directories(papi-static INTERFACE ${PAPI_INCLUDE_DIR})
    get_filename_component(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR} REALPATH)
    get_filename_component(PAPI_LIBRARIES ${PAPI_LIBRARY} REALPATH)
    if(PAPI_pfm_LIBRARY)
        target_link_libraries(papi-shared INTERFACE ${PAPI_pfm_LIBRARY})
        list(APPEND PAPI_LIBRARIES ${PAPI_pfm_LIBRARY})
    elseif(PAPI_pfm_STATIC_LIBRARY)
        target_link_libraries(papi-shared INTERFACE ${PAPI_pfm_STATIC_LIBRARY})
        list(APPEND PAPI_LIBRARIES ${PAPI_pfm_STATIC_LIBRARY})
    endif()
    if(PAPI_pfm_STATIC_LIBRARY)
        target_link_libraries(papi-static INTERFACE ${PAPI_pfm_STATIC_LIBRARY})
    else()
        target_link_libraries(papi-static INTERFACE ${PAPI_pfm_LIBRARY})
    endif()
endif()

mark_as_advanced(PAPI_INCLUDE_DIR PAPI_LIBRARY)
