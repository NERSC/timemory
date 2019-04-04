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
#  PAPI_ROOT (not cached) The preferred installation prefix for searching for
#  PAPI_ROOT_DIR (cached) PAPI. Set this if the module has problems finding
#                         the proper PAPI installation.
#
# If you don't supply PAPI_ROOT, the module will search on the standard
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

#------------------------------------------------------------------------------#

find_path(PAPI_ROOT
          NAMES
              include/papi.h
          HINTS
              ENV PAPI_ROOT
          DOC
              "PAPI root installation directory")

#------------------------------------------------------------------------------#

find_path(PAPI_INCLUDE_DIR
          NAMES
              papi.h
          HINTS
              ${PAPI_ROOT}
              ENV PAPI_ROOT
              ENV CPATH
          PATH_SUFFIXES
              include
          DOC
              "Path to the PAPI headers")

#------------------------------------------------------------------------------#

find_library(PAPI_LIBRARY
             NAMES
                 papi
             HINTS
                 ${PAPI_ROOT}
                 ENV PAPI_ROOT
                 ENV LD_LIBRARY_PATH
                 ENV LIBRARY_PATH
                 ENV DYLD_LIBRARY_PATH
             PATH_SUFFIXES
                 lib
                 lib64
             DOC
                 "Path to the PAPI library")

#------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set PAPI_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PAPI DEFAULT_MSG
    PAPI_INCLUDE_DIR PAPI_LIBRARY)

#------------------------------------------------------------------------------#

if(PAPI_FOUND)
    get_filename_component(PAPI_INCLUDE_DIRS
        ${PAPI_INCLUDE_DIR} REALPATH)
    get_filename_component(PAPI_LIBRARIES
        ${PAPI_LIBRARY} REALPATH)
endif()

mark_as_advanced(PAPI_INCLUDE_DIR PAPI_LIBRARY)
