# Try to find the TAU library and headers
# Usage of this module is as follows
#
#     find_package( TAU )
#     if(TAU_FOUND)
#         include_directories(${TAU_INCLUDE_DIRS})
#         add_executable(foo foo.cc)
#         target_link_libraries(foo ${TAU_LIBRARIES})
#     endif()
#
# You can provide a minimum version number that should be used.
# If you provide this version number and specify the REQUIRED attribute,
# this module will fail if it can't find a TAU of the specified version
# or higher. If you further specify the EXACT attribute, then this module
# will fail if it can't find a TAU with a version eaxctly as specified.
#
# ===========================================================================
# Variables used by this module which can be used to change the default
# behaviour, and hence need to be set before calling find_package:
#
#  TAU_ROOT_DIR
#       The preferred installation prefix for searching for TAU
#       Set this if the module has problems finding the proper TAU installation.
#
# If you don't supply TAU_ROOT_DIR, the module will search on the standard
# system paths.
#
# ============================================================================
# Variables set by this module:
#
#  TAU_FOUND           System has TAU.
#
#  TAU_INCLUDE_DIRS    TAU include directories: not cached.
#
#  TAU_LIBRARIES       Link to these to use the TAU library: not cached.
#
# ===========================================================================
# If TAU is installed in a non-standard way, e.g. a non GNU-style install
# of <prefix>/{lib,include}, then this module may fail to locate the headers
# and libraries as needed. In this case, the following cached variables can
# be editted to point to the correct locations.
#
#  TAU_INCLUDE_DIR    The path to the TAU include directory: cached
#
#  TAU_LIBRARY        The path to the TAU library: cached
#
# You should not need to set these in the vast majority of cases
#

#----------------------------------------------------------------------------------------#

find_path(TAU_ROOT_DIR
    NAMES
        include/TAU.h tau/include/TAU.h include/tau/TAU.h
    HINTS
        ENV TAU_ROOT_DIR
    PATH_SUFFIXES
        lib lib64
    DOC
        "TAU root installation directory")

#----------------------------------------------------------------------------------------#

find_path(TAU_INCLUDE_DIR
    NAMES
        TAU.h
    HINTS
        ${TAU_ROOT_DIR}
        ENV TAU_ROOT_DIR
        ENV CPATH
    PATH_SUFFIXES
        include include/tau tau/include
    DOC
        "Path to the TAU headers")

#----------------------------------------------------------------------------------------#

find_library(TAU_LIBRARY
    NAMES
        TAU
    HINTS
        ${TAU_ROOT_DIR}
        ENV TAU_ROOT_DIR
        ENV LD_LIBRARY_PATH
        ENV LIBRARY_PATH
        ENV DYLD_LIBRARY_PATH
    PATH_SUFFIXES
        lib
        lib64
        tau
        lib/tau
        lib64/tau
        # system processor
        lib/tau/${CMAKE_SYSTEM_PROCESSOR}
        lib64/tau/${CMAKE_SYSTEM_PROCESSOR}
        lib/tau/${CMAKE_SYSTEM_PROCESSOR}/lib
        lib64/tau/${CMAKE_SYSTEM_PROCESSOR}/lib64
        lib64/tau/${CMAKE_SYSTEM_PROCESSOR}/lib
        ${CMAKE_SYSTEM_PROCESSOR}/lib
        ${CMAKE_SYSTEM_PROCESSOR}/lib/tau
        ${CMAKE_SYSTEM_PROCESSOR}/lib64
        ${CMAKE_SYSTEM_PROCESSOR}/lib64/tau
    DOC
        "Path to the TAU library")

#----------------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set TAU_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(TAU DEFAULT_MSG TAU_INCLUDE_DIR TAU_LIBRARY)

#----------------------------------------------------------------------------------------#

if(TAU_FOUND)
    add_library(TAU INTERFACE)
    target_link_libraries(TAU INTERFACE ${TAU_LIBRARY})
    target_include_directories(TAU INTERFACE ${TAU_INCLUDE_DIR})
    get_filename_component(TAU_INCLUDE_DIRS ${TAU_INCLUDE_DIR} REALPATH)
    get_filename_component(TAU_LIBRARIES ${TAU_LIBRARY} REALPATH)
endif()

mark_as_advanced(TAU_INCLUDE_DIR TAU_LIBRARY)
