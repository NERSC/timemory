# ------------------------------------------------------------------------------#
#
# Finds headers and libraries for JuliaCxxWrap library
#
# ------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)

find_program(
    Julia_EXECUTABLE
    NAMES julia
    PATH_SUFFIXES bin)

set(JuliaCxxWrap_VARS)
if(NOT JuliaCxxWrap_ROOT_DIR AND Julia_EXECUTABLE)
    execute_process(
        COMMAND ${Julia_EXECUTABLE} -E "using CxxWrap; CxxWrap.prefix_path()"
        OUTPUT_VARIABLE JULIA_CXX_WRAP_ROOT_DIR
        RESULT_VARIABLE RET
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(RET EQUAL 0)
        string(REGEX REPLACE "(^\\\"|\\\"$)" "" JULIA_CXX_WRAP_ROOT_DIR
                             "${JULIA_CXX_WRAP_ROOT_DIR}")
        set(JuliaCxxWrap_ROOT_DIR
            "${JULIA_CXX_WRAP_ROOT_DIR}"
            CACHE PATH "Path to root folder of JuliaCxxWrap")
        set(JuliaCxxWrap_VARS JuliaCxxWrap_ROOT_DIR)
    endif()
elseif(JuliaCxxWrap_ROOT_DIR)
    set(JuliaCxxWrap_VARS JuliaCxxWrap_ROOT_DIR)
endif()

find_path(
    JuliaCxxWrap_INCLUDE_DIR
    NAMES jlcxx/jlcxx.hpp
    PATH_SUFFIXES include
    HINTS ${JuliaCxxWrap_ROOT_DIR}
    PATHS ${JuliaCxxWrap_ROOT_DIR})

find_library(
    JuliaCxxWrap_LIBRARY
    NAMES cxxwrap_julia
    PATH_SUFFIXES lib lib64 lib 64 lib/64
    HINTS ${JuliaCxxWrap_ROOT_DIR}
    PATHS ${JuliaCxxWrap_ROOT_DIR})

unset(FIND_PACKAGE_MESSAGE_DETAILS_JuliaCxxWrap CACHE)
mark_as_advanced(JuliaCxxWrap_INCLUDE_DIR JuliaCxxWrap_LIBRARY)
find_package_handle_standard_args(
    JuliaCxxWrap REQUIRED_VARS ${JuliaCxxWrap_VARS} JuliaCxxWrap_INCLUDE_DIR
                               JuliaCxxWrap_LIBRARY)

if(JuliaCxxWrap_FOUND)
    set(JuliaCxxWrap_INCLUDE_DIRS ${JuliaCxxWrap_INCLUDE_DIR})
    set(JuliaCxxWrap_LIBRARIES ${JuliaCxxWrap_LIBRARY})
    add_library(JuliaCxxWrap::JuliaCxxWrap IMPORTED INTERFACE)
    target_include_directories(JuliaCxxWrap::JuliaCxxWrap SYSTEM
                               INTERFACE ${JuliaCxxWrap_INCLUDE_DIR})
    target_link_libraries(JuliaCxxWrap::JuliaCxxWrap INTERFACE ${JuliaCxxWrap_LIBRARY})
endif()

unset(JuliaCxxWrap_VARS)
