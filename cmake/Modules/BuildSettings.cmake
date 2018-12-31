################################################################################
#
#        Handles the build settings
#
################################################################################

include(GNUInstallDirs)
include(Compilers)


# ---------------------------------------------------------------------------- #
#
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_C_STANDARD 11 CACHE STRING "C language standard")
set(CMAKE_CXX_STANDARD 14 CACHE STRING "CXX language standard")
set(CMAKE_C_STANDARD_REQUIRED ON CACHE BOOL "Require the C language standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require the CXX language standard")


# ---------------------------------------------------------------------------- #
# set the output directory (critical on Windows)
#
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    # if ${PROJECT_NAME}_OUTPUT_DIR is not defined, set to CMAKE_BINARY_DIR
    if(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
        set(${PROJECT_NAME}_OUTPUT_DIR ${CMAKE_BINARY_DIR})
    endif(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
    # set the CMAKE_{ARCHIVE,LIBRARY,RUNTIME}_OUTPUT_DIRECTORY variables
    if(WIN32)
        # on Windows, separate types into different directories
        string(TOLOWER "${_TYPE}" _LTYPE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR}/outputs/${_LTYPE})
    else(WIN32)
        # on UNIX, just set to same directory
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR})
    endif(WIN32)
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)


# ---------------------------------------------------------------------------- #
#  debug macro
#
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DDEBUG)
else("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DNDEBUG)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")


# ---------------------------------------------------------------------------- #
# used by configure_package_*
set(LIBNAME timemory)


# ---------------------------------------------------------------------------- #
# set the compiler flags
add_c_flag_if_avail("-W")
add_c_flag_if_avail("-Wall")
add_c_flag_if_avail("-Wextra")
add_c_flag_if_avail("-Wno-unused-variable")
add_c_flag_if_avail("-Wno-unknown-pragmas")
add_c_flag_if_avail("-Wno-unused-parameter")
add_c_flag_if_avail("-Wno-reserved-id-macro")
add_c_flag_if_avail("-Wunused-but-set-parameter")

# SIMD OpenMP
# add_c_flag_if_avail("-fopenmp-simd")
# add_cxx_flag_if_avail("-fopenmp-simd")

# Intel floating-point model
add_c_flag_if_avail("-fp-model=precise")
add_cxx_flag_if_avail("-fp-model=precise")

add_cxx_flag_if_avail("-W")
add_cxx_flag_if_avail("-Wall")
add_cxx_flag_if_avail("-Wextra")
add_cxx_flag_if_avail("-Wno-unused-value")
add_cxx_flag_if_avail("-Wno-unused-variable")
add_cxx_flag_if_avail("-Wno-unknown-pragmas")
add_cxx_flag_if_avail("-Wno-unknown-pragmas")
add_cxx_flag_if_avail("-Wno-unused-parameter")
add_cxx_flag_if_avail("-Wno-implicit-fallthrough")
add_cxx_flag_if_avail("-Wunused-but-set-parameter")
add_cxx_flag_if_avail("-faligned-new")

if(NOT CMAKE_CXX_COMPILER_IS_GNU)
    # these flags succeed with GNU compiler but are unknown (clang flags)
    add_cxx_flag_if_avail("-Wno-exceptions")
    add_cxx_flag_if_avail("-Wno-reserved-id-macro")
    add_cxx_flag_if_avail("-Wno-unused-private-field")
endif()

if(TIMEMORY_USE_ARCH)
    if(CMAKE_C_COMPILER_IS_INTEL)
        add_c_flag_if_avail("-xHOST")
        if(TIMEMORY_USE_AVX512)
            add_c_flag_if_avail("-axMIC-AVX512")
        endif()
    else()
        add_c_flag_if_avail("-march")
        add_c_flag_if_avail("-msse2")
        add_c_flag_if_avail("-msse3")
        add_c_flag_if_avail("-msse4")
        add_c_flag_if_avail("-mavx")
        add_c_flag_if_avail("-mavx2")
        if(TIMEMORY_USE_AVX512)
            add_c_flag_if_avail("-mavx512f")
            add_c_flag_if_avail("-mavx512pf")
            add_c_flag_if_avail("-mavx512er")
            add_c_flag_if_avail("-mavx512cd")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_IS_INTEL)
        add_cxx_flag_if_avail("-xHOST")
        if(TIMEMORY_USE_AVX512)
            add_cxx_flag_if_avail("-axMIC-AVX512")
        endif()
    else()
        add_cxx_flag_if_avail("-march")
        add_cxx_flag_if_avail("-msse2")
        add_cxx_flag_if_avail("-msse3")
        add_cxx_flag_if_avail("-msse4")
        add_cxx_flag_if_avail("-mavx")
        add_cxx_flag_if_avail("-mavx2")
        if(TIMEMORY_USE_AVX512)
            add_cxx_flag_if_avail("-mavx512f")
            add_cxx_flag_if_avail("-mavx512pf")
            add_cxx_flag_if_avail("-mavx512er")
            add_cxx_flag_if_avail("-mavx512cd")
        endif()
    endif()
endif()


# ---------------------------------------------------------------------------- #
# user customization
add(${PROJECT_NAME}_C_FLAGS "${CFLAGS}")
add(${PROJECT_NAME}_C_FLAGS "$ENV{CFLAGS}")
add(${PROJECT_NAME}_CXX_FLAGS "${CXXFLAGS}")
add(${PROJECT_NAME}_CXX_FLAGS "$ENV{CXXFLAGS}")

if(TIMEMORY_EXCEPTIONS)
    add_definitions(-DTIMEMORY_EXCEPTIONS)
endif(TIMEMORY_EXCEPTIONS)
