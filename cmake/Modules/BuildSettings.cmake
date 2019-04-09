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
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX language standard")
set(CMAKE_C_STANDARD_REQUIRED ON CACHE BOOL "Require the C language standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require the CXX language standard")


# ---------------------------------------------------------------------------- #
# set the output directory (critical on Windows)
#
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    # if ${PROJECT_NAME}_OUTPUT_DIR is not defined, set to CMAKE_BINARY_DIR
    if(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
        set(${PROJECT_NAME}_OUTPUT_DIR ${CMAKE_BINARY_DIR})
    endif()

    # set the CMAKE_{ARCHIVE,LIBRARY,RUNTIME}_OUTPUT_DIRECTORY variables
    if(WIN32)
        # on Windows, separate types into different directories
        string(TOLOWER "${_TYPE}" _LTYPE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR}/outputs/${_LTYPE})
    else()
        # on UNIX, just set to same directory
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR})
    endif()
endforeach()


# ---------------------------------------------------------------------------- #
#  debug macro
#
set(DEBUG OFF)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(DEBUG ON)
    list(APPEND ${PROJECT_NAME}_DEFINITIONS DEBUG)
else()
    list(APPEND ${PROJECT_NAME}_DEFINITIONS NDEBUG)
endif()


# ---------------------------------------------------------------------------- #
# used by configure_package_*
set(LIBNAME timemory)


# ---------------------------------------------------------------------------- #
# non-debug optimizations
if(NOT DEBUG)
    add_c_flag_if_avail("-funroll-loops")
    add_c_flag_if_avail("-ftree-vectorize")
    add_c_flag_if_avail("-finline-functions")
    add_c_flag_if_avail("-fira-loop-pressure")

    add_cxx_flag_if_avail("-funroll-loops")
    add_cxx_flag_if_avail("-ftree-vectorize")
    add_cxx_flag_if_avail("-finline-functions")
    add_cxx_flag_if_avail("-fira-loop-pressure")
endif()

# Intel floating-point model
add_c_flag_if_avail("-fp-model=precise")
add_cxx_flag_if_avail("-fp-model=precise")


# ---------------------------------------------------------------------------- #
# set the compiler flags
add_c_flag_if_avail("-W")
add_c_flag_if_avail("-Wall")
add_c_flag_if_avail("-Wextra")
add_c_flag_if_avail("-Wno-unused-value")
add_c_flag_if_avail("-Wno-unknown-pragmas")
add_c_flag_if_avail("-Wno-reserved-id-macro")
add_c_flag_if_avail("-Wno-deprecated-declarations")

add_cxx_flag_if_avail("-W")
add_cxx_flag_if_avail("-Wall")
add_cxx_flag_if_avail("-Wextra")
# add_cxx_flag_if_avail("-Wshadow")
add_cxx_flag_if_avail("-Wno-unused-value")
add_cxx_flag_if_avail("-Wno-class-memaccess")
add_cxx_flag_if_avail("-Wno-unknown-pragmas")
add_cxx_flag_if_avail("-Wno-c++17-extensions")
add_cxx_flag_if_avail("-Wno-implicit-fallthrough")
add_cxx_flag_if_avail("-Wno-deprecated-declarations")
add_cxx_flag_if_avail("-faligned-new")

# add_c_flag_if_avail("-flto")
# add_cxx_flag_if_avail("-flto")

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

if(TIMEMORY_USE_SANITIZER)
    add_c_flag_if_avail("-fsanitize=${SANITIZER_TYPE}")
    add_cxx_flag_if_avail("-fsanitize=${SANITIZER_TYPE}")

    if(c_fsanitize_${SANITIZER_TYPE} AND cxx_fsanitize_${SANITIZER_TYPE})
        if("${SANITIZER_TYPE}" STREQUAL "address")
            list(APPEND EXTERNAL_LIBRARIES asan)
        elseif("${SANITIZER_TYPE}" STREQUAL "memory")
            list(APPEND EXTERNAL_LIBRARIES msan)
        elseif("${SANITIZER_TYPE}" STREQUAL "thread")
            list(APPEND EXTERNAL_LIBRARUES tsan)
        elseif("${SANITIZER_TYPE}" STREQUAL "leak")
            list(APPEND EXTERNAL_LIBRARIES lsan)
        endif()
    else()
        unset(SANITIZER_TYPE CACHE)
        set(TIMEMORY_USE_SANITIZER OFF)
    endif()
endif()

# ---------------------------------------------------------------------------- #
# user customization
set(_CFLAGS ${CFLAGS} $ENV{CFLAGS})
set(_CXXFLAGS ${CXXFLAGS} $ENV{CXXFLAGS})
string(REPLACE " " ";" _CFLAGS "${_CFLAGS}")
string(REPLACE " " ";" _CXXFLAGS "${_CXXFLAGS}")
list(APPEND ${PROJECT_NAME}_C_FLAGS ${_CFLAGS})
list(APPEND ${PROJECT_NAME}_C_FLAGS ${_CXXFLAGS})
