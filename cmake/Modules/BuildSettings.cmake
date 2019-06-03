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
# standard
set(CMAKE_C_STANDARD 11 CACHE STRING "C language standard")
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX language standard")
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "CUDA language standard")
# standard required
set(CMAKE_C_STANDARD_REQUIRED ON CACHE BOOL "Require the C language standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require the CXX language standard")
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "Require the CUDA language standard")
# extensions
set(CMAKE_C_EXTENSIONS OFF CACHE BOOL "C language extensions")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "CXX language extensions")
set(CMAKE_CUDA_EXTENSIONS OFF CACHE BOOL "CUDA language extensions")

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
    elseif(XCODE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
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
# set the compiler flags
add_c_flag_if_avail("-W")
if(NOT WIN32)
    add_c_flag_if_avail("-Wall")
else()
    add_c_flag_if_avail("/bigobj")
endif()
add_c_flag_if_avail("-Wextra")
add_c_flag_if_avail("-Wshadow")
add_c_flag_if_avail("-Wno-unused-value")
add_c_flag_if_avail("-Wno-unused-function")
add_c_flag_if_avail("-Wno-unknown-pragmas")
add_c_flag_if_avail("-Wno-reserved-id-macro")
add_c_flag_if_avail("-Wno-deprecated-declarations")

add_cxx_flag_if_avail("-W")
if(NOT WIN32)
    add_cxx_flag_if_avail("-Wall")
else()
    add_cxx_flag_if_avail("/bigobj")
endif()
add_cxx_flag_if_avail("-Wextra")
add_cxx_flag_if_avail("-Wshadow")
add_cxx_flag_if_avail("-Wno-unused-value")
add_cxx_flag_if_avail("-Wno-unused-function")
add_cxx_flag_if_avail("-Wno-unknown-pragmas")
add_cxx_flag_if_avail("-Wno-c++17-extensions")
add_cxx_flag_if_avail("-Wno-implicit-fallthrough")
add_cxx_flag_if_avail("-Wno-deprecated-declarations")
add_cxx_flag_if_avail("-ftemplate-backtrace-limit=0")

if(NOT CMAKE_CXX_COMPILER_IS_GNU)
    # these flags succeed with GNU compiler but are unknown (clang flags)
    add_cxx_flag_if_avail("-Wno-exceptions")
    add_cxx_flag_if_avail("-Wno-class-memaccess")
    add_cxx_flag_if_avail("-Wno-reserved-id-macro")
    add_cxx_flag_if_avail("-Wno-unused-private-field")
endif()

# ---------------------------------------------------------------------------- #
# non-debug optimizations
#
if(NOT DEBUG)
    add_c_flag_if_avail("-funroll-loops")
    add_c_flag_if_avail("-ftree-vectorize")
    add_c_flag_if_avail("-finline-functions")
    add_c_flag_if_avail("-ftree-loop-optimize")
    add_c_flag_if_avail("-ftree-loop-vectorize")
    # add_c_flag_if_avail("-fira-loop-pressure")

    add_cxx_flag_if_avail("-funroll-loops")
    add_cxx_flag_if_avail("-ftree-vectorize")
    add_cxx_flag_if_avail("-finline-functions")
    add_cxx_flag_if_avail("-ftree-loop-optimize")
    add_cxx_flag_if_avail("-ftree-loop-vectorize")
    # add_cxx_flag_if_avail("-fira-loop-pressure")
endif()

# ---------------------------------------------------------------------------- #
# Intel floating-point model (implies -fprotect-parens)
#
# add_c_flag_if_avail("-fp-model=precise")
# add_cxx_flag_if_avail("-fp-model=precise")

# ---------------------------------------------------------------------------- #
# debug-safe optimizations
#
add_cxx_flag_if_avail("-faligned-new")
add_cxx_flag_if_avail("-ftls-model=${TIMEMORY_TLS_MODEL}")


if(TIMEMORY_BUILD_LTO)
    add_c_flag_if_avail("-flto")
    add_cxx_flag_if_avail("-flto")
endif()

# ---------------------------------------------------------------------------- #
# architecture optimizations
#
add_interface_library(timemory-arch)
add_interface_library(timemory-avx512)

if(CMAKE_C_COMPILER_IS_INTEL)
    add_target_flag_if_avail(timemory-arch "-xHOST")
    add_target_flag_if_avail(timemory-avx512 "-axMIC-AVX512")
else()
    add_target_flag_if_avail(timemory-arch "-march=native" "-msse2" "-msse3" "-msse4" "-mavx" "-mavx2")
    add_target_flag_if_avail(timemory-avx512 "-mavx512f" "-mavx512pf" "-mavx512er" "-mavx512cd")
endif()

target_link_libraries(timemory-avx512 INTERFACE timemory-arch)

if(TIMEMORY_USE_ARCH)
    list(APPEND ${PROJECT_NAME}_TARGET_LIBRARIES timemory-arch)
    if(TIMEMORY_USE_AVX512)
        list(APPEND ${PROJECT_NAME}_TARGET_LIBRARIES timemory-avx512)
    endif()
endif()

# ---------------------------------------------------------------------------- #
# sanitizer
#
if(TIMEMORY_USE_SANITIZER)
    set(SANITIZER_TYPES address memory thread leak)

    set(asan_key "address")
    set(msan_key "memory")
    set(tsan_key "thread")
    set(lsan_key "leak")

    set(address_lib asan)
    set(memory_lib msan)
    set(thread_lib tsan)
    set(leak_lib lsan)

    find_library(SANITIZER_asan_LIBRARY NAMES asan)
    find_library(SANITIZER_msan_LIBRARY NAMES msan)
    find_library(SANITIZER_tsan_LIBRARY NAMES tsan)
    find_library(SANITIZER_lsan_LIBRARY NAMES lsan)

    string(TOLOWER "${SANITIZER_TYPE}" SANITIZER_TYPE)
    list(REMOVE_ITEM SANITIZER_TYPES ${SANITIZER_TYPE})
    set(SANITIZER_TYPES ${SANITIZER_TYPE} ${SANITIZER_TYPES})

    foreach(_TYPE ${SANITIZER_TYPES})
        set(_LIB ${${_TYPE}_lib})
        add_interface_library(timemory-${_TYPE}-sanitizer)
        add_target_flag_if_avail(timemory-${_TYPE}-sanitizer "-fsanitize=${SANITIZER_TYPE}")
        target_link_libraries(timemory-${_TYPE}-sanitizer INTERFACE ${SANITIZER_${_LIB}_LIBRARY})
    endforeach()

    foreach(_TYPE ${SANITIZER_TYPES})
        set(_LIB ${${_TYPE}_lib})
        if(c_timemory_${_TYPE}_sanitizer_fsanitize_${SANITIZER_TYPE} AND
                cxx_timemory_${_TYPE}_sanitizer_fsanitize_${SANITIZER_TYPE} AND
                SANITIZER_${_LIB}_LIBRARY)
            add_interface_library(timemory-sanitizer)
            add_target_flag_if_avail(timemory-sanitizer "-fno-omit-frame-pointer")
            target_compile_definitions(timemory-sanitizer INTERFACE TIMEMORY_USE_SANITIZER)
            target_link_libraries(timemory-sanitizer INTERFACE timemory-${_TYPE}-sanitizer)
            break()
        else()
            message(STATUS "${_TYPE} sanitizer not found. library: ${SANITIZER_${_LIB}_LIBRARY}...")
        endif()
    endforeach()

    if(NOT TARGET timemory-sanitizer)
        message(WARNING "TIMEMORY_USE_SANITIZER not found. Tried: ${SANITIZER_TYPES}")
        unset(SANITIZER_TYPE CACHE)
        set(TIMEMORY_USE_SANITIZER OFF)
    endif()

endif()


# ---------------------------------------------------------------------------- #
# user customization
#
set(_CFLAGS ${CFLAGS} $ENV{CFLAGS})
set(_CXXFLAGS ${CXXFLAGS} $ENV{CXXFLAGS})
string(REPLACE " " ";" _CFLAGS "${_CFLAGS}")
string(REPLACE " " ";" _CXXFLAGS "${_CXXFLAGS}")
list(APPEND ${PROJECT_NAME}_C_FLAGS ${_CFLAGS})
list(APPEND ${PROJECT_NAME}_CXX_FLAGS ${_CXXFLAGS})
