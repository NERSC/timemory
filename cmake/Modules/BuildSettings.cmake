# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Handles the build settings
#
##########################################################################################


include(GNUInstallDirs)
include(Compilers)


if(CMAKE_DL_LIBS)
    set(dl_LIBRARY ${CMAKE_DL_LIBS})
    target_link_libraries(timemory-compile-options INTERFACE ${CMAKE_DL_LIBS})
else()
    find_library(dl_LIBRARY NAMES dl)
    if(dl_LIBRARY)
        target_link_libraries(timemory-compile-options INTERFACE ${dl_LIBRARY})
    endif()
endif()

if(WIN32)
    set(OS_FLAG "/bigobj")
else()
    set(OS_FLAG "-Wall")
endif()

#----------------------------------------------------------------------------------------#
# set the compiler flags
#
add_flag_if_avail(
    "-W"
    "${OS_FLAG}"
    "-Wno-unknown-pragmas"
    "-Wno-ignored-attributes"
    "-Wno-attributes"
    "-Wno-missing-field-initializers")

add_cxx_flag_if_avail(
    "-Wno-mismatched-tags")

if(CMAKE_CXX_COMPILER_IS_GNU)
    add_target_cxx_flag_if_avail(
        timemory-compile-options
        "-Wno-class-memaccess"
        "-Wno-cast-function-type")
endif()

if(TIMEMORY_BUILD_QUIET)
    add_flag_if_avail(
        "-Wno-unused-value"
        "-Wno-unused-function"
        "-Wno-unknown-pragmas"
        "-Wno-deprecated-declarations"
        "-Wno-implicit-fallthrough"
        "-Wno-unused-command-line-argument")
endif()

#----------------------------------------------------------------------------------------#
# non-debug optimizations
#
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS)
    add_flag_if_avail(
        "-finline-functions"
        "-funroll-loops"
        "-ftree-vectorize"
        "-ftree-loop-optimize"
        "-ftree-loop-vectorize"
        "-fno-signaling-nans"
        "-fno-trapping-math"
        "-fno-signed-zeros"
        "-ffinite-math-only"
        "-fno-math-errno"
        "-fpredictive-commoning"
        "-fvariable-expansion-in-unroller")
    # add_flag_if_avail("-freciprocal-math" "-fno-signed-zeros" "-mfast-fp")
endif()

#----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
add_cxx_flag_if_avail("-faligned-new")
if(NOT TIMEMORY_USE_SANITIZER)
    add_cxx_flag_if_avail("-ftls-model=${TIMEMORY_TLS_MODEL}")
endif()

add_interface_library(timemory-lto "Adds link-time-optimization flags")
add_target_flag_if_avail(timemory-lto "-flto=thin")
if(NOT cxx_timemory_lto_flto_thin)
    add_target_flag_if_avail(timemory-lto "-flto")
    if(NOT cxx_timemory_lto_flto)
        add_disabled_interface(timemory-compile-timing)
    else()
        set_target_properties(timemory-lto PROPERTIES
            INTERFACE_LINK_OPTIONS -flto)
    endif()
else()
    set_target_properties(timemory-lto PROPERTIES
        INTERFACE_LINK_OPTIONS -flto=thin)
endif()

if(TIMEMORY_BUILD_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    target_link_libraries(timemory-compile-options INTERFACE timemory-lto)
endif()

#----------------------------------------------------------------------------------------#
# print compilation timing reports (Clang compiler)
#
add_interface_library(timemory-compile-timing
    "Adds compiler flags which report compilation timing metrics")
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_target_flag_if_avail(timemory-compile-timing "-ftime-trace")
    if(NOT cxx_timemory_compile_timing_ftime_trace)
        add_target_flag_if_avail(timemory-compile-timing "-ftime-report")
    endif()
else()
    add_target_flag_if_avail(timemory-compile-timing "-ftime-report")
endif()

if(TIMEMORY_USE_COMPILE_TIMING)
    target_link_libraries(timemory-compile-options INTERFACE timemory-compile-timing)
endif()

if(NOT cxx_timemory_compile_timing_ftime_report AND NOT cxx_timemory_compile_timing_ftime_trace)
    add_disabled_interface(timemory-compile-timing)
endif()

#----------------------------------------------------------------------------------------#
# use xray instrumentation
#
add_interface_library(timemory-xray
    "Adds compiler flags to enable xray-instrumentation (Clang only)")
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_target_flag_if_avail(timemory-xray "-fxray-instrument" "-fxray-instruction-threshold=1")
    if(NOT cxx_timemory_xray_fxray_instrument)
        add_disabled_interface(timemory-xray)
    endif()
else()
    add_disabled_interface(timemory-xray)
endif()

#----------------------------------------------------------------------------------------#
# use built-in instrumentation
#
add_interface_library(timemory-instrument-functions
    "Adds compiler flags to enable compile-time instrumentation")
add_target_flag_if_avail(timemory-instrument-functions "-finstrument-functions")
if(NOT cxx_timemory_instrument_finstrument_functions)
    add_disabled_interface(timemory-instrument-functions)
endif()

#----------------------------------------------------------------------------------------#
# developer build flags
#
add_interface_library(timemory-develop-options "Adds developer compiler flags")
if(TIMEMORY_BUILD_DEVELOPER)
    add_target_flag_if_avail(timemory-develop-options
        # "-Wabi"
        "-Wdouble-promotion"
        "-Wshadow"
        "-Wextra"
        "-Wpedantic"
        "-Werror"
        "/showIncludes")
endif()

#----------------------------------------------------------------------------------------#
# visibility build flags
#
add_interface_library(timemory-default-visibility
    "Adds -fvisibility=default compiler flag")
add_interface_library(timemory-hidden-visibility
    "Adds -fvisibility=hidden compiler flag")

add_target_flag_if_avail(timemory-default-visibility "-fvisibility=default")
add_target_flag_if_avail(timemory-hidden-visibility "-fvisibility=hidden")

foreach(_TYPE default hidden)
    if(NOT cxx_timemory_${_TYPE}_visibility_fvisibility_${_TYPE})
        add_disabled_interface(timemory-${_TYPE}-visibility)
    else()
        timemory_target_compile_definitions(timemory-${_TYPE}-visibility INTERFACE
            TIMEMORY_USE_VISIBILITY)
    endif()
endforeach()

#----------------------------------------------------------------------------------------#
# developer build flags
#
if(dl_LIBRARY)
    # This instructs the linker to add all symbols, not only used ones, to the dynamic
    # symbol table. This option is needed for some uses of dlopen or to allow obtaining
    # backtraces from within a program.
    if(NOT CMAKE_CXX_COMPILER_IS_CLANG AND APPLE)
        add_flag_if_avail("-rdynamic")
    endif()
endif()

#----------------------------------------------------------------------------------------#
# architecture optimizations
#
add_interface_library(timemory-vector
    "Adds pre-processor definition of the max vectorization width in bytes")
add_interface_library(timemory-arch
    "Adds architecture-specific compiler flags")
target_link_libraries(timemory-compile-options INTERFACE timemory-vector)

set(VECTOR_DEFINITION               TIMEMORY_VEC)
set(VECTOR_INTERFACE_TARGET         timemory-vector)
set(ARCH_INTERFACE_TARGET           timemory-arch)

include(ArchConfig)

add_cmake_defines(TIMEMORY_VEC VALUE)

#----------------------------------------------------------------------------------------#
# sanitizer
#
set(SANITIZER_TYPES address memory thread leak undefined unreachable null bounds alignment)
set_property(CACHE SANITIZER_TYPE PROPERTY STRINGS "${SANITIZER_TYPES}")
add_interface_library(timemory-sanitizer-compile-options "Adds compiler flags for sanitizers")
add_interface_library(timemory-sanitizer
    "Adds compiler flags to enable ${SANITIZER_TYPE} sanitizer (-fsanitizer=${SANITIZER_TYPE})")

set(COMMON_SANITIZER_FLAGS "-fno-optimize-sibling-calls" "-fno-omit-frame-pointer" "-fno-inline-functions")
add_target_flag(timemory-sanitizer-compile-options ${COMMON_SANITIZER_FLAGS})

foreach(_TYPE ${SANITIZER_TYPES})
    set(_FLAG "-fsanitize=${_TYPE}")
    add_interface_library(timemory-${_TYPE}-sanitizer
        "Adds compiler flags to enable ${_TYPE} sanitizer (${_FLAG})")
    add_target_flag(timemory-${_TYPE}-sanitizer ${_FLAG})
    target_link_libraries(timemory-${_TYPE}-sanitizer INTERFACE
        timemory-sanitizer-compile-options)
    set_property(TARGET timemory-${_TYPE}-sanitizer PROPERTY
        INTERFACE_LINK_OPTIONS ${_FLAG} ${COMMON_SANITIZER_FLAGS})
endforeach()

unset(_FLAG)
unset(COMMON_SANITIZER_FLAGS)

if(TIMEMORY_USE_SANITIZER)
    foreach(_TYPE ${SANITIZER_TYPE})
        if(TARGET timemory-${_TYPE}-sanitizer)
            target_link_libraries(timemory-sanitizer INTERFACE timemory-${_TYPE}-sanitizer)
        else()
            message(FATAL_ERROR "Error! Target 'timemory-${_TYPE}-sanitizer' does not exist!")
        endif()
    endforeach()
else()
    set(TIMEMORY_USE_SANITIZER OFF)
    inform_empty_interface(timemory-sanitizer "${SANITIZER_TYPE} sanitizer")
endif()


#----------------------------------------------------------------------------------------#
# user customization
#
add_user_flags(timemory-compile-options "C")
add_user_flags(timemory-compile-options "CXX")
add_user_flags(timemory-compile-options "CUDA")
