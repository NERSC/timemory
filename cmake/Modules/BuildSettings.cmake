# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Handles the build settings
#
##########################################################################################


include(GNUInstallDirs)
include(Compilers)


find_library(dl_LIBRARY NAMES dl)
if(dl_LIBRARY)
    target_link_libraries(timemory-compile-options INTERFACE ${dl_LIBRARY})
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
    "-W" "${OS_FLAG}" "-Wno-unknown-pragmas" "-Wno-ignored-attributes" "-Wno-attributes"
    "-Wno-mismatched-tags" "-Wno-missing-field-initializers")

if(CMAKE_CXX_COMPILER_IS_GNU)
    add_cxx_flag_if_avail("-Wno-class-memaccess")
    add_cxx_flag_if_avail("-Wno-cast-function-type")
endif()

if(TIMEMORY_BUILD_QUIET)
    add_flag_if_avail("-Wno-unused-value" "-Wno-unused-function"
        "-Wno-unknown-pragmas" "-Wno-deprecated-declarations" "-Wno-implicit-fallthrough"
        "-Wno-unused-command-line-argument"
        )
endif()

if(NOT CMAKE_CXX_COMPILER_IS_GNU)
    # these flags succeed with GNU compiler but are unknown (clang flags)
    # add_cxx_flag_if_avail("-Wno-exceptions")
    # add_cxx_flag_if_avail("-Wno-unused-private-field")
else()
    # add_cxx_flag_if_avail("-Wno-class-memaccess")
endif()

#----------------------------------------------------------------------------------------#
# non-debug optimizations
#
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS)
    add_flag_if_avail("-finline-functions" "-funroll-loops"
        "-ftree-vectorize" "-ftree-loop-optimize" "-ftree-loop-vectorize")
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
        set(TIMEMORY_BUILD_LTO OFF)
        add_disabled_interface(timemory-compile-timing)
    endif()
endif()

if(TIMEMORY_BUILD_LTO)
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
# developer build flags
#
add_interface_library(timemory-develop-options "Adds developer compiler flags")
if(TIMEMORY_BUILD_DEVELOPER)
    add_target_flag_if_avail(timemory-develop-options
        "-Wshadow" "-Wextra" "-Wpedantic" "-Werror")
endif()

#----------------------------------------------------------------------------------------#
# visibility build flags
#
add_interface_library(timemory-default-visibility
    "Adds -fvisibility=default compiler flag")
add_interface_library(timemory-protected-visibility
    "Adds -fvisibility=protected compiler flag")
add_interface_library(timemory-hidden-visibility
    "Adds -fvisibility=hidden compiler flag")

add_target_flag_if_avail(timemory-default-visibility "-fvisibility=default")
add_target_flag_if_avail(timemory-protected-visibility "-fvisibility=protected")
add_target_flag_if_avail(timemory-hidden-visibility "-fvisibility=hidden")

foreach(_TYPE default protected hidden)
    if(NOT cxx_timemory_${_TYPE}_visibility_fvisibility_${_TYPE})
        add_disabled_interface(timemory-${_TYPE}-visibility)
    else()
        target_compile_definitions(timemory-${_TYPE}-visibility INTERFACE TIMEMORY_USE_VISIBILITY)
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

foreach(_TYPE ${SANITIZER_TYPES})
    add_interface_library(timemory-${_TYPE}-sanitizer
        "Adds compiler flags to enable ${_TYPE} sanitizer (-fsanitizer=${_TYPE})")
    set(_FLAGS "-fno-optimize-sibling-calls" "-fno-omit-frame-pointer"
        "-fno-inline-functions" "-fsanitize=${_TYPE}")
    add_target_flag(timemory-${_TYPE}-sanitizer ${_FLAGS})
    set_property(TARGET timemory-${_TYPE}-sanitizer PROPERTY INTERFACE_LINK_OPTIONS ${_FLAGS})
endforeach()

add_interface_library(timemory-sanitizer
    "Adds compiler flags to enable ${SANITIZER_TYPE} sanitizer (-fsanitizer=${SANITIZER_TYPE})")
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
