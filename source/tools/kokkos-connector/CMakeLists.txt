cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_BUILD_KOKKOS_TOOLS)
    return()
endif()

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(STATUS "Error! Building from the source directory may overwrite Makefile")
    message(STATUS "Remove 'CMakeCache.txt' and 'CMakeFiles' and build in separate directory")
    message(FATAL_ERROR "In-source build")
endif()

project(timemory-kokkos-connector LANGUAGES C CXX)

# if built in kokkos-tools or in timemory
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME kokkos)
set(TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/timemory/kokkos-tools)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kokkos-tools)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kokkos-tools)

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakeParseArguments)

set(LINKER_LANGUAGE CXX)
set(CUDA_AVAILABLE OFF)
if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(LINKER_LANGUAGE CUDA)
    set(CUDA_AVAILABLE ON)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build flags" FORCE)
endif()

if(NOT CMAKE_VERSION VERSION_LESS 3.18)
    cmake_policy(SET CMP0104 OLD)
endif()

set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ language standard")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "CUDA language standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "C++ language flags required")
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "CUDA language flags required")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

add_option(TIMEMORY_KOKKOS_BUILD_SAMPLE "Build sample test" OFF NO_FEATURE)

if(NOT TIMEMORY_USE_CUDA)
    set(LINKER_LANGUAGE CXX)
endif()

# CMake INTERFACE target for timemory that provides include path
set(COMPONENTS headers vector dmp)

if(TARGET timemory-cxx-shared)
    list(APPEND COMPONENTS cxx-shared)
endif()

message(STATUS "")
set(TIMEMORY_KOKKOS_COMPONENTS "" CACHE STRING
    "Explicit list of timemory components")

# configure libraries
set(timemory_INTERFACE_LIBRARY)
foreach(_COMP ${COMPONENTS} ${TIMEMORY_KOKKOS_COMPONENTS})
    list(APPEND timemory_INTERFACE_LIBRARY timemory::timemory-${_COMP})
endforeach()

add_library(kp_timemory SHARED ${PROJECT_SOURCE_DIR}/kp_timemory.cpp
    ${PROJECT_SOURCE_DIR}/kp_timemory.hpp)
target_link_libraries(kp_timemory ${timemory_INTERFACE_LIBRARY})
set_target_properties(kp_timemory PROPERTIES
    PREFIX          ""
    LINKER_LANGUAGE ${LINKER_LANGUAGE})

install(TARGETS kp_timemory DESTINATION ${TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR})

add_library(kp_timemory_filter SHARED ${PROJECT_SOURCE_DIR}/kp_timemory_filter.cpp)
target_link_libraries(kp_timemory_filter ${timemory_INTERFACE_LIBRARY})
set_target_properties(kp_timemory_filter PROPERTIES
    PREFIX          ""
    LINKER_LANGUAGE ${LINKER_LANGUAGE})

install(TARGETS kp_timemory_filter DESTINATION ${TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR})

#
#   Build the sample
#
if(TIMEMORY_KOKKOS_BUILD_SAMPLE)
    # create test executable
    add_executable(sample sample/sample.cpp)
    target_link_libraries(sample kp_timemory)
endif()

#
#   Build the configurations
#
# general options
set(TIMEMORY_BUILD_KOKKOS_CONNECTORS "all" CACHE STRING
    "List of connector configurations to build")
add_option(TIMEMORY_BUILD_KOKKOS_CONFIG "Build various connector configurations" OFF)
add_feature(TIMEMORY_BUILD_KOKKOS_CONNECTORS "List of connector configurations to build" DOC)
if(DEFINED TIMEMORY_KOKKOS_BUILD_CONFIG)
    set(TIMEMORY_BUILD_KOKKOS_CONFIG ${TIMEMORY_KOKKOS_BUILD_CONFIG})
endif()
if(DEFINED TIMEMORY_KOKKOS_BUILD_CONNECTORS)
    set(TIMEMORY_BUILD_KOKKOS_CONNECTORS ${TIMEMORY_KOKKOS_BUILD_CONNECTORS})
endif()
string(TOUPPER "${TIMEMORY_BUILD_KOKKOS_CONNECTORS}" _CONNECTORS)

##--------------------------------------------------------------------------------------##

if(NOT TIMEMORY_BUILD_KOKKOS_CONFIG)
    message(STATUS "")
    message(STATUS "[${PROJECT_NAME}] Disabled building explicit configurations with 'TIMEMORY_BUILD_KOKKOS_CONFIG=OFF'")
    message(STATUS "")
    return()
endif()

##--------------------------------------------------------------------------------------##

message(STATUS "")
message(STATUS "Building explicit connector configurations: ${_CONNECTORS}")
message(STATUS "    Disable building explicit configurations with '-DTIMEMORY_BUILD_KOKKOS_CONFIG=OFF'")
message(STATUS "    Reduce explicit configurations with semi-colon delimited '-DTIMEMORY_BUILD_KOKKOS_CONNECTORS=\"...\"'")
message(STATUS "")
message(STATUS "Generating a custom configuration:")
message(STATUS "    '-DTIMEMORY_KOKKOS_USER_CONNECTOR_CONFIG=\"...\"' --> name for the configuration")
message(STATUS "    '-DTIMEMORY_KOKKOS_USER_CONNECTOR_TYPES=\"...\"'  --> semi-colon delimited list of components")
message(STATUS "")

##--------------------------------------------------------------------------------------##

function(ADD_CONNECTOR_CONFIG _NAME)
    cmake_parse_arguments(CONNECTOR "GPU" "" "COMPONENTS;TARGETS;DEFINITIONS" ${ARGN})
    if(NOT TIMEMORY_BUILD_KOKKOS_CONFIG)
        return()
    endif()

    if(NOT "${_NAME}" IN_LIST _CONNECTORS AND NOT "${_CONNECTORS}" STREQUAL "ALL")
        return()
    endif()

    if("${CONNECTOR_COMPONENTS}" STREQUAL "")
        return()
    endif()

    if(CONNECTOR_GPU AND NOT CUDA_AVAILABLE)
        return()
    endif()

    # configure the extension
    set(_EXT cpp)
    if(CONNECTOR_GPU AND CUDA_AVAILABLE)
        set(_EXT cu)
    endif()

    # configure the components
    string(REPLACE ";" "," _COMPONENTS "${CONNECTOR_COMPONENTS}")
    message(STATUS "Building 'kp_timemory_${_NAME}' with '${_COMPONENTS}'...")
    set(KP_COMPONENTS ${_COMPONENTS})

    set(SOURCE_FILE ${PROJECT_BINARY_DIR}/kp_timemory_${_NAME}.${_EXT})
    configure_file(${PROJECT_SOURCE_DIR}/kp_timemory.cpp.in ${SOURCE_FILE} @ONLY)

    add_library(kp_timemory_${_NAME} SHARED
        ${SOURCE_FILE} ${PROJECT_SOURCE_DIR}/kp_timemory.hpp)
    target_include_directories(kp_timemory_${_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR})
    target_link_libraries(kp_timemory_${_NAME} PRIVATE
        timemory::timemory-headers
        ${CONNECTOR_TARGETS})
    target_compile_definitions(kp_timemory_${_NAME} PRIVATE
        ${CONNECTOR_DEFINITIONS})
    set_target_properties(kp_timemory_${_NAME} PROPERTIES
        PREFIX ""
        LINKER_LANGUAGE ${LINKER_LANGUAGE})
    install(TARGETS kp_timemory_${_NAME}
        DESTINATION ${TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR})
endfunction()

##--------------------------------------------------------------------------------------##

add_connector_config(trip_count
    COMPONENTS trip_count
    TARGETS timemory::timemory-headers)

add_connector_config(timers
    COMPONENTS wall_clock cpu_clock cpu_util
    TARGETS timemory::timemory-headers)

add_connector_config(memory
    COMPONENTS peak_rss page_rss virtual_memory
    TARGETS timemory::timemory-headers)

add_connector_config(timers_memory
    COMPONENTS wall_clock cpu_clock cpu_util peak_rss page_rss virtual_memory
    TARGETS timemory::timemory-headers)

add_connector_config(io
    COMPONENTS written_bytes read_bytes
    TARGETS timemory::timemory-headers)

add_connector_config(context_switch
    COMPONENTS priority_context_switch voluntary_context_switch
    TARGETS timemory::timemory-headers)

if(TIMEMORY_USE_GPERFTOOLS)
    add_connector_config(gperftools
        COMPONENTS gperftools
        TARGETS timemory::timemory-gperftools)
endif()

if(TIMEMORY_USE_PAPI)
    add_connector_config(papi
        COMPONENTS papi_array_t
        TARGETS timemory::timemory-papi)
    add_connector_config(cpu_roofline
        COMPONENTS cpu_roofline_flops
        TARGETS timemory::timemory-cpu-roofline)
    add_connector_config(cpu_flops
        COMPONENTS papi_tuple<PAPI_DP_OPS,PAPI_SP_OPS>
        TARGETS timemory::timemory-papi)
    add_connector_config(cpu_instructions
        COMPONENTS papi_tuple<PAPI_TOT_INS,PAPI_LD_INS,PAPI_SR_INS>
        TARGETS timemory::timemory-papi)
endif()

if(TIMEMORY_USE_CUDA)
    add_connector_config(cuda_profiler
        COMPONENTS cuda_profiler
        TARGETS timemory::timemory-cuda)
    add_connector_config(cuda_event
        COMPONENTS cuda_event
        TARGETS timemory::timemory-cuda)
    add_connector_config(cuda_nvtx
        COMPONENTS nvtx_marker
        TARGETS timemory::timemory-cupti timemory::timemory-cuda timemory::timemory-nvtx)
    add_connector_config(cuda_activity
        COMPONENTS cupti_activity
        TARGETS timemory::timemory-cupti timemory::timemory-cuda)
    add_connector_config(cuda_hw_counters
        COMPONENTS cupti_counters
        TARGETS timemory::timemory-cupti timemory::timemory-cuda)
    add_connector_config(gpu_roofline
        GPU
        COMPONENTS gpu_roofline_flops
        TARGETS timemory::timemory-gpu-roofline)
endif()

if(TIMEMORY_USE_VTUNE)
    add_connector_config(vtune_profiler
        COMPONENTS vtune_profiler
        TARGETS timemory::timemory-vtune)
    add_connector_config(vtune_frame
        COMPONENTS vtune_frame
        TARGETS timemory::timemory-vtune)
    add_connector_config(vtune_event
        COMPONENTS vtune_event
        TARGETS timemory::timemory-vtune)
endif()

if(TIMEMORY_USE_CALIPER)
    add_connector_config(caliper
        COMPONENTS caliper
        TARGETS timemory::timemory-caliper)
endif()

if(TIMEMORY_USE_LIKWID)
    add_connector_config(likwid
        COMPONENTS likwid_marker likwid_nvmarker
        TARGETS timemory::timemory-likwid)
endif()

if(TIMEMORY_USE_TAU)
    add_connector_config(tau
        COMPONENTS tau_marker
        TARGETS timemory::timemory-tau)
endif()

set(_OMPT_LIB)
if(TARGET timemory::timemory-ompt-shared)
    set(_OMPT_LIB timemory::timemory-ompt-shared)
elseif(TARGET timemory::timemory-ompt-static)
    set(_OMPT_LIB timemory::timemory-ompt-static)
endif()

if(TIMEMORY_USE_OMPT AND _OMPT_LIB)
    add_connector_config(ompt
        COMPONENTS ompt_handle<tim::api::native_tag>
        DEFINITIONS TIMEMORY_USE_OMPT_LIBRARY
        TARGETS ${_OMPT_LIB})
endif()

if(USER_CONNECTOR_CONFIG AND USER_CONNECTOR_TYPES)
    string(TOLOWER "${USER_CONNECTOR_CONFIG}" _TAG)
    string(TOLOWER "${USER_CONNECTOR_TYPES}" _TYPES)
    add_connector_config(${_TAG}
        COMPONENTS ${_TYPES}
        TARGETS ${timemory_INTERFACE_LIBRARY})
endif()

message(STATUS "")
