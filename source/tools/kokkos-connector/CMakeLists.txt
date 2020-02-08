cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(STATUS "Error! Building from the source directory may overwrite Makefile")
    message(STATUS "Remove 'CMakeCache.txt' and 'CMakeFiles' and build in separate directory")
    message(FATAL_ERROR "In-source build")
endif()

project(timemory-kokkos-connector LANGUAGES C CXX)

# if built in kokkos-tools or in timemory
set(TIMEMORY_KOKKOS_TOOLS_IN_SOURCE OFF)
# external project (i.e. in kokkos-tools) references interface libraries via "timemory::timemory-XYZ"
# interal project (i.e. in timemory) references interface libraries via "timemory-XYZ"
set(TIMEMORY_KOKKOS_TOOLS_PREFIX "timemory::")
# the prefix for the options
set(TIMEMORY_KOKKOS_TOOLS_OPTION "")
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    set(TIMEMORY_KOKKOS_TOOLS_IN_SOURCE ON)
    set(TIMEMORY_KOKKOS_TOOLS_PREFIX "")
    set(TIMEMORY_KOKKOS_TOOLS_OPTION "TIMEMORY_KOKKOS_")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kokkos-tools)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kokkos-tools)
endif()

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakeParseArguments)

set(LINKER_LANGUAGE CXX)
set(CUDA_AVAILABLE OFF)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(LINKER_LANGUAGE CUDA)
    set(CUDA_AVAILABLE ON)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build flags" FORCE)
endif()

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ language standard")
set(CMAKE_CUDA_STANDARD 11 CACHE STRING "CUDA language standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "C++ language flags required")
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "CUDA language flags required")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON CACHE BOOL "Use link path to set default rpath")

option(${TIMEMORY_KOKKOS_TOOLS_OPTION}BUILD_SAMPLE "Build sample test" OFF)

set(_OPTIONS_LIST)
function(ADD_CONNECTOR_OPTION _OPT _DESCRIPT _DEFAULT)

    # message(STATUS "TIMEMORY_${_OPT} = ${TIMEMORY_${_OPT}}")

    set(LONG_OPT  ${TIMEMORY_KOKKOS_TOOLS_OPTION}${_OPT})
    set(SHORT_OPT ${_OPT})

    if(TIMEMORY_KOKKOS_TOOLS_IN_SOURCE)
        option(${LONG_OPT} "timemory kokkos-tools: ${_DESCRIPT}" ${TIMEMORY_${_OPT}})
    else()
        option(${LONG_OPT} "timemory kokkos-tools: ${_DESCRIPT}" ${_DEFAULT})
    endif()

    # message(STATUS "${LONG_OPT} = ${${LONG_OPT}}")
    # message(STATUS "${_OPT} = ${${_OPT}}")
    set(${SHORT_OPT} ${${LONG_OPT}})
    set(${SHORT_OPT} ${${LONG_OPT}} PARENT_SCOPE)

    if(NOT TIMEMORY_KOKKOS_TOOLS_IN_SOURCE)
        message(STATUS "${_OPT} = ${${_OPT}}")
    else()
        # message(STATUS "${_OPT} = ${${_OPT}}")
    endif()

    set(${TIMEMORY_KOKKOS_TOOLS_OPTION}${_OPT}_COMPONENTS ${ARGN} PARENT_SCOPE)
    list(APPEND _OPTIONS_LIST ${TIMEMORY_KOKKOS_TOOLS_OPTION}${_OPT})
    set(_OPTIONS_LIST "${_OPTIONS_LIST}" PARENT_SCOPE)
endfunction()

message(STATUS "")

add_connector_option(USE_MPI "Enable MPI support" OFF mpi)
add_connector_option(USE_TAU "Enable TAU support" OFF tau)
add_connector_option(USE_CUDA "Enable CUDA support" OFF cuda cupti)
add_connector_option(USE_PAPI "Enable PAPI support" OFF papi)
add_connector_option(USE_NVTX "Enable NVTX support" OFF nvtx)
add_connector_option(USE_UPCXX "Enable UPC++ support" OFF upcxx)
add_connector_option(USE_VTUNE "Enable VTune support" OFF vtune)
add_connector_option(USE_GOTCHA "Enable GOTCHA support" OFF gotcha)
add_connector_option(USE_LIKWID "Enable LIKWID support" OFF likwid)
add_connector_option(USE_CALIPER "Enable Caliper support" OFF caliper)
add_connector_option(USE_ROOFLINE "Enable roofline support" OFF roofline)
add_connector_option(USE_GPERFTOOLS_CPU "Enable gperftools support" OFF gperftools-cpu)
add_connector_option(USE_GPERFTOOLS_HEAP "Enable gperftools support" OFF gperftools-heap)
add_connector_option(USE_LIBRARY "Enable timemory library and its config options" OFF cxx shared)

if(NOT USE_CUDA)
    set(LINKER_LANGUAGE CXX)
endif()

# CMake INTERFACE target for timemory that provides include path
set(COMPONENTS headers vector)

foreach(_OPT ${_OPTIONS_LIST})
    if(${_OPT})
        list(APPEND COMPONENTS ${${_OPT}_COMPONENTS})
    endif()
endforeach()

message(STATUS "")
set(${TIMEMORY_KOKKOS_TOOLS_OPTION}TIMEMORY_COMPONENTS "" CACHE STRING "Explicit list of timemory components")

# find the timemory package
if(NOT TIMEMORY_KOKKOS_TOOLS_IN_SOURCE)
    set(timemory_INTERFACE_LIBRARY timemory)
    find_package(timemory REQUIRED COMPONENTS headers vector ${COMPONENTS} ${TIMEMORY_COMPONENTS})
else()
    set(timemory_INTERFACE_LIBRARY)
    foreach(_COMP ${${TIMEMORY_KOKKOS_TOOLS_OPTION}TIMEMORY_COMPONENTS} ${COMPONENTS})
        list(APPEND timemory_INTERFACE_LIBRARY timemory-${_COMP})
    endforeach()
endif()

add_library(kp_timemory SHARED ${PROJECT_SOURCE_DIR}/kp_timemory.cpp
    ${PROJECT_SOURCE_DIR}/kp_timemory.hpp)
target_link_libraries(kp_timemory ${timemory_INTERFACE_LIBRARY})
set_target_properties(kp_timemory PROPERTIES
    PREFIX          ""
    LINKER_LANGUAGE ${LINKER_LANGUAGE})

install(TARGETS kp_timemory DESTINATION ${CMAKE_INSTALL_LIBDIR}/kokkos-tools)

#
#   Build the sample
#
if(BUILD_SAMPLE)
    # create test executable
    add_executable(sample sample/sample.cpp)
    target_link_libraries(sample kp_timemory)
endif()

#
#   Build the configurations
#

if(USE_ROOFLINE AND USE_CUDA)
    configure_file(${PROJECT_SOURCE_DIR}/kp_timemory.cpp
        ${PROJECT_BINARY_DIR}/kp_timemory.cu COPYONLY)
endif()

# general options
option(${TIMEMORY_KOKKOS_TOOLS_OPTION}BUILD_CONFIG "Build various connector configurations" ON)
set(${TIMEMORY_KOKKOS_TOOLS_OPTION}BUILD_CONNECTORS "all" CACHE STRING "List of connector configurations to build")
string(TOUPPER "${${TIMEMORY_KOKKOS_TOOLS_OPTION}BUILD_CONNECTORS}" _CONNECTORS)

##--------------------------------------------------------------------------------------##

message(STATUS "")
message(STATUS "Building explicit connector configurations: ${_CONNECTORS}")
message(STATUS "    Disable building explicit configurations with '-DBUILD_CONFIG=OFF'")
message(STATUS "    Reduce explicit configurations with semi-colon delimited '-DBUILD_CONNECTORS=\"...\"'")
message(STATUS "")
message(STATUS "Generating a custom configuration:")
message(STATUS "    '-DUSER_CONNECTOR_CONFIG=\"...\"' --> name for the configuration")
message(STATUS "    '-DUSER_CONNECTOR_TYPES=\"...\"'  --> semi-colon delimited list of components")
message(STATUS "")

##--------------------------------------------------------------------------------------##

function(ADD_CONNECTOR_CONFIG _NAME)
    cmake_parse_arguments(CONNECTOR "GPU" "" "COMPONENTS;TARGETS" ${ARGN})
    if(NOT ${TIMEMORY_KOKKOS_TOOLS_OPTION}BUILD_CONFIG)
        return()
    endif()

    if(NOT "${_NAME}" IN_LIST _CONNECTORS AND NOT "${_CONNECTORS}" STREQUAL "ALL")
        return()
    endif()

    if("${CONNECTOR_COMPONENTS}" STREQUAL "")
        return()
    endif()

    set(SOURCE_FILE ${PROJECT_SOURCE_DIR}/kp_timemory.cpp)
    if(CONNECTOR_GPU AND CUDA_AVAILABLE)
        set_source_files_properties(${SOURCE_FILE} PROPERTIES
            LANGUAGE CUDA
            LINKER_LANGUAGE CUDA)
    endif()

    string(REPLACE ";" "," _COMPONENTS "${CONNECTOR_COMPONENTS}")
    message(STATUS "Building 'kp_timemory_${_NAME}' with '${_COMPONENTS}'...")

    add_library(kp_timemory_${_NAME} SHARED
        ${SOURCE_FILE} ${PROJECT_SOURCE_DIR}/kp_timemory.hpp)
    target_include_directories(kp_timemory_${_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR})
    target_link_libraries(kp_timemory_${_NAME} PRIVATE
        ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-headers
        ${CONNECTOR_TARGETS})
    target_compile_definitions(kp_timemory_${_NAME} PRIVATE
        KP_COMPONENTS=${_COMPONENTS})
    set_target_properties(kp_timemory_${_NAME} PROPERTIES
        PREFIX ""
        LINKER_LANGUAGE ${LINKER_LANGUAGE})
    install(TARGETS kp_timemory_${_NAME}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/kokkos-tools)
endfunction()

##--------------------------------------------------------------------------------------##

add_connector_config(trip_count
    COMPONENTS trip_count
    TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-headers)

add_connector_config(timers
    COMPONENTS wall_clock cpu_clock cpu_util
    TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-headers)

add_connector_config(memory
    COMPONENTS peak_rss page_rss virtual_memory
    TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-headers)

add_connector_config(timers_memory
    COMPONENTS wall_clock cpu_clock cpu_util peak_rss page_rss virtual_memory
    TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-headers)

add_connector_config(io
    COMPONENTS written_bytes read_bytes
    TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-headers)

add_connector_config(context_switch
    COMPONENTS priority_context_switch voluntary_context_switch
    TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-headers)

if(USE_GPERFTOOLS_CPU)
    add_connector_config(gperftools_cpu
        COMPONENTS gperf_cpu_profiler
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-gperftools-cpu)
endif()

if(USE_GPERFTOOLS_HEAP)
    add_connector_config(gperftools_heap
        COMPONENTS gperf_heap_profiler
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-gperftools-heap)
endif()

if(USE_ROOFLINE)
    #add_connector_config(roofline
    #    GPU
    #    COMPONENTS cpu_roofline_flops gpu_roofline_flops
    #    TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-roofline
    #    ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-papi ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cupti
    #    ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cuda)
endif()

if(USE_PAPI)
    add_connector_config(papi
        COMPONENTS papi_array_t
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-papi)
    add_connector_config(cpu_roofline
        COMPONENTS cpu_roofline_flops
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cpu-roofline)
    add_connector_config(cpu_flops
        COMPONENTS papi_tuple<PAPI_DP_OPS,PAPI_SP_OPS>
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-papi)
    add_connector_config(cpu_instructions
        COMPONENTS papi_tuple<PAPI_TOT_INS,PAPI_LD_INS,PAPI_SR_INS>
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-papi)
endif()

if(USE_CUDA)
    add_connector_config(cuda_profiler
        COMPONENTS cuda_profiler
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cuda)
    add_connector_config(cuda_event
        COMPONENTS cuda_event
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cuda)
    add_connector_config(cuda_nvtx
        COMPONENTS nvtx_marker
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cupti ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cuda ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-nvtx)
    add_connector_config(cuda_activity
        COMPONENTS cupti_activity
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cupti ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cuda)
    add_connector_config(cuda_hw_counters
        COMPONENTS cupti_counters
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cupti ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-cuda)
    add_connector_config(gpu_roofline
        GPU
        COMPONENTS gpu_roofline_flops
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-gpu-roofline)
endif()

if(USE_VTUNE)
    add_connector_config(vtune_profiler
        COMPONENTS vtune_profiler
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-vtune)
    add_connector_config(vtune_frame
        COMPONENTS vtune_frame
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-vtune)
endif()

if(USE_CALIPER)
    add_connector_config(caliper
        COMPONENTS caliper
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-caliper)
endif()

if(USE_LIKWID)
    add_connector_config(likwid
        COMPONENTS likwid_perfmon likwid_nvmon
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-likwid)
endif()

if(USE_TAU)
    add_connector_config(tau
        COMPONENTS tau_marker
        TARGETS ${TIMEMORY_KOKKOS_TOOLS_PREFIX}timemory-tau)
endif()

if(USE_GOTCHA)
    set(${TIMEMORY_KOKKOS_TOOLS_OPTION}GOTCHA_COMPONENTS "" CACHE STRING "gotcha configuration components")
    add_connector_config(gotcha
        COMPONENTS ${GOTCHA_COMPONENTS}
        TARGETS ${timemory_INTERFACE_LIBRARY})
endif()

if(USER_CONNECTOR_CONFIG AND USER_CONNECTOR_TYPES)
    string(TOLOWER "${USER_CONNECTOR_CONFIG}" _TAG)
    string(TOLOWER "${USER_CONNECTOR_TYPES}" _TYPES)
    add_connector_config(${_TAG}
        COMPONENTS ${_TYPES}
        TARGETS ${timemory_INTERFACE_LIBRARY})
endif()

message(STATUS "")
