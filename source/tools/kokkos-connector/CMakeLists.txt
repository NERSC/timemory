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

#
#   Project declaration
#
project(timemory-kokkos-connector LANGUAGES C CXX)

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakeParseArguments)

# if built in kokkos-tools or in timemory
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME kokkos)
set(TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/timemory/kokkos-tools)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kokkos-tools)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kokkos-tools)

set(LINKER_LANGUAGE CXX)
set(CUDA_AVAILABLE OFF)
if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(LINKER_LANGUAGE CUDA)
    set(CUDA_AVAILABLE ON)
endif()
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()
if(NOT CMAKE_VERSION VERSION_LESS 3.18)
    cmake_policy(SET CMP0104 OLD)
endif()
set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "CXX language standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "CXX language flags required")
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "CUDA language standard")
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "CUDA language flags required")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(TIMEMORY_KOKKOS_COMPONENTS "" CACHE STRING "Explicit list of timemory components")
set(TIMEMORY_BUILD_KOKKOS_CONNECTORS "all" CACHE STRING "List of connector configurations to build")

#
#   Options
#
add_option(TIMEMORY_KOKKOS_BUILD_SAMPLE "Build sample test" OFF NO_FEATURE)
add_option(TIMEMORY_BUILD_KOKKOS_CONFIG "Build various connector configurations" OFF)
add_feature(TIMEMORY_BUILD_KOKKOS_CONNECTORS "List of connector configurations to build" DOC)
if(DEFINED TIMEMORY_KOKKOS_BUILD_CONFIG)
    set(TIMEMORY_BUILD_KOKKOS_CONFIG ${TIMEMORY_KOKKOS_BUILD_CONFIG})
endif()
if(DEFINED TIMEMORY_KOKKOS_BUILD_CONNECTORS)
    set(TIMEMORY_BUILD_KOKKOS_CONNECTORS ${TIMEMORY_KOKKOS_BUILD_CONNECTORS})
endif()
string(TOUPPER "${TIMEMORY_BUILD_KOKKOS_CONNECTORS}" _CONNECTORS)

message(STATUS "")
#
#   Configure common target
#
set(timemory_INTERFACE_LIBRARY)
foreach(_COMP headers vector dmp cxx-shared mpip-library ompt-library ncclp-library
        ${TIMEMORY_KOKKOS_COMPONENTS})
    if(TARGET timemory::timemory-${_COMP})
        list(APPEND timemory_INTERFACE_LIBRARY timemory::timemory-${_COMP})
    endif()
endforeach()

#
#   Function for common routines on libraries
#   Any additional args are treated as libraries
#
function(configure_library _NAME)
    target_include_directories(${_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(${_NAME} PRIVATE ${timemory_INTERFACE_LIBRARY} ${ARGN})
    set_target_properties(${_NAME} PROPERTIES
        PREFIX          ""
        LINKER_LANGUAGE ${LINKER_LANGUAGE})
    install(TARGETS ${_NAME} DESTINATION ${TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR})
endfunction()

#
#   Build object library
#
add_library(kp_timemory_common OBJECT ${PROJECT_SOURCE_DIR}/kp_timemory_common.cpp
    ${PROJECT_SOURCE_DIR}/kp_timemory.hpp)
target_link_libraries(kp_timemory_common PUBLIC
    timemory::timemory-headers
    ${timemory_INTERFACE_LIBRARY})
set(timemory_COMMON_SOURCES $<TARGET_OBJECTS:kp_timemory_common>)

#
#   Build generic library
#
add_library(kp_timemory SHARED ${PROJECT_SOURCE_DIR}/kp_timemory.cpp
    ${timemory_COMMON_SOURCES})
configure_library(kp_timemory)

#
#   Build regex filtering library
#
add_library(kp_timemory_filter SHARED ${PROJECT_SOURCE_DIR}/kp_timemory_filter.cpp
     ${timemory_COMMON_SOURCES})
configure_library(kp_timemory_filter)

#
#   Build the sample (test executable)
#
if(TIMEMORY_KOKKOS_BUILD_SAMPLE)
    add_executable(sample sample/sample.cpp)
    target_link_libraries(sample PRIVATE kp_timemory)
endif()

##--------------------------------------------------------------------------------------##

if(NOT TIMEMORY_BUILD_KOKKOS_CONFIG)
    message(STATUS "")
    message(STATUS
        "[${PROJECT_NAME}] Disabled building explicit configurations with 'TIMEMORY_BUILD_KOKKOS_CONFIG=OFF'")
    message(STATUS "")
    return()
endif()

##--------------------------------------------------------------------------------------##

message(STATUS "")
message(STATUS "Building explicit connector configurations: ${_CONNECTORS}")
message(STATUS "    Disable building explicit configurations with '-DTIMEMORY_BUILD_KOKKOS_CONFIG=OFF'")
message(STATUS "    Reduce explicit configurations with semi-colon delimited '-DTIMEMORY_BUILD_KOKKOS_CONNECTORS=\"...\"'")
message(STATUS "")

##--------------------------------------------------------------------------------------##

function(ADD_CONNECTOR_CONFIG _NAME)
    cmake_parse_arguments(CONNECTOR "GPU" "" "COMPONENTS;TARGETS;DEFINITIONS;ENVIRONMENT" ${ARGN})
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
    set(_ENVIRON)
    foreach(_ENV ${CONNECTOR_ENVIRONMENT})
       set(_ENVIRON "${_ENVIRON}tim::set_env(\"${_ENV}\", 1);\n")
    endforeach()
    string(REPLACE "=" "\", \"" _ENVIRON "${_ENVIRON}")
    message(STATUS "Building 'kp_timemory_${_NAME}' with '${_COMPONENTS}'...")
    set(KP_COMPONENTS ${_COMPONENTS})
    set(KP_ENVIRON ${_ENVIRON})
    set(KP_LIBNAME "kp_timemory_${_NAME}")

    set(SOURCE_FILE ${PROJECT_BINARY_DIR}/kp_timemory_${_NAME}.${_EXT})
    configure_file(${PROJECT_SOURCE_DIR}/kp_timemory.cpp.in ${SOURCE_FILE} @ONLY)

    add_library(kp_timemory_${_NAME} SHARED
        ${SOURCE_FILE} ${timemory_COMMON_SOURCES})
    configure_library(kp_timemory_${_NAME} ${CONNECTOR_TARGETS})
    target_compile_definitions(kp_timemory_${_NAME} PRIVATE
        ${CONNECTOR_DEFINITIONS})
endfunction()

##--------------------------------------------------------------------------------------##

add_connector_config(trip_count
    COMPONENTS trip_count
    TARGETS timemory::timemory-headers)

add_connector_config(timers
    COMPONENTS wall_clock cpu_clock cpu_util
    TARGETS timemory::timemory-headers)

add_connector_config(memory
    COMPONENTS peak_rss page_rss virtual_memory current_peak_rss
    TARGETS timemory::timemory-headers)

add_connector_config(timers_memory
    COMPONENTS wall_clock cpu_clock cpu_util peak_rss page_rss virtual_memory
    TARGETS timemory::timemory-headers)

add_connector_config(io
    COMPONENTS read_bytes read_char written_bytes written_char
    TARGETS timemory::timemory-headers)

add_connector_config(context_switch
    COMPONENTS priority_context_switch voluntary_context_switch
    TARGETS timemory::timemory-headers)

if(TIMEMORY_USE_GPERFTOOLS)
    add_connector_config(gperftools
        COMPONENTS gperftools_cpu_profiler gperftools_heap_profiler
        TARGETS timemory::timemory-gperftools)
endif()

if(TIMEMORY_USE_PAPI)
    add_connector_config(papi
        COMPONENTS papi_array_t
        TARGETS timemory::timemory-papi-component)

    add_connector_config(cpu_roofline
        COMPONENTS cpu_roofline_flops
        TARGETS
        timemory::timemory-ert
        timemory::timemory-cpu-roofline
        timemory::timemory-papi-component
        timemory::timemory-roofline-component)

    # creates a normal connector config and a rates version
    function(add_papi_connector_config _NAME)
        cmake_parse_arguments(
            CONNECTOR "" "" "COMPONENTS;TARGETS;DEFINITIONS;ENVIRONMENT" ${ARGN})
        add_connector_config(${_NAME}
            COMPONENTS  ${CONNECTOR_COMPONENTS}
            TARGETS     ${CONNECTOR_TARGETS}
            DEFINITIONS ${CONNECTOR_DEFINITIONS}
            ENVIRONMENT ${CONNECTOR_ENVIRONMENT}
            TARGETS timemory::timemory-papi)
        string(REPLACE "papi_tuple" "papi_rate_tuple"
            CONNECTOR_COMPONENTS "${CONNECTOR_COMPONENTS}")
        add_connector_config(${_NAME}_rate
            COMPONENTS  ${CONNECTOR_COMPONENTS}
            TARGETS     ${CONNECTOR_TARGETS}
            DEFINITIONS ${CONNECTOR_DEFINITIONS}
            ENVIRONMENT ${CONNECTOR_ENVIRONMENT}
            TARGETS timemory::timemory-papi)
    endfunction()

    add_papi_connector_config(cpu_flops
        COMPONENTS papi_tuple<PAPI_DP_OPS,PAPI_SP_OPS>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_instructions
        COMPONENTS papi_tuple<PAPI_TOT_INS,PAPI_LD_INS,PAPI_SR_INS>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_cache_misses
        COMPONENTS papi_tuple<PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_tlb
        COMPONENTS papi_tuple<PAPI_TLB_DM,PAPI_TLB_IM,PAPI_TLB_TL,PAPI_TLB_SD>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_branching
        COMPONENTS
        papi_tuple<PAPI_BR_UCN,PAPI_BR_CN,PAPI_BR_TKN,PAPI_BR_NTK,PAPI_BR_MSP,PAPI_BR_PRC,PAPI_BR_INS>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_stalls
        COMPONENTS
        papi_tuple<PAPI_MEM_SCY,PAPI_MEM_RCY,PAPI_MEM_WCY,PAPI_RES_STL,PAPI_FP_STAL>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_l1_misses
        COMPONENTS papi_tuple<PAPI_L1_DCM,PAPI_L1_ICM,PAPI_L1_TCM,PAPI_L1_LDM,PAPI_L1_STM>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_l2_misses
        COMPONENTS papi_tuple<PAPI_L2_DCM,PAPI_L2_ICM,PAPI_L2_TCM,PAPI_L2_LDM,PAPI_L2_STM>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_l3_misses
        COMPONENTS papi_tuple<PAPI_L3_DCM,PAPI_L3_ICM,PAPI_L3_TCM,PAPI_L3_LDM,PAPI_L3_STM>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_l1_total
        COMPONENTS papi_tuple<PAPI_L1_TCM,PAPI_L1_TCH,PAPI_L1_TCA,PAPI_L1_TCR,PAPI_L1_TCW>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_l2_total
        COMPONENTS papi_tuple<PAPI_L2_TCM,PAPI_L2_TCH,PAPI_L2_TCA,PAPI_L2_TCR,PAPI_L2_TCW>
        TARGETS timemory::timemory-papi)

    add_papi_connector_config(cpu_l3_total
        COMPONENTS papi_tuple<PAPI_L3_TCM,PAPI_L3_TCH,PAPI_L3_TCA,PAPI_L3_TCR,PAPI_L3_TCW>
        TARGETS timemory::timemory-papi)
endif()

if(TIMEMORY_USE_CUDA)
    add_connector_config(cuda_profiler
        COMPONENTS nvtx_marker
        TARGETS
        timemory::timemory-cuda-component)

    add_connector_config(cuda_event
        COMPONENTS cuda_event
        TARGETS
        timemory::timemory-cuda-component)

    add_connector_config(cuda_activity
        COMPONENTS cupti_activity
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(cuda_hw_counters
        COMPONENTS cupti_counters
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_roofline
        GPU
        COMPONENTS gpu_roofline_flops
        TARGETS
        timemory::timemory-ert
        timemory::timemory-gpu-roofline
        timemory::timemory-roofline-component)

    add_connector_config(gpu_instructions
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=inst_issued,inst_executed,thread_inst_executed,ldst_executed,inst_integer,inst_fp_16,inst_fp_32,inst_fp_64,inst_inter_thread_communication"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_cache_misses
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=tex_cache_transactions,tex_cache_hit_rate,tex0_cache_sector_queries,tex0_cache_sector_misses,tex1_cache_sector_queries,tex1_cache_sector_misses,l2_tex_hit_rate,dram_read_bytes,dram_write_bytes"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_branching
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=divergent_branch,branch,branch_efficiency"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_stalls
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=stall_inst_fetch,stall_exec_dependency,stall_memory_dependency,stall_texture,stall_sync,stall_other,stall_constant_memory_dependency,stall_pipe_busy,stall_memory_throttle,stall_not_selected"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_l2_total
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=l2_read_transactions,l2_write_transactions,l2_atomic_transactions,l2_global_load_bytes,l2_local_load_bytes,l2_local_global_store_bytes,l2_global_reduction_bytes,l2_global_atomic_store_bytes"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_efficiency
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=branch_efficiency,warp_execution_efficiency,warp_nonpred_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency,sm_efficiency,flop_hp_efficiency,flop_sp_efficiency,flop_dp_efficiency"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_ipc
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=ipc,issued_ipc"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_warps
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=warps_launched,active_warps,inst_per_warp,warp_execution_efficiency,warp_nonpred_execution_efficiency,unique_warps_launched,eligible_warps_per_cycle"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_double_flops
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,flop_dp_efficiency"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_single_flops
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_sp_efficiency"
        TARGETS
        timemory::timemory-cupti-component)

    add_connector_config(gpu_half_flops
        COMPONENTS cupti_counters
        ENVIRONMENT "TIMEMORY_CUPTI_METRICS=flop_count_hp,flop_count_hp_add,flop_count_hp_mul,flop_count_hp_fma,flop_hp_efficiency"
        TARGETS
        timemory::timemory-cupti-component)
endif()

if(TIMEMORY_USE_VTUNE)
    add_connector_config(vtune
        COMPONENTS vtune_profiler vtune_frame vtune_event
        TARGETS timemory::timemory-vtune)
endif()

if(TIMEMORY_USE_CALIPER)
    add_connector_config(caliper
        COMPONENTS caliper_config caliper_marker
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
