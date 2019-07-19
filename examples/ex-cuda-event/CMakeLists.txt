cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CUDA-Event-Example LANGUAGES C CXX CUDA)

set(EXE_NAME test_cuda_event)
set(COMPONENTS cuda papi threading compile-options analysis-tools)

option(USE_CUPTI "Enable CUPTI" OFF)
if(USE_CUPTI)
    list(APPEND COMPONENTS cupti)
endif()

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS extern-templates)
endif()

option(USE_LOGGING "Enable logging in debug mode" OFF)
add_library(logging INTERFACE)
if(USE_LOGGING)
    target_compile_definitions(logging INTERFACE $<$<CONFIG:Debug>:DEBUG>)
endif()

set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-cuda-event-example)
find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})


add_executable(${EXE_NAME} ${EXE_NAME}.cu)
target_link_libraries(${EXE_NAME} timemory-cuda-event-example logging)
target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

if(USE_CUPTI)
    add_executable(demo demo.cu)
    target_link_libraries(demo timemory-headers timemory-cuda timemory-cupti logging)
    target_include_directories(demo PRIVATE ${CMAKE_CURRENT_LIST_DIR})
endif()

install(TARGETS ${EXE_NAME} DESTINATION bin)
