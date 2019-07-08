cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CUDA-Event-Example LANGUAGES C CXX CUDA)

set(EXE_NAME test_cuda_event)
set(COMPONENTS cxx cuda papi threading compile-options analysis-tools)

option(USE_CUPTI "Enable CUPTI" OFF)
if(USE_CUPTI)
    list(APPEND COMPONENTS cupti)
endif()

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS extern-templates)
endif()

set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-cuda-event-example)
find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cu)
target_link_libraries(${EXE_NAME} timemory-cuda-event-example)
target_compile_options(${EXE_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
install(TARGETS ${EXE_NAME} DESTINATION bin)

if(USE_CUPTI)
    add_executable(demo demo.cu)
    target_link_libraries(demo timemory-headers timemory-cuda timemory-cupti)
    target_include_directories(demo PRIVATE ${CMAKE_CURRENT_LIST_DIR})
endif()
