cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#------------------------------------------------------------------------------#
project(TiMemory-CUDA-Event-Example LANGUAGES C CXX CUDA)
message(STATUS "Project: ${PROJECT_NAME}")

set(EXE_NAME test_cuda_event)
find_package(TiMemory REQUIRED)

add_executable(${EXE_NAME} ${EXE_NAME}.cu)

target_link_libraries(${EXE_NAME} PRIVATE
    ${TiMemory_LIBRARIES})

target_include_directories(${EXE_NAME} SYSTEM PRIVATE
    ${TiMemory_INCLUDE_DIRS})

target_compile_options(${EXE_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${TiMemory_C_COMPILE_OPTIONS} ${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${TiMemory_CXX_COMPILE_OPTIONS} ${${PROJECT_NAME}_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${TiMemory_CUDA_COMPILE_OPTIONS} ${${PROJECT_NAME}_CUDA_FLAGS}>)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON)

if(DEFINED PROJECT_DEPENDS)
    add_dependencies(${EXE_NAME} ${PROJECT_DEPENDS})
endif()

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif()
