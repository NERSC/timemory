cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#------------------------------------------------------------------------------#
project(TiMemory-CXX-Tuple-Example LANGUAGES C CXX)
message(STATUS "Project: ${PROJECT_NAME}")

set(EXE_NAME test_cxx_tuple)
set(CMAKE_THREAD_PREFER_PTHREAD ON)

find_package(TiMemory REQUIRED)
find_package(Threads REQUIRED)

if(WIN32)
    list(APPEND ${PROJECT_NAME}_CXX_FLAGS /bigobj)
else()
    list(APPEND TiMemory_CXX_LINK_OPTIONS -lpthread)
endif()

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)

target_link_libraries(${EXE_NAME} PRIVATE
    TiMemory::TiMemory_CXX Threads::Threads)

target_include_directories(${EXE_NAME} SYSTEM PRIVATE
    ${TiMemory_INCLUDE_DIRS})

target_compile_options(${EXE_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${TiMemory_C_COMPILE_OPTIONS} ${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${TiMemory_CXX_COMPILE_OPTIONS} ${${PROJECT_NAME}_CXX_FLAGS}>)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)

foreach(_OPT ${TiMemory_CXX_LINK_OPTIONS})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_OPT}")
endforeach()

#if(DEFINED PROJECT_DEPENDS)
#    add_dependencies(${EXE_NAME} ${PROJECT_DEPENDS})
#endif(DEFINED PROJECT_DEPENDS)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
