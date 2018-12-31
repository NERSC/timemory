cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#------------------------------------------------------------------------------#
project(TiMemory-CXX-Example LANGUAGES C CXX)
message(STATUS "Project: ${PROJECT_NAME}")

set(EXE_NAME test_cxx_timing)
find_package(TiMemory REQUIRED)

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME}
    PUBLIC ${TiMemory_LIBRARIES} ${EXTERNAL_LIBRARIES})
target_include_directories(${EXE_NAME} SYSTEM
    PUBLIC ${TiMemory_INCLUDE_DIRS})
target_compile_options(${EXE_NAME}
    PUBLIC ${TiMemory_CXX_COMPILE_OPTIONS})
set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)

if(DEFINED PROJECT_DEPENDS)
    add_dependencies(${EXE_NAME} ${PROJECT_DEPENDS})
endif(DEFINED PROJECT_DEPENDS)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
