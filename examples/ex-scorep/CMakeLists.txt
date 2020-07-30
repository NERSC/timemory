cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_SCOREP)
    return()
endif()

project(timemory-ScoreP-Example LANGUAGES C CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-scorep-example)
set(COMPONENTS headers compile-options analysis-tools cxx scorep-component)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

# headers are always available
add_executable(ex_scorep ex_scorep.cpp)
target_link_libraries(ex_scorep timemory-scorep-example)
# target_link_libraries(ex_scorep timemory-scorep-component)
install(TARGETS ex_scorep DESTINATION bin
    COMPONENT examples OPTIONAL)
