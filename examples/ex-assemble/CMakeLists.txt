cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(timemory-Assemble-Example LANGUAGES CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-assemble-example)
set(COMPONENTS headers compile-options analysis-tools OPTIONAL_COMPONENTS cxx)

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

# headers are always available
add_executable(ex_assemble ex_assemble.cpp)
target_link_libraries(ex_assemble timemory-assemble-example)
install(TARGETS ex_assemble DESTINATION bin)
