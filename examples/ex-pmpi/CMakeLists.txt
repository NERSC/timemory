cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

if(WIN32)
	return()
endif()

project(timemory-pmpi-Example LANGUAGES CXX)

include(GNUInstallDirs)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-pmpi-shared)
set(COMPONENTS mpi cxx shared compile-options arch vector gotcha)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_library(timemory_pmpi SHARED timemory_pmpi.cpp)
target_link_libraries(timemory_pmpi PUBLIC timemory-pmpi-shared)

install(TARGETS timemory_pmpi DESTINATION ${CMAKE_INSTALL_LIBDIR})
