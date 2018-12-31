cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#------------------------------------------------------------------------------#
project(TiMemory-CXX-MPI-Example LANGUAGES C CXX)
message(STATUS "Project: ${PROJECT_NAME}")

set(EXE_NAME test_cxx_mpi_timing)

find_package(MPI REQUIRED)
find_package(TiMemory REQUIRED)

foreach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)
    list(APPEND EXTERNAL_LIBRARIES ${MPI_${_TYPE}})
endforeach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)

set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ library standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ library standard")

#------------------------------------------------------------------------------#
# Add the executable, and link it to the TiMemory libraries
#
add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} ${TiMemory_LIBRARIES} ${EXTERNAL_LIBRARIES})
target_include_directories(${EXE_NAME} SYSTEM
    PUBLIC ${TiMemory_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
target_compile_options(${EXE_NAME}
    PUBLIC ${TiMemory_CXX_COMPILE_OPTIONS} ${MPI_CXX_COMPILE_FLAGS})
set_target_properties(${EXE_NAME} PROPERTIES
    LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)

if(DEFINED PROJECT_DEPENDS)
    add_dependencies(${EXE_NAME} ${PROJECT_DEPENDS})
endif(DEFINED PROJECT_DEPENDS)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
