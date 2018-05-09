cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE OFF)
else(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE ON)
    cmake_policy(SET CMP0048 NEW)
endif(CMAKE_VERSION VERSION_LESS 3.1.3)
cmake_policy(SET CMP0042 NEW)

#------------------------------------------------------------------------------#
project(TiMemory-CXX-MPI-Example C CXX)

set(EXE_NAME test_cxx_mpi_timing)

find_package(MPI REQUIRED)
find_package(TiMemory REQUIRED)
foreach(_DIR ${TiMemory_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach(_DIR ${TiMemory_INCLUDE_DIRS})

# Add the MPI-specific compiler and linker flags
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH}
    ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

foreach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)
    set(_TYPE MPI_${_TYPE})
    if(${_TYPE})
        list(APPEND MPI_LIBRARIES ${${_TYPE}})
    endif(${_TYPE})
endforeach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)

list(APPEND EXTERNAL_LIBRARIES ${MPI_LIBRARIES})

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
endif(NOT DEFINED CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TiMemory_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TiMemory_CXX_FLAGS}")

#------------------------------------------------------------------------------#
# Add the executable, and link it to the TiMemory libraries
#
add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} ${TiMemory_LIBRARIES} ${EXTERNAL_LIBRARIES})

if(DEFINED PROJECT_DEPENDS)
    add_dependencies(${EXE_NAME} ${PROJECT_DEPENDS})
endif(DEFINED PROJECT_DEPENDS)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
