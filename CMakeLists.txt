cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)


project(TiMEmory LANGUAGES C CXX VERSION 1.0.0)


################################################################################
#
#        Macro to add to string
#
################################################################################

macro(add _VAR _FLAG)
    if(NOT "${_FLAG}" STREQUAL "")
        if("${${_VAR}}" STREQUAL "")
            set(${_VAR} "${_FLAG}")
        else()
            set(${_VAR} "${${_VAR}} ${_FLAG}")
        endif()
    endif()
endmacro()

################################################################################
#
#        MPI
#
################################################################################

find_package(MPI REQUIRED)

# Add the MPI-specific compiler and linker flags
add(CMAKE_CXX_FLAGS  "${MPI_CXX_COMPILE_FLAGS}")
add(CMAKE_EXE_LINKER_FLAGS "${MPI_CXX_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

set(MPI_LIBRARIES )
foreach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)
    set(_TYPE MPI_${_TYPE})
    if(${_TYPE})
        list(APPEND MPI_LIBRARIES ${${_TYPE}})
    endif(${_TYPE})
endforeach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)
add(EXTERNAL_LIBRARIES ${MPI_LIBRARIES})


################################################################################
#
#        Miscellaneous
#
################################################################################

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)


################################################################################
#
#        TiMEmory
#
################################################################################

add_subdirectory(pybind11)
add_subdirectory(source)
