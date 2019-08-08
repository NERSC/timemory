# include guard
include_guard(DIRECTORY)

#
# Project settings
#
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _CONFIG)

if(WIN32)
    set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard")
else(WIN32)
    set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
endif(WIN32)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TIMEMORY_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

add_feature(CMAKE_C_FLAGS_${_CONFIG} "C compiler build type flags")
add_feature(CMAKE_CXX_FLAGS_${_CONFIG} "C++ compiler build type flags")

##########################################################################################
#
#   Non-python installation directories
#
##########################################################################################

# cmake installation folder
set(TIMEMORY_INSTALL_CMAKEDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    CACHE PATH "Installation directory for CMake package config files")
# the rest of the installation folders
foreach(_TYPE in DATAROOT INCLUDE LIB BIN MAN DOC)
    set(TIMEMORY_INSTALL_${_TYPE}DIR ${CMAKE_INSTALL_${_TYPE}DIR})
endforeach(_TYPE in DATAROOT INCLUDE LIB BIN MAN DOC)

# create the full path version and generic path versions
foreach(_TYPE in DATAROOT CMAKE INCLUDE LIB BIN MAN DOC)
    # set the absolute versions
    if(NOT IS_ABSOLUTE "${TIMEMORY_INSTALL_${_TYPE}DIR}")
        set(TIMEMORY_INSTALL_FULL_${_TYPE}DIR ${CMAKE_INSTALL_PREFIX}/${TIMEMORY_INSTALL_${_TYPE}DIR})
    else(NOT IS_ABSOLUTE "${TIMEMORY_INSTALL_${_TYPE}DIR}")
        set(TIMEMORY_INSTALL_FULL_${_TYPE}DIR ${TIMEMORY_INSTALL_${_TYPE}DIR})
    endif(NOT IS_ABSOLUTE "${TIMEMORY_INSTALL_${_TYPE}DIR}")

    # generic "PROJECT_INSTALL_" variables (used by documentation)"
    set(PROJECT_INSTALL_${_TYPE}DIR ${TIMEMORY_INSTALL_${TYPE}DIR})
    set(PROJECT_INSTALL_FULL_${_TYPE}DIR ${TIMEMORY_INSTALL_FULL_${TYPE}DIR})

endforeach()
