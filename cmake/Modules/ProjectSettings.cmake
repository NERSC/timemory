# include guard
include_guard(DIRECTORY)
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

add_feature(CMAKE_C_FLAGS_${_CONFIG} "C compiler build type flags")
add_feature(CMAKE_CXX_FLAGS_${_CONFIG} "C++ compiler build type flags")

##########################################################################################
#
#   Non-python installation directories
#
##########################################################################################

# cmake installation folder
set(CMAKE_INSTALL_CONFIGDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    CACHE PATH "Installation directory for CMake package config files")

# create the full path version and generic path versions
foreach(_TYPE DATAROOT CMAKE INCLUDE LIB BIN MAN DOC)
    # generic "PROJECT_INSTALL_" variables (used by documentation)"
    set(PROJECT_INSTALL_${_TYPE}DIR ${CMAKE_INSTALL_${TYPE}DIR})
endforeach()
