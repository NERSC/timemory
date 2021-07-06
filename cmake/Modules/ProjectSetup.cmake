# only include this once
include_guard(GLOBAL)

#----------------------------------------------------------------------------------------#
#   handle ccache
#----------------------------------------------------------------------------------------#

option(TIMEMORY_CCACHE_BUILD "Enable ccache build" OFF)
mark_as_advanced(TIMEMORY_CCACHE_BUILD)
if(TIMEMORY_CCACHE_BUILD)
    find_program(TIMEMORY_CCACHE_EXE ccache PATH_SUFFIXES bin)
    if(TIMEMORY_CCACHE_EXE)
        if(NOT EXISTS "${TIMEMORY_CCACHE_EXE}")
            message(WARNING
                "TIMEMORY_CCACHE_BUILD is ON but TIMEMORY_CCACHE_EXE (${TIMEMORY_CCACHE_EXE}) does not exist!")
        else()
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${TIMEMORY_CCACHE_EXE}")
        endif()
    endif()
endif()

#----------------------------------------------------------------------------------------#
#   options
#----------------------------------------------------------------------------------------#

# make sure testing enabled
if(TIMEMORY_BUILD_MINIMAL_TESTING)
    set(TIMEMORY_BUILD_TESTING ON)
endif()
# this gets annoying
if(TIMEMORY_BUILD_GOOGLE_TEST OR TIMEMORY_BUILD_TESTING)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
        "Suppress Warnings that are meant for the author of the CMakeLists.txt files")
endif()
# override any cache settings
if(TIMEMORY_BUILD_TESTING)
    set(TIMEMORY_BUILD_GOOGLE_TEST ON)
    if(NOT TIMEMORY_BUILD_MINIMAL_TESTING)
        set(TIMEMORY_BUILD_EXAMPLES ON)
    endif()
else()
    if(TIMEMORY_BUILD_MINIMAL_TESTING)
        set(TIMEMORY_BUILD_GOOGLE_TEST ON)
        set(TIMEMORY_BUILD_EXAMPLES OFF)
    endif()
endif()
#
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

#----------------------------------------------------------------------------------------#
#   versioning
#----------------------------------------------------------------------------------------#

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(REGEX REPLACE
    "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)"
    "\\1.\\2.\\3"
    VERSION_STRING "${FULL_VERSION_STRING}")
set(TIMEMORY_VERSION "${VERSION_STRING}")
if(NOT "${TIMEMORY_VERSION}" STREQUAL "${FULL_VERSION_STRING}")
    message(STATUS "[timemory] version ${TIMEMORY_VERSION} (${FULL_VERSION_STRING})")
else()
    message(STATUS "[timemory] version ${TIMEMORY_VERSION}")
endif()
set(TIMEMORY_VERSION_STRING "${FULL_VERSION_STRING}")
string(REPLACE "." ";" VERSION_LIST "${VERSION_STRING}")
LIST(GET VERSION_LIST 0 TIMEMORY_VERSION_MAJOR)
LIST(GET VERSION_LIST 1 TIMEMORY_VERSION_MINOR)
LIST(GET VERSION_LIST 2 TIMEMORY_VERSION_PATCH)
set(TIMEMORY_VERSION
    "${TIMEMORY_VERSION_MAJOR}.${TIMEMORY_VERSION_MINOR}.${TIMEMORY_VERSION_PATCH}")

math(EXPR TIMEMORY_VERSION_CODE
    "${TIMEMORY_VERSION_MAJOR} * 10000 + ${TIMEMORY_VERSION_MINOR} * 100 + ${TIMEMORY_VERSION_PATCH}")

#----------------------------------------------------------------------------------------#
#   setup.py
#----------------------------------------------------------------------------------------#

if(SKBUILD AND TIMEMORY_USE_PYTHON)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()
