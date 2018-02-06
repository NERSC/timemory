################################################################################
#
#        Handles the build settings
#
################################################################################

include(GNUInstallDirs)
add_option(TIMEMORY_EXCEPTIONS "Signal handler throws exceptions (default: exit)" OFF)

if(NOT SUBPROJECT)
    set(SANITIZE_TYPE leak CACHE STRING "-fsantitize=<TYPE>")
    add_option(ENABLE_SANITIZE "Enable -fsanitize flag (=${SANITIZE_TYPE})" OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    if(WIN32)
        set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard")
    else(WIN32)
        set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
    endif(WIN32)
    add_feature(CMAKE_CXX_STANDARD "C++11 STL standard")

    if(GOOD_CMAKE)

        add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ standard" ON)
        add_option(CMAKE_CXX_EXTENSIONS "Build with CXX extensions (e.g. gnu++11)" OFF)
        set(CMAKE_INSTALL_MESSAGE LAZY)

    endif(GOOD_CMAKE)

    # ensure only C++11, C++14, or C++17
    if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "11" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "14" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "17" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "1y" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "1z")

        if(WIN32)
            set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard" FORCE)
        else(WIN32)
            set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard" FORCE)
        endif(WIN32)

    endif(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "11" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "14" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "17" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "1y" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "1z")

    if(CMAKE_CXX_COMPILER_IS_GNU)
        add(CMAKE_CXX_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
    elseif(CMAKE_CXX_COMPILER_IS_CLANG)
        add(CMAKE_CXX_FLAGS "-std=c++${CMAKE_CXX_STANDARD} -stdlib=libc++")
    elseif(CMAKE_CXX_COMPILER_IS_INTEL)
        add(CMAKE_CXX_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
    elseif(CMAKE_CXX_COMPILER_IS_PGI)
        add(CMAKE_CXX_FLAGS "--c++${CMAKE_CXX_STANDARD} -A")
    elseif(CMAKE_CXX_COMPILER_IS_XLC)
        if(CMAKE_CXX_STANDARD GREATER 11)
            add(CMAKE_CXX_FLAGS "-std=c++1y")
        else(CMAKE_CXX_STANDARD GREATER 11)
            add(CMAKE_CXX_FLAGS "-std=c++11")
        endif(CMAKE_CXX_STANDARD GREATER 11)
    elseif(CMAKE_CXX_COMPILER_IS_MSVC)
        add(CMAKE_CXX_FLAGS "-std:c++${CMAKE_CXX_STANDARD}")
    endif(CMAKE_CXX_COMPILER_IS_GNU)

endif(NOT SUBPROJECT)


# set the output directory (critical on Windows
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    # if TIMEMORY_OUTPUT_DIR is not defined, set to CMAKE_BINARY_DIR
    if(NOT DEFINED TIMEMORY_OUTPUT_DIR OR "${TIMEMORY_OUTPUT_DIR}" STREQUAL "")
        set(TIMEMORY_OUTPUT_DIR ${CMAKE_BINARY_DIR})
    endif(NOT DEFINED TIMEMORY_OUTPUT_DIR OR "${TIMEMORY_OUTPUT_DIR}" STREQUAL "")
    # set the CMAKE_{ARCHIVE,LIBRARY,RUNTIME}_OUTPUT_DIRECTORY variables
    if(WIN32)
        # on Windows, separate types into different directories
        string(TOLOWER "${_TYPE}" _LTYPE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${TIMEMORY_OUTPUT_DIR}/outputs/${_LTYPE})
    else(WIN32)
        # on UNIX, just set to same directory
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${TIMEMORY_OUTPUT_DIR})
    endif(WIN32)
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)

# cmake installation folder
set(PROJECT_INSTALL_CMAKEDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    CACHE PATH "Installation directory for CMake package config files")

# used by configure_package_*
set(INCLUDE_INSTALL_DIR     ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR         ${CMAKE_INSTALL_LIBDIR})
set(LIBNAME                 timemory)

# set the compiler flags if not on Windows
if(NOT SUBPROJECT AND NOT WIN32)

    add(CMAKE_CXX_FLAGS "-W -Wall -Wextra -faligned-new ${CXXFLAGS} $ENV{CXXFLAGS}")
    add(CMAKE_CXX_FLAGS "-Wno-unused-parameter")
    add(CMAKE_CXX_FLAGS "-Wunused-but-set-parameter -Wno-unused-variable")

    if(NOT CMAKE_CXX_COMPILER_IS_INTEL)
        add(CMAKE_CXX_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_CXX_FLAGS "-Wno-implicit-fallthrough")
        add(CMAKE_CXX_FLAGS "-Wno-shadow-field-in-constructor-modified")
        add(CMAKE_CXX_FLAGS "-Wno-exceptions")
        add(CMAKE_CXX_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_CXX_FLAGS "-Wno-unused-private-field")
    endif(NOT CMAKE_CXX_COMPILER_IS_INTEL)

    add(CMAKE_C_FLAGS "-W -Wall -Wextra -faligned-new ${CFLAGS} $ENV{CFLAGS}")
    add(CMAKE_C_FLAGS "-Wno-unused-parameter")
    add(CMAKE_C_FLAGS "-Wunused-but-set-parameter -Wno-unused-variable")

    if(NOT CMAKE_C_COMPILER_IS_INTEL)
        add(CMAKE_C_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_C_FLAGS "-Wno-implicit-fallthrough")
        add(CMAKE_C_FLAGS "-Wno-shadow-field-in-constructor-modified")
        add(CMAKE_C_FLAGS "-Wno-exceptions")
        add(CMAKE_C_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_C_FLAGS "-Wno-unused-private-field")
    endif(NOT CMAKE_C_COMPILER_IS_INTEL)

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

        add(CMAKE_C_FLAGS "-DDEBUG")
        add(CMAKE_CXX_FLAGS "-DDEBUG")

    else("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

        if(CMAKE_CXX_COMPILER_IS_INTEL)
            add(CMAKE_CXX_FLAGS "-xHOST")
        else(CMAKE_CXX_COMPILER_IS_INTEL)
            add(CMAKE_CXX_FLAGS "-march=native")
        endif(CMAKE_CXX_COMPILER_IS_INTEL)

        if(CMAKE_C_COMPILER_IS_INTEL)
            add(CMAKE_C_FLAGS "-xHOST")
        else(CMAKE_C_COMPILER_IS_INTEL)
            add(CMAKE_C_FLAGS "-march=native")
        endif(CMAKE_C_COMPILER_IS_INTEL)

    endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

    if(UNIX)
        add(CMAKE_CXX_FLAGS "-pthread")
        add(CMAKE_C_FLAGS "-pthread")
    endif(UNIX)

    if(ENABLE_SANITIZE)
        add_subfeature(ENABLE_SANITIZE SANITIZE_TYPE "Sanitizer type")
        add(CMAKE_CXX_FLAGS "-fsanitize=${SANITIZE_TYPE}")
    endif(ENABLE_SANITIZE)

    add_c_flags(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    add_cxx_flags(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

endif(NOT SUBPROJECT AND NOT WIN32)

if(TIMEMORY_EXCEPTIONS)
    if(WIN32)
        add_definitions(/DTIMEMORY_EXCEPTIONS)
    else(WIN32)
        add_definitions(-DTIMEMORY_EXCEPTIONS)
    endif(WIN32)
endif()
