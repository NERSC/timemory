################################################################################
#
#        Handles the build settings
#
################################################################################

include(GNUInstallDirs)

if(NOT SUBPROJECT)

    # Special Intel compiler flags for NERSC Cori
    foreach(_LANG C CXX)
        if(CMAKE_${_LANG}_COMPILER_IS_INTEL)
            add_option(TIMEMORY_INTEL_${_LANG}_AVX512 "Enable -axMIC-AVX512 flags ${_LANG} compiler" ON)
            if(TIMEMORY_INTEL_${_LANG}_AVX512)
                set(INTEL_${_LANG}_COMPILER_FLAGS "-xHOST -axMIC-AVX512")
            else(TIMEMORY_INTEL_${_LANG}_AVX512)
                set(INTEL_${_LANG}_COMPILER_FLAGS "-xHOST")
            endif(TIMEMORY_INTEL_${_LANG}_AVX512)
            add_feature(INTEL_${_LANG}_COMPILER_FLAGS "Intel ${_LANG} compiler flags")
        endif(CMAKE_${_LANG}_COMPILER_IS_INTEL)
    endforeach(_LANG C CXX)

    set(SANITIZE_TYPE leak CACHE STRING "-fsantitize=<TYPE>")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    if(WIN32)
        set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard")
    else(WIN32)
        set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
    endif(WIN32)

    if(GOOD_CMAKE)
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

# used by configure_package_*
set(LIBNAME timemory)

# set the compiler flags if not on Windows
if(NOT SUBPROJECT AND NOT WIN32)

    ############
    #    CXX   #
    ############
    add(CMAKE_CXX_FLAGS "-W -Wall -Wextra ${CXXFLAGS} $ENV{CXXFLAGS}")
    add(CMAKE_CXX_FLAGS "-Wno-unused-parameter -Wno-unknown-pragmas")
    add(CMAKE_CXX_FLAGS "-Wunused-but-set-parameter -Wno-unused-variable")

    if(NOT CMAKE_CXX_COMPILER_IS_INTEL)
        add(CMAKE_CXX_FLAGS "-faligned-new")
        add(CMAKE_CXX_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_CXX_FLAGS "-Wno-implicit-fallthrough")
        add(CMAKE_CXX_FLAGS "-Wno-shadow-field-in-constructor-modified")
        add(CMAKE_CXX_FLAGS "-Wno-exceptions")
        add(CMAKE_CXX_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_CXX_FLAGS "-Wno-unused-private-field")
    else(NOT CMAKE_CXX_COMPILER_IS_INTEL)
        # Intel compiler 18.0 sets -fno-protect-parens by default
        add(CMAKE_CXX_FLAGS "-fprotect-parens")
    endif(NOT CMAKE_CXX_COMPILER_IS_INTEL)

    ############
    #     C    #
    ############
    add(CMAKE_C_FLAGS "-W -Wall -Wextra ${CFLAGS} $ENV{CFLAGS}")
    add(CMAKE_C_FLAGS "-Wno-unused-parameter")
    add(CMAKE_C_FLAGS "-Wunused-but-set-parameter")
    add(CMAKE_C_FLAGS "-Wno-unused-variable")

    if(NOT CMAKE_C_COMPILER_IS_INTEL)
        add(CMAKE_C_FLAGS "-Wno-implicit-fallthrough")
    else(NOT CMAKE_C_COMPILER_IS_INTEL)
        # Intel compiler 18.0 sets -fno-protect-parens by default
        add(CMAKE_C_FLAGS "-fprotect-parens")
    endif(NOT CMAKE_C_COMPILER_IS_INTEL)

    ############
    #   other
    ############
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_definitions(-DDEBUG)
    else("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_definitions(-DNDEBUG)
    endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

    if(UNIX)
        add(CMAKE_C_FLAGS   "-pthread")
        add(CMAKE_CXX_FLAGS "-pthread")
    endif(UNIX)

    if(TIMEMORY_USE_SANITIZE)
        add_subfeature(TIMEMORY_USE_SANITIZE SANITIZE_TYPE "Sanitizer type")
        add(CMAKE_C_FLAGS   "-fsanitize=${SANITIZE_TYPE}")
        add(CMAKE_CXX_FLAGS "-fsanitize=${SANITIZE_TYPE}")
    endif(TIMEMORY_USE_SANITIZE)

    foreach(_LANG C CXX)
        if(CMAKE_${_LANG}_COMPILER_IS_INTEL)
            add(CMAKE_${_LANG}_FLAGS "${INTEL_${_LANG}_COMPILER_FLAGS}")
        endif(CMAKE_${_LANG}_COMPILER_IS_INTEL)
    endforeach(_LANG C CXX)

    add_c_flags(CMAKE_C_FLAGS       "${CMAKE_C_FLAGS}")
    add_cxx_flags(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS}")

elseif(NOT WIN32)

    #add_c_flags(CMAKE_C_FLAGS "${CFLAGS} $ENV{CFLAGS}")
    #add_cxx_flags(CMAKE_CXX_FLAGS "${CXXFLAGS} $ENV{CXXFLAGS}")

endif(NOT SUBPROJECT AND NOT WIN32)

set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")

if(TIMEMORY_EXCEPTIONS)
    add_definitions(-DTIMEMORY_EXCEPTIONS)
endif(TIMEMORY_EXCEPTIONS)
