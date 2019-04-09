
################################################################################
#
#        TiMemory Options
#
################################################################################

include(MacroUtilities)

set(SANITIZER_TYPE leak CACHE STRING "Sanitizer type")
set(_USE_PAPI OFF)
if(UNIX AND NOT APPLE)
    set(_USE_PAPI ON)
endif()

# CMake options
add_option(CMAKE_C_STANDARD_REQUIRED "Require C standard" ON)
add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ standard" ON)
add_option(CMAKE_CXX_EXTENSIONS "Build with CXX extensions (e.g. gnu++11)" OFF)
add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Embed RPATH using link path" ON)
add_option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Build settings
add_option(TIMEMORY_DEVELOPER_INSTALL "Python developer installation from setup.py" OFF)
add_option(TIMEMORY_DOXYGEN_DOCS "Make a `doc` make target" OFF)
add_option(TIMEMORY_BUILD_TESTING "Build testing for dashboard" OFF)
add_option(TIMEMORY_BUILD_EXAMPLES "Build the examples" ${TIMEMORY_BUILD_TESTING})
add_option(TIMEMORY_BUILD_C "Build the C compatible library" ON)
add_option(TIMEMORY_BUILD_PYTHON "Build Python binds for ${PROJECT_NAME}" ON)

# Features
add_feature(CMAKE_C_STANDARD "C language standard")
add_feature(CMAKE_CXX_STANDARD "C++ language standard")
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
add_feature(${PROJECT_NAME}_CXX_FLAGS "C++ compiler flags")
add_feature(TIMEMORY_INSTALL_PREFIX "${PROJECT_NAME} installation")
add_feature(${PROJECT_NAME}_DEFINITIONS "${PROJECT_NAME} compile definitions")

# TiMemory options
add_option(TIMEMORY_USE_EXCEPTIONS "Signal handler throws exceptions (default: exit)" OFF)
add_option(TIMEMORY_USE_MPI "Enable MPI usage" ON)
add_option(TIMEMORY_USE_SANITIZER "Enable -fsanitize flag (=${SANITIZER_TYPE})" OFF)
add_option(TIMEMORY_USE_PAPI "Enable PAPI" ${_USE_PAPI})
add_option(TIMEMORY_USE_FILTERING "Enable filtering out types not implemented" ON)

if(TIMEMORY_USE_MPI)
    add_option(TIMEMORY_TEST_MPI "Enable MPI tests" ON)
endif(TIMEMORY_USE_MPI)

# cereal options
add_option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF NO_FEATURE)
add_option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON NO_FEATURE)
add_option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON NO_FEATURE)
add_option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON NO_FEATURE)

if(TIMEMORY_DOXYGEN_DOCS)
    add_option(TIMEMORY_BUILD_DOXYGEN "Include `doc` make target in all" OFF NO_FEATURE)
    mark_as_advanced(TIMEMORY_BUILD_DOXYGEN)
endif()

set(CTEST_SITE "${HOSTNAME}" CACHE STRING "CDash submission site")
set(CTEST_MODEL "Continuous" CACHE STRING "CDash submission track")

if(TIMEMORY_BUILD_TESTING)
    # if this is directory we are running CDash (don't set to ON)
    add_option(TIMEMORY_DASHBOARD_MODE
        "Internally used to skip generation of CDash files" OFF NO_FEATURE)
    mark_as_advanced(TIMEMORY_DASHBOARD_MODE)
    add_feature(CTEST_MODEL "CDash submission track")
    add_feature(CTEST_SITE "CDash submission site")

    if(NOT TIMEMORY_DASHBOARD_MODE)
        add_option(CTEST_LOCAL_CHECKOUT
            "Use the local source tree for CTest/CDash" OFF NO_FEATURE)
    endif()
endif()

if(TIMEMORY_BUILD_PYTHON)
    set(PYBIND11_INSTALL OFF CACHE BOOL "Don't install Pybind11")
endif()

if(TIMEMORY_USE_EXCEPTIONS)
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TIMEMORY_EXCEPTIONS)
endif()

if(TIMEMORY_USE_FILTERING)
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TIMEMORY_USE_FILTERING)
endif()
