
################################################################################
#
#        TiMemory Options
#
################################################################################

include(MacroUtilities)

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(SUBPROJECT OFF CACHE INTERNAL "TiMemory is not a subproject" FORCE)
else("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(SUBPROJECT ON  CACHE INTERNAL "TiMemory is a subproject" FORCE)
endif("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

add_option(TIMEMORY_EXCEPTIONS "Signal handler throws exceptions (default: exit)" OFF)
add_option(TIMEMORY_USE_MPI "Enable MPI usage" ON)
add_option(TIMEMORY_USE_PYTHON_BINDING "Build Python binds for TiMemory" ON)
add_option(TIMEMORY_SETUP_PY "Python build from setup.py" OFF NO_FEATURE)
add_option(TIMEMORY_DEVELOPER_INSTALL "Python developer installation from setup.py" OFF)
add_option(TIMEMORY_BUILD_TESTING "Build testing for dashboard" OFF NO_FEATURE)
add_option(TIMEMORY_DOXYGEN_DOCS "Make a `doc` make target" OFF)
add_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
add_dependent_option(TIMEMORY_BUILD_EXAMPLES "Build the C++ examples"
    ON "TIMEMORY_BUILD_TESTING" OFF)
if(TIMEMORY_USE_MPI)
    add_option(TIMEMORY_TEST_MPI "Enable MPI tests" ON)
endif(TIMEMORY_USE_MPI)
# cereal options
add_option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF NO_FEATURE)
add_option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON NO_FEATURE)
add_option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON NO_FEATURE)
add_option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON NO_FEATURE)
# CMake options
add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ standard" ON)
add_option(CMAKE_CXX_EXTENSIONS "Build with CXX extensions (e.g. gnu++11)" OFF)
# features
add_feature(TIMEMORY_INSTALL_PREFIX "TiMemory installation")
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(CMAKE_C_FLAGS "C compiler flags")
add_feature(CMAKE_CXX_FLAGS "C++ compiler flags")
add_feature(CMAKE_CXX_STANDARD "C++11 STL standard")

if(NOT SUBPROJECT)
    add_option(TIMEMORY_USE_SANITIZE "Enable -fsanitize flag (=${SANITIZE_TYPE})" OFF)
endif(NOT SUBPROJECT)

if(TIMEMORY_DOXYGEN_DOCS)
    add_option(TIMEMORY_BUILD_DOXYGEN_DOCS "Include `doc` make target in all" OFF)
    mark_as_advanced(TIMEMORY_BUILD_DOXYGEN_DOCS)
endif(TIMEMORY_DOXYGEN_DOCS)

mark_as_advanced(TIMEMORY_BUILD_TESTING)

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
    endif(NOT TIMEMORY_DASHBOARD_MODE)
endif(TIMEMORY_BUILD_TESTING)

if(TIMEMORY_USE_PYTHON_BINDING)

    if(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)
        add_dependent_option(PYBIND11_INSTALL "PyBind11 installation" OFF
            "TIMEMORY_DEVELOPER_INSTALL" ON)
    else(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)
        set(PYBIND11_INSTALL OFF CACHE BOOL "Don't install Pybind11" FORCE)
    endif(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)

endif(TIMEMORY_USE_PYTHON_BINDING)

