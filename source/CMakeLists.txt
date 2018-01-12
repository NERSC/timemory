
include(GNUInstallDirs)

include_directories(${CMAKE_CURRENT_LIST_DIR})

################################################################################
#
#        Cereal
#
################################################################################

# cereal options
option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF)
option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON)
option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON)
option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON)

set(DEV_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})
# this gets annoying
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

# add cereal
add_subdirectory(cereal)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

include_directories(${CMAKE_CURRENT_LIST_DIR}/cereal/include)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)

add_library(timemory-object OBJECT ${sources} ${headers})
add_library(timemory-shared SHARED $<TARGET_OBJECTS:timemory-object>)
add_library(timemory-static STATIC $<TARGET_OBJECTS:timemory-object>)
target_link_libraries(timemory-shared ${EXTERNAL_LIBRARIES})
target_link_libraries(timemory-static ${EXTERNAL_LIBRARIES})

set_target_properties(timemory-shared PROPERTIES
    OUTPUT_NAME timemory
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
set_target_properties(timemory-static PROPERTIES
    OUTPUT_NAME timemory
    VERSION ${PROJECT_VERSION})

#----------------------------------------------------------------------------
# PyBind11
#
pybind11_add_module(timemory ${CMAKE_CURRENT_LIST_DIR}/timemory.cc)
target_link_libraries(timemory PRIVATE timemory-shared)

if(NOT PYBIND11_PYTHON_VERSION)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} --version
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE "[ A-Za-z]" "" PYTHON_VERSION "${PYTHON_VERSION}")
    string(REGEX REPLACE "\.([0-9]+)$" "" PYTHON_VERSION "${PYTHON_VERSION}")
    set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}"
        CACHE STRING "Python version" FORCE)
endif(NOT PYBIND11_PYTHON_VERSION)

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS timemory-shared timemory-static 
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/timemory)
install(TARGETS timemory
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${PYBIND11_PYTHON_VERSION}/site-packages)
