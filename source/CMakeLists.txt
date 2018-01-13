
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

add_library(${LIBNAME}-object OBJECT ${sources} ${headers})
add_library(${LIBNAME}-shared SHARED $<TARGET_OBJECTS:${LIBNAME}-object>)
add_library(${LIBNAME}-static STATIC $<TARGET_OBJECTS:${LIBNAME}-object>)
target_link_libraries(${LIBNAME}-shared ${EXTERNAL_LIBRARIES})
target_link_libraries(${LIBNAME}-static ${EXTERNAL_LIBRARIES})

set_target_properties(${LIBNAME}-shared PROPERTIES
    OUTPUT_NAME ${LIBNAME}
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
set_target_properties(${LIBNAME}-static PROPERTIES
    OUTPUT_NAME ${LIBNAME}
    VERSION ${PROJECT_VERSION})

#----------------------------------------------------------------------------
# PyBind11
#
pybind11_add_module(${LIBNAME} ${CMAKE_CURRENT_LIST_DIR}/${LIBNAME}.cc)
target_link_libraries(${LIBNAME} PRIVATE ${LIBNAME}-shared)

#----------------------------------------------------------------------------
# Install the targets and export
#
install(TARGETS ${LIBNAME}-shared ${LIBNAME}-static
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}LibraryDepends)
install(FILES ${headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/timemory)
install(TARGETS ${LIBNAME}
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR})
# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}LibraryDepends
    DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
    COMPONENT dev)
