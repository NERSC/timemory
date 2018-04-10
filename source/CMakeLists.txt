
include_directories(${CMAKE_CURRENT_LIST_DIR})

#if(TIMEMORY_DYNAMIC_LINK)
#    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_FULL_LIBDIR})
#    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
#endif(TIMEMORY_DYNAMIC_LINK)

################################################################################
#
#        Cereal
#
################################################################################

# cereal options
add_option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF NO_FEATURE)
add_option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON NO_FEATURE)
add_option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON NO_FEATURE)
add_option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON NO_FEATURE)

set(DEV_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})
# this gets annoying
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

# add cereal
if(NOT SETUP_PY)
    add_subdirectory(cereal)
endif(NOT SETUP_PY)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)
#message(STATUS "Dev warnings: ${DEV_WARNINGS}")
include_directories(${CMAKE_CURRENT_LIST_DIR}/cereal/include)

################################################################################
#
#        TiMemory (C++)
#
################################################################################

#------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)

# libraries to install
set(INSTALL_LIBRARIES )

#------------------------------------------------------------------------------#
# macro to build a library of type: shared, static
macro(_BUILD_LIBRARY TYPE OUTPUT_NAME)

    string(TOUPPER "${TYPE}" LIB_TYPE)
    add_library(${OUTPUT_NAME}-${TYPE} ${LIB_TYPE} ${ARGN})
    target_link_libraries(${OUTPUT_NAME}-${TYPE} PUBLIC ${EXTERNAL_LIBRARIES})

    # Windows
	if(WIN32)

        if("${LIB_TYPE}" STREQUAL "SHARED")
            set_target_properties(${OUTPUT_NAME}-${TYPE} PROPERTIES
                OUTPUT_NAME lib${OUTPUT_NAME}
                COMPILE_DEFINITIONS _TIMEMORY_DLL
                LANGUAGE CXX
                # C++ shared library fallback installation
                LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory/lib
                POSITION_INDEPENDENT_CODE ON)
        else("${LIB_TYPE}" STREQUAL "SHARED")
            set_target_properties(${OUTPUT_NAME}-${TYPE} PROPERTIES
                OUTPUT_NAME lib${OUTPUT_NAME}
                COMPILE_DEFINITIONS _TIMEMORY_STATIC
                LANGUAGE CXX
                POSITION_INDEPENDENT_CODE ON)
        endif("${LIB_TYPE}" STREQUAL "SHARED")
        #target_compile_options(${OUTPUT_NAME}-${TYPE} PRIVATE /MP /bigobj /Tp)

	else(WIN32)

        if(XCODE)
            # xcode has issue with "cmake_symlink_library"
            set_target_properties(${OUTPUT_NAME}-${TYPE} PROPERTIES
                OUTPUT_NAME ${OUTPUT_NAME}
                # C++ shared library fallback installation
                LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory/lib
                POSITION_INDEPENDENT_CODE ON)
        else(XCODE)
            set_target_properties(${OUTPUT_NAME}-${TYPE} PROPERTIES
                OUTPUT_NAME ${OUTPUT_NAME}
                VERSION ${PROJECT_VERSION}
                SOVERSION ${PROJECT_VERSION_MAJOR}
                # C++ shared library fallback installation
                LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory/lib
                POSITION_INDEPENDENT_CODE ON)
        endif(XCODE)

    endif(WIN32)

    list(APPEND INSTALL_LIBRARIES ${LIBNAME}-${TYPE})

endmacro(_BUILD_LIBRARY TYPE OUTPUT_NAME)
#------------------------------------------------------------------------------#


# this includes compile definitions for headers
add_subdirectory(timemory)

if(NOT XCODE AND NOT WIN32)
    message(STATUS "\nBuilding ${PROJECT_NAME} with object library...\n")
    # build an object library
    add_library(${LIBNAME}-object OBJECT ${sources} ${headers})
    set(LIBRARY_SOURCES $<TARGET_OBJECTS:${LIBNAME}-object>)
else(NOT XCODE AND NOT WIN32)
    # directly compile sources
    set(LIBRARY_SOURCES ${sources} ${headers})
endif(NOT XCODE AND NOT WIN32)

_build_library(shared ${LIBNAME} ${LIBRARY_SOURCES})
_build_library(static ${LIBNAME} ${LIBRARY_SOURCES})

#------------------------------------------------------------------------------#
# Install the targets and export
#
if(NOT SETUP_PY OR PYTHON_DEVELOPER_INSTALL)
    # C++ compiled libraries
    install(TARGETS ${INSTALL_LIBRARIES}
        DESTINATION ${TIMEMORY_INSTALL_LIBDIR}
        EXPORT ${PROJECT_NAME}LibraryDepends
        COMPONENT development)

    # C++ development headers
    install(FILES ${headers}
        DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory
        COMPONENT development)

    # Install the export set for use with the install-tree
    install(EXPORT ${PROJECT_NAME}LibraryDepends
        DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR}
        COMPONENT development)

    export(TARGETS ${INSTALL_LIBRARIES}
        FILE ${CMAKE_BINARY_DIR}/TiMemoryBuild.cmake)

endif(NOT SETUP_PY OR PYTHON_DEVELOPER_INSTALL)


#------------------------------------------------------------------------------#
# Python bindings
#
if(TIMEMORY_PYTHON_BINDING)

    add_subdirectory(python)

    if(TIMEMORY_DYNAMIC_LINK)
        set(LINK_TYPE shared)
        install(TARGETS ${LIBNAME}-shared
            DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib
            COMPONENT python)
    endif(TIMEMORY_DYNAMIC_LINK)

endif(TIMEMORY_PYTHON_BINDING)

