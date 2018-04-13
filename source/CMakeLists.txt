
include_directories(${CMAKE_CURRENT_LIST_DIR})

################################################################################
#
#        Cereal
#
################################################################################

set(DEV_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})
# this gets annoying
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

# add cereal
if(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)
    add_subdirectory(cereal)
endif(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

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
# this is only needed for windows
file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/python/*.cpp)

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
                POSITION_INDEPENDENT_CODE ON)
        else("${LIB_TYPE}" STREQUAL "SHARED")
            set_target_properties(${OUTPUT_NAME}-${TYPE} PROPERTIES
                OUTPUT_NAME lib${OUTPUT_NAME}-archive
				COMPILE_DEFINITIONS _TIMEMORY_ARCHIVE
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

if(WIN32)
	# directly compile sources
	set(LIBRARY_SOURCES ${sources} ${headers})

	# Windows DLLs restrict the code from compiling both
	if(TIMEMORY_DYNAMIC_LINK)

		set_source_files_properties(${LIBRARY_SOURCES} ${pysources} PROPERTIES
			COMPILE_DEFINITIONS _TIMEMORY_DLL)

		_build_library(shared ${LIBNAME} ${LIBRARY_SOURCES})

	else(TIMEMORY_DYNAMIC_LINK)

		set_source_files_properties(${LIBRARY_SOURCES} ${pysources} PROPERTIES
			COMPILE_DEFINITIONS _TIMEMORY_ARCHIVE)

		_build_library(static ${LIBNAME} ${LIBRARY_SOURCES})

	endif(TIMEMORY_DYNAMIC_LINK)

else(WIN32)

	if(XCODE)
		# directly compile sources
		set(LIBRARY_SOURCES ${sources} ${headers})
	else(XCODE)
		# build an object library
		add_library(${LIBNAME}-object OBJECT ${sources} ${headers})
		set(LIBRARY_SOURCES $<TARGET_OBJECTS:${LIBNAME}-object>)
	endif(XCODE)

	_build_library(shared ${LIBNAME} ${LIBRARY_SOURCES})
	_build_library(static ${LIBNAME} ${LIBRARY_SOURCES})

endif(WIN32)


#------------------------------------------------------------------------------#
# Install the targets and export
#
if(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)

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

endif(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)


#------------------------------------------------------------------------------#
# Python bindings
#
if(TIMEMORY_USE_PYTHON_BINDING)

    add_subdirectory(python)

    if(TIMEMORY_DYNAMIC_LINK)
        set(LINK_TYPE shared)
        install(TARGETS ${LIBNAME}-shared
            DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib
            COMPONENT python)
    endif(TIMEMORY_DYNAMIC_LINK)

endif(TIMEMORY_USE_PYTHON_BINDING)

#------------------------------------------------------------------------------#
# install the plotting.py module as a Python executable
# named 'timemory-plotter' as C++ JSON outputs can use this
# to generate plots
#
configure_file(${PROJECT_SOURCE_DIR}/timemory/plotting/plotting.py
    ${PROJECT_BINARY_DIR}/timemory-plotter COPYONLY)

install(FILES ${PROJECT_BINARY_DIR}/timemory-plotter
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development
    PERMISSIONS
    OWNER_EXECUTE OWNER_READ OWNER_WRITE
    GROUP_EXECUTE GROUP_READ
    WORLD_EXECUTE WORLD_READ)
