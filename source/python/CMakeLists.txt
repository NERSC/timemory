
##########################################################################################
#
#        timemory (Python)
#
##########################################################################################

if(TIMEMORY_BUILD_PYTHON)
    set(CMAKE_CXX_CLANG_TIDY)
    set(CMAKE_UNITY_BUILD OFF)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME python)

    add_library(timemory-python-compile-options INTERFACE)
    add_library(${PROJECT_NAME}::timemory-python-compile-options ALIAS
        timemory-python-compile-options)
    add_cxx_flag_if_avail("-frtti" timemory-python-compile-options)
    add_cxx_flag_if_avail("-ftls-model=global-dynamic" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-deprecated-declarations" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-unused-but-set-parameter" timemory-python-compile-options)

    file(GLOB_RECURSE pybind_headers ${PROJECT_SOURCE_DIR}/external/pybind11/include/*.h)
    file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/libpytimemory*.hpp)
    file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/libpytimemory*.cpp)

    if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER AND TIMEMORY_BUILD_PYTHON_CUDA)
        set_source_files_properties(${pysources} PROPERTIES
            LANGUAGE CUDA
            LINKER_LANGUAGE CUDA)
    endif()

    add_library(libpytimemory MODULE ${pysources} ${pyheaders} ${pybind_headers})
    add_library(${PROJECT_NAME}::libpytimemory ALIAS libpytimemory)
    target_compile_definitions(libpytimemory PRIVATE TIMEMORY_PYBIND11_SOURCE)

    target_include_directories(libpytimemory SYSTEM PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/pybind11/include>)

    set(pybind_libs pybind11::module)
    if(TIMEMORY_BUILD_LTO)
        list(APPEND pybind_libs pybind11::thin_lto)
    endif()

    target_link_libraries(libpytimemory PRIVATE
        ${pybind_libs}
        timemory::timemory-python
        timemory::timemory-headers
        timemory::timemory-cxx-shared
        timemory::timemory-compile-options
        timemory::timemory-external-shared
        timemory::timemory-mpip-library
        timemory::timemory-ompt-library
        timemory::timemory-ncclp-library
        timemory::timemory-python-compile-options
        timemory::timemory-default-visibility)

    if(TIMEMORY_USE_CUDA)
        set(EXTRA_PROPERTIES LINKER_LANGUAGE CUDA)
    endif()

    set_target_properties(libpytimemory PROPERTIES
        PREFIX                      ""
        SUFFIX                      "${PYTHON_MODULE_EXTENSION}"
        LIBRARY_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/timemory
        ARCHIVE_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/timemory
        RUNTIME_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/timemory
        PDB_OUTPUT_DIRECTORY        ${PROJECT_BINARY_DIR}/timemory
        INSTALL_RPATH_USE_LINK_PATH ON
        ${EXTRA_PROPERTIES})

    set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})
    if(NOT IS_ABSOLUTE "${_PYLIB}")
        set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
    endif()
    file(RELATIVE_PATH LIB_RELPATH "${_PYLIB}"
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    # message(STATUS "RELATIVE PATH: ${LIB_RELPATH}")

    if(UNIX AND NOT APPLE)
        set_target_properties(libpytimemory PROPERTIES INSTALL_RPATH
            "\$ORIGIN/${LIB_RELPATH}:\$ORIGIN/${LIB_RELPATH}/timemory:\$ORIGIN/../../..:\$ORIGIN/../../../timemory:\$ORIGIN:${CMAKE_INSTALL_RPATH}")
    endif()

    install(TARGETS libpytimemory
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})

    message(STATUS "Adding source/python/tools...")
    add_subdirectory(tools)
endif()
