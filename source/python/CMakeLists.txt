
##########################################################################################
#
#        timemory (Python)
#
##########################################################################################

# if set, will screw up loading library
unset(CMAKE_DEBUG_POSTFIX)

set(_visibility timemory::timemory-default-visibility)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME python)

if(TIMEMORY_BUILD_PYTHON_HIDDEN)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
    set(_visibility timemory::timemory-hidden-visibility)
endif()

##########################################################################################

function(TIMEMORY_CONFIGURE_PYTARGET _TARGET)

    add_library(timemory::${_TARGET} ALIAS ${_TARGET})
    target_link_libraries(${_TARGET} PRIVATE libpytimemory-interface)

    set(_SUBDIR ${ARGN})
    if(_SUBDIR)
        set(_SUBDIR "/${_SUBDIR}")
    endif()

    if(TIMEMORY_USE_CUDA)
        set(EXTRA_PROPERTIES LINKER_LANGUAGE CUDA)
    endif()

    set_target_properties(${_TARGET} PROPERTIES
        PREFIX                      ""
        SUFFIX                      "${PYTHON_MODULE_EXTENSION}"
        LIBRARY_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/timemory${_SUBDIR}
        ARCHIVE_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/timemory${_SUBDIR}
        RUNTIME_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/timemory${_SUBDIR}
        PDB_OUTPUT_DIRECTORY        ${PROJECT_BINARY_DIR}/timemory${_SUBDIR}
        INSTALL_RPATH_USE_LINK_PATH ON
        ${EXTRA_PROPERTIES})

    set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/timemory${_SUBDIR})
    if(NOT IS_ABSOLUTE "${_PYLIB}")
        set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
    endif()

    file(RELATIVE_PATH LIB_RELPATH "${_PYLIB}"
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

    if(UNIX AND NOT APPLE)
        set_target_properties(${_TARGET} PROPERTIES INSTALL_RPATH
            "\$ORIGIN/${LIB_RELPATH}:\$ORIGIN/${LIB_RELPATH}/timemory${_SUBDIR}:\$ORIGIN/../../..:\   $ORIGIN/../../../timemory${_SUBDIR}:\$ORIGIN:${CMAKE_INSTALL_RPATH}")
    endif()

    install(TARGETS ${_TARGET}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/timemory${_SUBDIR})

    if(NOT "${_TARGET}" STREQUAL "libpytimemory")
        add_dependencies(libpytimemory ${_TARGET})
    endif()
endfunction()

##########################################################################################

if(TIMEMORY_BUILD_PYTHON)

    add_library(timemory-python-compile-options INTERFACE)
    add_library(timemory::timemory-python-compile-options ALIAS
        timemory-python-compile-options)
    add_cxx_flag_if_avail("-frtti" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-unused-value" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-range-loop-analysis" timemory-python-compile-options)
    add_cxx_flag_if_avail("-ftls-model=global-dynamic" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-deprecated-declarations" timemory-python-compile-options)
    add_cxx_flag_if_avail("-Wno-unused-but-set-parameter" timemory-python-compile-options)

    file(GLOB_RECURSE pybind_headers ${PROJECT_SOURCE_DIR}/external/pybind11/include/*.h)
    file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/libpytimemory*.hpp)
    file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/libpytimemory*.cpp)
    set(pysources
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-api.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-auto-timer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-component-bundle.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-component-list.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-components.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-enumeration.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-hardware-counters.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-profile.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-rss-usage.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-settings.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-signals.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-trace.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory-units.cpp
        ${CMAKE_CURRENT_LIST_DIR}/libpytimemory.cpp)

    if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER AND TIMEMORY_BUILD_PYTHON_CUDA)
        set_source_files_properties(${pysources} PROPERTIES
            LANGUAGE CUDA
            LINKER_LANGUAGE CUDA)
    endif()

    set(pybind_libs pybind11::module)
    if(TIMEMORY_BUILD_LTO)
        list(APPEND pybind_libs pybind11::thin_lto)
    endif()

    add_library(libpytimemory-interface INTERFACE)
    target_link_libraries(libpytimemory-interface INTERFACE
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
        ${_visibility})

    target_compile_definitions(libpytimemory-interface INTERFACE
        TIMEMORY_PYBIND11_SOURCE)

    target_include_directories(libpytimemory-interface SYSTEM INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/pybind11/include>)

    add_library(libpytimemory MODULE ${pysources} ${pyheaders} ${pybind_headers})
    timemory_configure_pytarget(libpytimemory)

    message(STATUS "Adding source/python/tools...")
    add_subdirectory(tools)
endif()
