# ########################################################################################
#
# timemory (Julia)
#
# ########################################################################################

if(DEFINED TIMEMORY_USE_JULIA AND NOT TIMEMORY_USE_JULIA)
    return()
endif()

find_package(Julia ${TIMEMORY_FIND_REQUIREMENT} ${TIMEMORY_FIND_QUIETLY})

if(NOT Julia_FOUND)
    return()
endif()

find_package(JuliaCxxWrap ${TIMEMORY_FIND_REQUIREMENT} ${TIMEMORY_FIND_QUIETLY})

if(NOT JuliaCxxWrap_FOUND)
    return()
endif()

target_include_directories(timemory-julia INTERFACE ${Julia_INCLUDE_DIRS}
                                                    ${JuliaCxxWrap_INCLUDE_DIRS})
target_link_libraries(timemory-julia INTERFACE ${JuliaCxxWrap_LIBRARIES})

# if set, will screw up loading library
unset(CMAKE_DEBUG_POSTFIX)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME julia)

# handle unity build for julia
option(TIMEMORY_BUILD_JULIA_UNITY "Build julia interface with unity build support"
       ${TIMEMORY_UNITY_BUILD})
mark_as_advanced(TIMEMORY_BUILD_JULIA_UNITY)
set(CMAKE_UNITY_BUILD ${TIMEMORY_BUILD_JULIA_UNITY})

set(_visibility timemory::timemory-default-visibility)
if(TIMEMORY_BUILD_JULIA_HIDDEN)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
    set(_visibility timemory::timemory-hidden-visibility)
endif()

# ########################################################################################

function(TIMEMORY_CONFIGURE_JULIA_TARGET _TARGET)

    add_library(timemory::${_TARGET} ALIAS ${_TARGET})
    target_link_libraries(${_TARGET} PRIVATE libtimemory-jl-interface)

    set(_SUBDIR ${ARGN})
    if(_SUBDIR)
        set(_SUBDIR "/${_SUBDIR}")
    endif()

    if(TIMEMORY_USE_CUDA)
        set(EXTRA_PROPERTIES LINKER_LANGUAGE CUDA)
    endif()

    set_target_properties(
        ${_TARGET}
        PROPERTIES PREFIX ""
                   SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/julia${_SUBDIR}
                   ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/julia${_SUBDIR}
                   RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/julia${_SUBDIR}
                   PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/julia${_SUBDIR}
                   INSTALL_RPATH_USE_LINK_PATH ON
                   ${EXTRA_PROPERTIES})

    if(UNIX AND NOT APPLE)
        set_target_properties(
            ${_TARGET}
            PROPERTIES
                INSTALL_RPATH
                "\$ORIGIN:\$ORIGIN/${LIB_RELPATH}:\$ORIGIN/../../../..:${CMAKE_INSTALL_RPATH}"
            )
    endif()

    install(
        TARGETS ${_TARGET}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/julia${_SUBDIR}
        OPTIONAL)

    if(NOT "${_TARGET}" STREQUAL "libtimemory-jl")
        add_dependencies(libtimemory-jl ${_TARGET})
    endif()
endfunction()

# ########################################################################################

if(NOT DEFINED TIMEMORY_USE_JULIA OR TIMEMORY_USE_JULIA)

    add_library(timemory-julia-compile-options INTERFACE)
    add_library(timemory::timemory-julia-compile-options ALIAS
                timemory-julia-compile-options)
    add_cxx_flag_if_avail("-frtti" timemory-julia-compile-options)
    add_cxx_flag_if_avail("-Wno-unused-value" timemory-julia-compile-options)
    add_cxx_flag_if_avail("-Wno-range-loop-analysis" timemory-julia-compile-options)
    add_cxx_flag_if_avail("-ftls-model=global-dynamic" timemory-julia-compile-options)
    add_cxx_flag_if_avail("-Wno-deprecated-declarations" timemory-julia-compile-options)
    add_cxx_flag_if_avail("-Wno-unused-but-set-parameter" timemory-julia-compile-options)

    set(jlsources ${CMAKE_CURRENT_LIST_DIR}/timemory-jl.cpp)

    set(_LINK shared)
    if((APPLE AND BUILD_STATIC_LIBS) OR (BUILD_STATIC_LIBS AND NOT BUILD_SHARED_LIBS))
        set(_LINK static)
    endif()

    add_library(libtimemory-jl-interface INTERFACE)
    target_link_libraries(
        libtimemory-jl-interface
        INTERFACE timemory::timemory-julia
                  timemory::timemory-headers
                  timemory::timemory-cxx-${_LINK}
                  timemory::timemory-compile-options
                  timemory::timemory-external-${_LINK}
                  timemory::timemory-mpip-library
                  timemory::timemory-ompt-library
                  timemory::timemory-ncclp-library
                  timemory::timemory-mallocp-library
                  timemory::timemory-julia-compile-options
                  ${_visibility})

    target_compile_definitions(libtimemory-jl-interface INTERFACE TIMEMORY_JULIA_SOURCE)

    add_library(libtimemory-jl SHARED ${TIMEMORY_EXCLUDE_FROM_ALL} ${jlsources})
    timemory_configure_julia_target(libtimemory-jl libs)

    configure_file(${CMAKE_CURRENT_LIST_DIR}/timemory.jl.in
                   ${PROJECT_BINARY_DIR}/julia/timemory.jl COPYONLY)
endif()
