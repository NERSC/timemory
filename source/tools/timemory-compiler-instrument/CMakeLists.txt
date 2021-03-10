
# windows not yet implemented
if(WIN32 OR NOT TIMEMORY_BUILD_COMPILER_INSTRUMENTATION)
    return()
endif()

cmake_policy(PUSH)
cmake_policy(SET CMP0063 NEW)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # ensure always pic
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_UNITY_BUILD OFF)

set(LINK_LIBS)
if(dl_LIBRARY)
    list(APPEND LINK_LIBS ${dl_LIBRARY})
endif()
if(rt_LIBRARY)
    list(APPEND LINK_LIBS ${rt_LIBRARY})
endif()

#----------------------------------------------------------------------------------------#

function(strip_target LIB_TARGET LIB_TYPE LIB_FUNC)
    if(APPLE)
        if("${LIB_TYPE}" STREQUAL "shared")
            set(_link_flags
                -exported_symbol=_${LIB_FUNC}_enter
                -exported_symbol=_${LIB_FUNC}_exit)
            if(NOT CMAKE_VERSION VERSION_LESS 3.13)
                target_link_options(${LIB_TARGET} PRIVATE ${_link_flags})
            else()
                target_compile_options(${LIB_TARGET} PRIVATE
                    $<LINK_ONLY:${_link_flags}>)
            endif()
        endif()
        set(_FILE ${CMAKE_CURRENT_BINARY_DIR}/${LIB_TARGET}-symbols.txt)
        file(WRITE ${_FILE} "_${LIB_FUNC}_enter\n_${LIB_FUNC}_exit\n")
        set(_STRIP_ARGS -i -S -x -X -N)
        add_custom_command(TARGET ${LIB_TARGET}
            POST_BUILD
            COMMAND ${CMAKE_STRIP}
                -s ${_FILE} ${_STRIP_ARGS}
                $<TARGET_FILE:${LIB_TARGET}>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    else()
        add_custom_command(TARGET ${LIB_TARGET}
            POST_BUILD
            COMMAND ${CMAKE_STRIP} -s
                --keep-symbol=${LIB_FUNC}_enter
                --keep-symbol=${LIB_FUNC}_exit
                --keep-symbol=_${LIB_FUNC}_enter
                --keep-symbol=_${LIB_FUNC}_exit
                $<TARGET_FILE:${LIB_TARGET}>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()

endfunction()

#----------------------------------------------------------------------------------------#

set(LIB_TARGET timemory-compiler-instrument-base)
set(LIB_TYPE STATIC)

build_library(
    PIC
    NO_CACHE_LIST
    TYPE                ${LIB_TYPE}
    TARGET_NAME         ${LIB_TARGET}
    OUTPUT_NAME         ${LIB_TARGET}
    LANGUAGE            CXX
    LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
    OUTPUT_DIR          ${PROJECT_BINARY_DIR}/compiler
    SOURCES             ${CMAKE_CURRENT_LIST_DIR}/compiler-instrument-base.cpp)

target_compile_definitions(${LIB_TARGET} PRIVATE
    TIMEMORY_COMPILER_INSTRUMENTATION)

target_link_libraries(${LIB_TARGET} PUBLIC
    timemory::timemory-dmp
    timemory::timemory-threading
    ${LINK_LIBS})

# only papi and gotcha are useful TPLs here
target_link_libraries(${LIB_TARGET} PRIVATE
    timemory::timemory-headers
    timemory::timemory-papi
    timemory::timemory-gotcha
    timemory::timemory-statistics
    timemory::timemory-compile-options
    timemory::timemory-hidden-visibility)

add_cxx_flag_if_avail("-ftls-model=local-exec" ${LIB_TARGET})

# strip_target(${LIB_TARGET} ${LIB_TYPE} "timemory_profile_func")

install(TARGETS ${LIB_TARGET}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/compiler
    OPTIONAL)

# build tree
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        compiler/libtimemory-compiler-instrument${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}
        libtimemory-compiler-instrument${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

# install tree
install(CODE "
EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        timemory/compiler/libtimemory-compiler-instrument${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}
        libtimemory-compiler-instrument${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
"
OPTIONAL)

if(NOT TARGET timemory::${LIB_TARGET})
    add_library(timemory::${LIB_TARGET} ALIAS ${LIB_TARGET})
endif()

#----------------------------------------------------------------------------------------#

set(_LIB_TYPES shared)

foreach(LIB_TYPE ${_LIB_TYPES})

    set(LIB_TARGET timemory-compiler-instrument-${LIB_TYPE})
    string(TOUPPER "${LIB_TYPE}" UC_LIB_TYPE)

    build_library(
        PIC
        NO_CACHE_LIST
        TYPE                ${UC_LIB_TYPE}
        TARGET_NAME         ${LIB_TARGET}
        OUTPUT_NAME         timemory-compiler-instrument
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}/compiler
        SOURCES             ${CMAKE_CURRENT_LIST_DIR}/compiler-instrument.cpp)

    target_link_libraries(${LIB_TARGET} PUBLIC
        timemory::timemory-dmp
        timemory::timemory-threading)

    target_link_libraries(${LIB_TARGET} PRIVATE
        timemory-compiler-instrument-base
        timemory::timemory-compile-options
        timemory::timemory-hidden-visibility
        )

    if(WIN32)
        # not yet implemented
        target_compile_options(${LIB_TARGET} INTERFACE
            $<$<COMPILE_LANGUAGE:C>:/Z7>
            $<$<COMPILE_LANGUAGE:CXX>:/Z7>
            $<$<COMPILE_LANGUAGE:C>:/DEBUG>
            $<$<COMPILE_LANGUAGE:CXX>:/DEBUG>
            $<$<COMPILE_LANGUAGE:C>:/GH>
            $<$<COMPILE_LANGUAGE:CXX>:/GH>
            $<$<COMPILE_LANGUAGE:C>:/Gh>
            $<$<COMPILE_LANGUAGE:CXX>:/Gh>)

        set_target_properties(${LIB_TARGET} PROPERTIES
            INTERFACE_LINK_OPTIONS "/DEBUG")
    else()
        target_link_libraries(${LIB_TARGET} INTERFACE
            timemory::timemory-compile-debuginfo
            timemory::timemory-instrument-functions)
    endif()

    if(NOT LINUX AND NOT "${LIB_TYPE}" STREQUAL "static")
        # strip_target(${LIB_TARGET} ${LIB_TYPE} "__cyg_profile_func")
    endif()

    install(TARGETS ${LIB_TARGET}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/compiler
        OPTIONAL)

    # build tree
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            compiler/libtimemory-compiler-instrument${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
            libtimemory-compiler-instrument${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # install tree
    install(CODE "
EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        timemory/compiler/libtimemory-compiler-instrument${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
        libtimemory-compiler-instrument${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    "
    OPTIONAL)

    if(NOT TARGET timemory::${LIB_TARGET})
        add_library(timemory::${LIB_TARGET} ALIAS ${LIB_TARGET})
    endif()
endforeach()

cmake_policy(POP)
