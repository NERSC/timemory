
# windows not yet implemented
if(WIN32 OR NOT TIMEMORY_BUILD_TOOLS OR NOT TIMEMORY_BUILD_COMPILER_INSTRUMENTATION)
    return()
endif()

cmake_policy(PUSH)
cmake_policy(SET CMP0063 NEW)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
unset(CMAKE_C_VISIBILITY_PRESET)
unset(CMAKE_CXX_VISIBILITY_PRESET)

foreach(LIB_TYPE shared static)

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

    # don't allow in-source instrumentation
    target_compile_definitions(${LIB_TARGET} INTERFACE
        DISABLE_TIMEMORY
        TIMEMORY_COMPILER_INSTRUMENTATION)

    target_compile_definitions(${LIB_TARGET} PRIVATE
        TIMEMORY_COMPILER_INSTRUMENTATION_IMPL)

    target_link_libraries(${LIB_TARGET} PUBLIC
        ${dl_LIBRARY})

    target_link_libraries(${LIB_TARGET} PRIVATE
        timemory::timemory-headers
        timemory::timemory-compile-options
        timemory::timemory-hidden-visibility)

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
        target_compile_options(${LIB_TARGET} INTERFACE
            $<$<COMPILE_LANGUAGE:C>:-g>
            $<$<COMPILE_LANGUAGE:CXX>:-g>
            $<$<COMPILE_LANGUAGE:C>:-fno-omit-frame-pointer>
            $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
            $<$<COMPILE_LANGUAGE:C>:-fno-optimize-sibling-calls>
            $<$<COMPILE_LANGUAGE:CXX>:-fno-optimize-sibling-calls>
            $<$<COMPILE_LANGUAGE:C>:-finstrument-functions>
            $<$<COMPILE_LANGUAGE:CXX>:-finstrument-functions>)

        target_compile_options(${LIB_TARGET} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-fvisibility=hidden>
            $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
            $<$<COMPILE_LANGUAGE:C>:-fvisibility-inlines-hidden>
            $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>)
    endif()

    install(TARGETS ${LIB_TARGET}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/compiler
      EXPORT timemory-library-depends)

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
    ")

endforeach()

cmake_policy(POP)
