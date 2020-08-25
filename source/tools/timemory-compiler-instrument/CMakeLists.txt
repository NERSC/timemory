
if(WIN32 OR NOT TIMEMORY_BUILD_TOOLS)
    return()
endif()

foreach(LIB_TYPE shared static)

    set(LIB_TARGET timemory-compiler-instrument-${LIB_TYPE})
    string(TOUPPER "${LIB_TYPE}" UC_LIB_TYPE)

    if(TARGET timemory-cxx-${LIB_TYPE})
        set(_LIBS timemory-cxx-${LIB_TYPE})
    else()
        set(_LIBS
            ${TIMEMORY_${UC_LIB_TYPE}_GLOBAL_LIBRARIES}
            ${TIMEMORY_${UC_LIB_TYPE}_FACTORY_LIBRARIES}
            ${TIMEMORY_${UC_LIB_TYPE}_COMPONENT_LIBRARIES}
            ${TIMEMORY_${UC_LIB_TYPE}_USER_BUNDLE_LIBRARIES}
            timemory-external-${LIB_TYPE})
    endif()

    build_library(
        PIC
        NO_CACHE_LIST
        TYPE                ${UC_LIB_TYPE}
        TARGET_NAME         timemory-compiler-instrument-${LIB_TYPE}
        OUTPUT_NAME         timemory-compiler-instrument
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}/compiler
        SOURCES             ${CMAKE_CURRENT_LIST_DIR}/compiler-instrument.cpp)

    # don't allow in-source instrumentation
    target_compile_definitions(timemory-compiler-instrument-${LIB_TYPE} INTERFACE
        DISABLE_TIMEMORY TIMEMORY_COMPILER_INSTRUMENTATION)

    target_link_libraries(timemory-compiler-instrument-${LIB_TYPE} PUBLIC ${dl_LIBRARY})

    target_link_libraries(timemory-compiler-instrument-${LIB_TYPE} PRIVATE
        ${_LIBS}
        timemory-headers
        timemory-compile-options
        timemory-hidden-visibility)

    target_compile_options(timemory-compiler-instrument-${LIB_TYPE} INTERFACE
        $<$<COMPILE_LANGUAGE:C>:-g>
        $<$<COMPILE_LANGUAGE:CXX>:-g>      
        $<$<COMPILE_LANGUAGE:C>:-fno-omit-frame-pointer>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>      
        $<$<COMPILE_LANGUAGE:C>:-fno-optimize-sibling-calls>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-optimize-sibling-calls>      
        $<$<COMPILE_LANGUAGE:C>:-finstrument-functions>
        $<$<COMPILE_LANGUAGE:CXX>:-finstrument-functions>
    )
      
    target_compile_options(timemory-compiler-instrument-${LIB_TYPE} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>)

    install(TARGETS timemory-compiler-instrument-${LIB_TYPE}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/compiler)

    # build tree
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            compiler/libtimemory-compiler-instrument.${CMAKE_SHARED_LIBRARY_SUFFIX}
            libtimemory-compiler-instrument.${CMAKE_SHARED_LIBRARY_SUFFIX}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # install tree
    install(CODE "
EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        timemory/compiler/libtimemory-compiler-instrument.${CMAKE_SHARED_LIBRARY_SUFFIX}
        libtimemory-compiler-instrument.${CMAKE_SHARED_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    ")

endforeach()
