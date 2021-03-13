
if(WIN32)
    return()
endif()

if(NOT timemory_MAIN_PROJECT OR NOT TIMEMORY_BUILD_STUBS OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

foreach(LIB_TYPE shared static)

    set(LIB_TARGET timemory-stubs-${LIB_TYPE})
    string(TOUPPER "${LIB_TYPE}" UC_LIB_TYPE)

    build_library(
        PIC
        NO_CACHE_LIST
        ${_EXCLUDE}
        TYPE                ${UC_LIB_TYPE}
        TARGET_NAME         timemory-stubs-${LIB_TYPE}
        OUTPUT_NAME         timemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}/stubs
        SOURCES             ${CMAKE_CURRENT_LIST_DIR}/library.cpp)

    target_compile_definitions(timemory-stubs-${LIB_TYPE} INTERFACE TIMEMORY_USE_EXTERN)

    target_link_libraries(timemory-stubs-${LIB_TYPE} PRIVATE
        timemory::timemory-headers
        timemory::timemory-compile-options)

    if(TARGET timemory-environment-${LIB_TYPE})
        target_link_libraries(timemory-stubs-${LIB_TYPE} PRIVATE
            timemory-environment-${LIB_TYPE})
    endif()

    install(TARGETS timemory-stubs-${LIB_TYPE}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/stubs
        EXPORT timemory-library-depends
        OPTIONAL)

    # build tree
    add_custom_command(
        OUTPUT libtimemory-stubs${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
        MAIN_DEPENDENCY timemory-stubs-${LIB_TYPE}
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            stubs/libtimemory${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
            libtimemory-stubs${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT "Creating symbolic link libtimemory-stubs${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}")

    # install tree
    install(CODE "
if(EXISTS \"${PROJECT_BINARY_DIR}/stubs/libtimemory${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}\")
    EXECUTE_PROCESS(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            timemory/stubs/libtimemory${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
            libtimemory-stubs${CMAKE_${UC_LIB_TYPE}_LIBRARY_SUFFIX}
        WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif()
    "
    OPTIONAL)

endforeach()
