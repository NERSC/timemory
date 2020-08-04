
if(WIN32)
    return()
endif()

foreach(LIB_TYPE shared static)

    set(LIB_TARGET timemory-jump-${LIB_TYPE})
    string(TOUPPER "${LIB_TYPE}" UC_LIB_TYPE)

    build_library(
        PIC
        NO_CACHE_LIST
        TYPE                ${UC_LIB_TYPE}
        TARGET_NAME         timemory-jump-${LIB_TYPE}
        OUTPUT_NAME         timemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}/jump
        SOURCES             ${CMAKE_CURRENT_LIST_DIR}/library.cpp)

    target_compile_definitions(timemory-jump-${LIB_TYPE} INTERFACE TIMEMORY_USE_EXTERN)

    target_link_libraries(timemory-jump-${LIB_TYPE} PRIVATE
        timemory-headers
        timemory-compile-options)

    if(TARGET timemory-environment-${LIB_TYPE})
        target_link_libraries(timemory-jump-${LIB_TYPE} PRIVATE
            timemory-environment-${LIB_TYPE})
    endif()

    install(TARGETS timemory-jump-${LIB_TYPE}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/jump)

    # build tree
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            jump/libtimemory.${CMAKE_SHARED_LIBRARY_SUFFIX}
            libtimemory-jump.${CMAKE_SHARED_LIBRARY_SUFFIX}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # install tree
    install(CODE "
EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        timemory/jump/libtimemory.${CMAKE_SHARED_LIBRARY_SUFFIX}
        libtimemory-jump.${CMAKE_SHARED_LIBRARY_SUFFIX}
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    ")

endforeach()
