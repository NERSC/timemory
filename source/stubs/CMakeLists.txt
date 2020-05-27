
if(WIN32)
    return()
endif()

foreach(LIB_TYPE shared static)

    set(LIB_TARGET timemory-stubs-${LIB_TYPE})
    string(TOUPPER "${LIB_TYPE}" UC_LIB_TYPE)

    build_library(
        PIC
        NO_CACHE_LIST
        TYPE                ${UC_LIB_TYPE}
        TARGET_NAME         timemory-stubs-${LIB_TYPE}
        OUTPUT_NAME         timemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}/stubs
        SOURCES             ${CMAKE_CURRENT_LIST_DIR}/library.cpp)

    target_compile_definitions(timemory-stubs-${LIB_TYPE} PUBLIC TIMEMORY_USE_EXTERN)

    target_link_libraries(timemory-stubs-${LIB_TYPE} PRIVATE
        timemory-headers
        timemory-compile-options
        timemory-environment-${LIB_TYPE})

    install(TARGETS timemory-stubs-${LIB_TYPE}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/stubs)

    install(CODE "
EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E create_symlink timemory/stubs/libtimemory.so libtimemory-stubs.so
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    ")

endforeach()
