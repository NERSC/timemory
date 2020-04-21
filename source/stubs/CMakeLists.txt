
if(WIN32)
    return()
endif()

foreach(LIB_TYPE shared static)

    if("${LIB_TYPE}" STREQUAL "shared")
        set(LIB_PARAMS PIC TYPE SHARED)
    elseif("${LIB_TYPE}" STREQUAL "static")
        set(LIB_PARAMS TYPE STATIC)
    endif()

    set(LIB_TARGET timemory-stubs-${LIB_TYPE})
    string(TOUPPER "${LIB_TYPE}" UC_LIB_TYPE)

    build_library(
        NO_CACHE_LIST
        ${LIB_PARAMS}
        TARGET_NAME         timemory-stubs-${LIB_TYPE}
        OUTPUT_NAME         timemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}/stubs
        SOURCES             ${CMAKE_CURRENT_LIST_DIR}/library.cpp)

    target_link_libraries(timemory-stubs-${LIB_TYPE} PUBLIC
        timemory-headers
        timemory-vector
        timemory-external-${LIB_TYPE}
        timemory-extern
        ${TIMEMORY_${UC_LIB_TYPE}_GLOBAL_LIBRARIES}
        ${TIMEMORY_${UC_LIB_TYPE}_EXTERN_LIBRARIES}
        ${TIMEMORY_${UC_LIB_TYPE}_COMPONENT_LIBRARIES}
        ${TIMEMORY_${UC_LIB_TYPE}_USER_BUNDLE_LIBRARIES})

    target_compile_definitions(timemory-stubs-${LIB_TYPE} PUBLIC TIMEMORY_USE_EXTERN)

    install(TARGETS timemory-stubs-${LIB_TYPE}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/timemory/stubs)

endforeach()
