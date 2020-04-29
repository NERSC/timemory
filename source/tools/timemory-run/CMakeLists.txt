
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

#----------------------------------------------------------------------------------------#
# Build and install timemory-run
#
if(TIMEMORY_USE_DYNINST)

    file(GLOB_RECURSE EXCLUDE_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/source/*.c
        ${PROJECT_SOURCE_DIR}/source/*.cu
        ${PROJECT_SOURCE_DIR}/source/*.cpp
        ${PROJECT_SOURCE_DIR}/external/*.c
        ${PROJECT_SOURCE_DIR}/external/*.C
        ${PROJECT_SOURCE_DIR}/external/*.cu
        ${PROJECT_SOURCE_DIR}/external/*.cc
        ${PROJECT_SOURCE_DIR}/external/*.cxx
        ${PROJECT_SOURCE_DIR}/external/*.cpp
    )

    foreach(_FILE ${EXCLUDE_SOURCE_FILES})
        if(NOT "${_FILE}" MATCHES "/tests/" AND
            NOT "${_FILE}" MATCHES "/test/" AND
            NOT "${_FILE}" MATCHES "/unittests/" AND
            NOT "${_FILE}" MATCHES "google-test" AND
            NOT "${_FILE}" MATCHES "/example/" AND
            NOT "${_FILE}" MATCHES "/examples/" AND
            NOT "${_FILE}" MATCHES "/doc/" AND
            NOT "${_FILE}" MATCHES "/docs/")
            get_filename_component(_FILE "${_FILE}" NAME)
            list(APPEND _EXCLUDE ${_FILE})
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _EXCLUDE)
    set(EXCLUDE_SOURCE_FILES ${_EXCLUDE})
    unset(_EXCLUDE)
    string(REPLACE ";" "|" EXCLUDE_SOURCE_FILES "${EXCLUDE_SOURCE_FILES}")
    configure_file(${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp.in
        ${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp @ONLY)

    add_executable(timemory-run timemory-run.cpp timemory-run-details.cpp
        ${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp)

    target_include_directories(timemory-run PRIVATE ${CMAKE_CURRENT_LIST_DIR})

    target_link_libraries(timemory-run PRIVATE
        timemory-headers
        timemory-dyninst
        timemory-arch
        timemory-compile-options
        timemory-roofline-options
    )

    set_target_properties(timemory-run PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON)

    install(TARGETS timemory-run DESTINATION bin)

    file(GLOB COLLECTION_FILES "${CMAKE_CURRENT_LIST_DIR}/collections/*")
    foreach(_FILE ${COLLECTION_FILES})
        install(FILES ${_FILE} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/timemory/collections)
    endforeach()
endif()
