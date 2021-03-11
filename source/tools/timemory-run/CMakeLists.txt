
if(WIN32)
    return()
endif()

if(NOT TIMEMORY_BUILD_TOOLS AND NOT TIMEMORY_BUILD_DYNINST_TOOLS)
  set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

set(CMAKE_BUILD_TYPE RelWithDebInfo)

#----------------------------------------------------------------------------------------#
# Build and install timemory-run
#
if(TIMEMORY_USE_DYNINST)

    if(NOT DEFINED EXCLUDE_SOURCE_FILES OR "${EXCLUDE_SOURCE_FILES}" STREQUAL "")

        unset(_EXCLUDE_SOURCE_FILES CACHE)

        file(GLOB_RECURSE _EXCLUDE_SOURCE_FILES
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

        foreach(_FILE ${_EXCLUDE_SOURCE_FILES})
            if(NOT "${_FILE}" MATCHES "/tests/" AND
                    NOT "${_FILE}" MATCHES "/test/" AND
                    NOT "${_FILE}" MATCHES "/unittests/" AND
                    NOT "${_FILE}" MATCHES "google-test" AND
                    NOT "${_FILE}" MATCHES "llvm-ompt" AND
                    NOT "${_FILE}" MATCHES "/example/" AND
                    NOT "${_FILE}" MATCHES "/examples/" AND
                    NOT "${_FILE}" MATCHES "/doc/" AND
                    NOT "${_FILE}" MATCHES "/docs/")
                get_filename_component(_FILE "${_FILE}" NAME)
                list(APPEND _EXCLUDE_FILES ${_FILE})
            endif()
        endforeach()

        list(REMOVE_DUPLICATES _EXCLUDE_FILES)
        string(REPLACE ";" "|" _EXCLUDE_FILES "${_EXCLUDE_FILES}")
        set(EXCLUDE_SOURCE_FILES "${_EXCLUDE_FILES}" CACHE INTERNAL "Excluded source files")

    endif()

    configure_file(${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp.in
        ${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp @ONLY)

    add_executable(timemory-run ${_EXCLUDE}
        ${CMAKE_CURRENT_LIST_DIR}/timemory-run.cpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory-run.hpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory-run-details.cpp
        ${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp)

    add_target_flag_if_avail(timemory-run "-Wno-unused-function")

    target_link_libraries(timemory-run PRIVATE
        timemory::timemory-dmp
        timemory::timemory-headers
        timemory::timemory-dyninst
        timemory::timemory-compile-options
        timemory::timemory-develop-options)

    target_include_directories(timemory-run PRIVATE ${CMAKE_CURRENT_LIST_DIR})

    set_target_properties(timemory-run PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

    install(
        TARGETS     timemory-run
        DESTINATION bin
        OPTIONAL)

    add_custom_target(timemory-run-extra
        ${CMAKE_COMMAND}
            -D INPUT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}
            -D EXE_BINARY_DIR=${PROJECT_BINARY_DIR}
            -D FILE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/collections
            -P ${CMAKE_CURRENT_LIST_DIR}/Generate.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating timemory-inst and collection files for install...")

    add_dependencies(timemory-run timemory-run-extra)

    file(GLOB COLLECTION_FILES "${CMAKE_CURRENT_LIST_DIR}/collections/*")
    # the configured collection files
    string(REPLACE "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
        COLLECTION_FILES "${COLLECTION_FILES}")

    # these will only be installed if timemory-run is built
    install(
        PROGRAMS    ${PROJECT_BINARY_DIR}/timemory-instr
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        OPTIONAL)

    foreach(_FILE ${COLLECTION_FILES})
        install(FILES ${_FILE}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/timemory/collections
            OPTIONAL)
    endforeach()
endif()
