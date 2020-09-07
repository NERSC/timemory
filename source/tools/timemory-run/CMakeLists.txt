
if(WIN32)
    return()
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

#----------------------------------------------------------------------------------------#
# Build and install timemory-run
#
if(TIMEMORY_USE_DYNINST)

    if(NOT DEFINED EXCLUDE_SOURCE_FILES OR "${EXCLUDE_SOURCE_FILES}" STREQUAL "")

        unset(EXCLUDE_SOURCE_FILES CACHE)

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
                    NOT "${_FILE}" MATCHES "llvm-ompt" AND
                    NOT "${_FILE}" MATCHES "/example/" AND
                    NOT "${_FILE}" MATCHES "/examples/" AND
                    NOT "${_FILE}" MATCHES "/doc/" AND
                    NOT "${_FILE}" MATCHES "/docs/")
                get_filename_component(_FILE "${_FILE}" NAME)
                list(APPEND _EXCLUDE ${_FILE})
            endif()
        endforeach()

        list(REMOVE_DUPLICATES _EXCLUDE)
        string(REPLACE ";" "|" _EXCLUDE "${_EXCLUDE}")
        set(EXCLUDE_SOURCE_FILES "${_EXCLUDE}" CACHE INTERNAL "Excluded source files")

    endif()

    configure_file(${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp.in
        ${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp @ONLY)

    add_library(timemory-run-interface INTERFACE)

    add_target_flag_if_avail(timemory-run-interface "-Wno-unused-function")

    if(TARGET timemory::timemory-core-static)
        set(_LIBRARY timemory::timemory-core-static)
    elseif(TARGET timemory::timemory-core-shared)
        set(_LIBRARY timemory::timemory-core-shared)
    endif()

    target_link_libraries(timemory-run-interface INTERFACE
        timemory-headers
        timemory-dyninst
        timemory-dmp
        timemory-arch
        timemory-compile-options
        timemory-roofline-options
        ${_LIBRARY})

    add_executable(timemory-run
        ${CMAKE_CURRENT_LIST_DIR}/timemory-run.cpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory-run.hpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory-run-fork.cpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory-run-fork.hpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory-run-details.cpp
        ${CMAKE_CURRENT_LIST_DIR}/generated/timemory-run-regex.cpp)

    target_include_directories(timemory-run PRIVATE ${CMAKE_CURRENT_LIST_DIR})

    target_link_libraries(timemory-run PRIVATE timemory-run-interface)

    set_target_properties(timemory-run PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

    install(TARGETS timemory-run DESTINATION bin)

    configure_file(${CMAKE_CURRENT_LIST_DIR}/generated/timemory-instr.sh
        ${PROJECT_BINARY_DIR}/timemory-instr COPYONLY)

    install(FILES ${PROJECT_BINARY_DIR}/timemory-instr
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        PERMISSIONS
        OWNER_EXECUTE OWNER_READ OWNER_WRITE
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ)

    file(GLOB COLLECTION_FILES "${CMAKE_CURRENT_LIST_DIR}/collections/*")
    foreach(_FILE ${COLLECTION_FILES})
        install(FILES ${_FILE} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/timemory/collections)
    endforeach()
endif()
