
set_property(GLOBAL APPEND PROPERTY
    BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})

get_property(_CMAKE_DEFINES GLOBAL PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES)
list(SORT _CMAKE_DEFINES)

set(${PROJECT_NAME}_CMAKE_DEFINES)
foreach(_DEF ${_CMAKE_DEFINES})
    set(${PROJECT_NAME}_CMAKE_DEFINES "${${PROJECT_NAME}_CMAKE_DEFINES}\n#cmakedefine ${_DEF}")
endforeach()
set(${PROJECT_NAME}_CMAKE_DEFINES "${${PROJECT_NAME}_CMAKE_DEFINES}\n")

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/defines.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/defines.h.in @ONLY)

configure_file(${CMAKE_CURRENT_BINARY_DIR}/defines.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/defines.h @ONLY)

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

if(timemory_MASTER_PROJECT)
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/defines.h
            ${CMAKE_CURRENT_BINARY_DIR}/version.h
        DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR}/timemory)
else()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_BINARY_DIR}/defines.h
            ${CMAKE_CURRENT_LIST_DIR}/defines.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        ERROR_VARIABLE ERR_MSG
        RESULT_VARIABLE RET
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_BINARY_DIR}/version.h
            ${CMAKE_CURRENT_LIST_DIR}/version.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        ERROR_VARIABLE ERR_MSG
        RESULT_VARIABLE RET
    )
    install(
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/defines.h
            ${CMAKE_CURRENT_LIST_DIR}/version.h
        DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR}/timemory)
endif()

file(GLOB_RECURSE header_files
    ${CMAKE_CURRENT_SOURCE_DIR}/utility/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hash/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/environment/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/settings/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plotting/*.hpp)
file(GLOB_RECURSE source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/utility/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hash/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/environment/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/settings/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plotting/*.cpp)

build_intermediate_library(
    FORCE_OBJECT
    USE_INTERFACE
    INSTALL_SOURCE
    NAME                core
    TARGET              core
    CATEGORY            GLOBAL
    FOLDER              components
    HEADERS             ${header_files}
    SOURCES             ${source_files})

# core
#add_subdirectory(utility)
#add_subdirectory(hash)
#add_subdirectory(environment)
#add_subdirectory(settings)
#add_subdirectory(plotting)

# global
add_subdirectory(manager)
add_subdirectory(ert)
add_subdirectory(config)
add_subdirectory(storage)
add_subdirectory(operations)
add_subdirectory(backends)
add_subdirectory(variadic)

# components
add_subdirectory(components)

add_subdirectory(containers)
add_subdirectory(runtime)

# add_subdirectory(compat)
# add_subdirectory(data)
# add_subdirectory(general)
# add_subdirectory(mpl)
# add_subdirectory(utility)

