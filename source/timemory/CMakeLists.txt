
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
    ${CMAKE_CURRENT_LIST_DIR}/defines.h.in @ONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/defines.h.in
    ${CMAKE_CURRENT_LIST_DIR}/defines.h @ONLY)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/defines.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/timemory)

# subdirs
add_subdirectory(utility)
add_subdirectory(hash)
add_subdirectory(environment)
add_subdirectory(settings)
add_subdirectory(plotting)
add_subdirectory(manager)
add_subdirectory(config)
add_subdirectory(operations)
add_subdirectory(backends)
add_subdirectory(variadic)

add_subdirectory(components)

add_subdirectory(containers)
add_subdirectory(runtime)
add_subdirectory(storage)
add_subdirectory(ert)

# add_subdirectory(compat)
# add_subdirectory(data)
# add_subdirectory(general)
# add_subdirectory(mpl)
# add_subdirectory(utility)

