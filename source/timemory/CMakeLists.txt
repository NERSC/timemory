
set_property(GLOBAL APPEND PROPERTY
    BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})

# subdirs
add_subdirectory(hash)
add_subdirectory(environment)
add_subdirectory(settings)
add_subdirectory(plotting)
add_subdirectory(manager)
add_subdirectory(config)
add_subdirectory(operations)
add_subdirectory(backends)

add_subdirectory(components)

add_subdirectory(containers)
add_subdirectory(storage)
add_subdirectory(runtime)
add_subdirectory(ert)

# add_subdirectory(compat)
# add_subdirectory(data)
# add_subdirectory(general)
# add_subdirectory(mpl)
# add_subdirectory(utility)
# add_subdirectory(variadic)
